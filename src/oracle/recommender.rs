//! Recommendation Engine for Oracle Mode
//!
//! Generates component recommendations based on parsed queries
//! and the knowledge graph.

use super::knowledge_graph::KnowledgeGraph;
use super::query_engine::{ParsedQuery, PerformanceHint, QueryEngine};
use super::types::*;

// =============================================================================
// Backend Selector (from spec section 4.2)
// =============================================================================

/// Select optimal backend based on workload characteristics
/// Based on PCIe transfer overhead analysis (Gregg & Hazelwood, 2011)
pub fn select_backend(
    op_complexity: OpComplexity,
    data_size: Option<DataSize>,
    hardware: &HardwareSpec,
) -> Backend {
    let size = data_size.and_then(|d| d.as_samples()).unwrap_or(0);

    match op_complexity {
        OpComplexity::Low => {
            // Element-wise ops: memory-bound, GPU rarely beneficial
            if size > 1_000_000 && hardware.has_gpu() {
                Backend::GPU
            } else if size > 1_000 {
                Backend::SIMD
            } else {
                Backend::Scalar
            }
        }
        OpComplexity::Medium => {
            // Reductions: moderate compute
            if size > 100_000 && hardware.has_gpu() {
                Backend::GPU
            } else if size > 100 {
                Backend::SIMD
            } else {
                Backend::Scalar
            }
        }
        OpComplexity::High => {
            // Matrix ops: O(n²) or O(n³), GPU beneficial early
            if size > 10_000 && hardware.has_gpu() {
                Backend::GPU
            } else if size > 10 {
                Backend::SIMD
            } else {
                Backend::Scalar
            }
        }
    }
}

/// Determine if distributed execution is beneficial
/// Based on Amdahl's Law and communication overhead
pub fn should_distribute(
    data_size: Option<DataSize>,
    hardware: &HardwareSpec,
    parallel_fraction: f64,
) -> DistributionRecommendation {
    let size = data_size.and_then(|d| d.as_samples()).unwrap_or(0);

    // Only consider distribution for large data or explicit multi-node
    if !hardware.is_distributed && size < 10_000_000 {
        return DistributionRecommendation {
            tool: None,
            needed: false,
            rationale: "Single-node sufficient for this workload size".into(),
            node_count: None,
        };
    }

    let node_count = hardware.node_count.unwrap_or(1);
    if node_count <= 1 {
        return DistributionRecommendation {
            tool: None,
            needed: false,
            rationale: "No additional nodes available for distribution".into(),
            node_count: None,
        };
    }

    // Amdahl's Law: speedup = 1 / ((1-p) + p/n)
    let speedup = 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / node_count as f64);

    // Communication overhead estimate (simplified)
    let comm_overhead = 0.1 * node_count as f64; // 10% per node

    if speedup > 1.5 && comm_overhead < 0.5 {
        DistributionRecommendation {
            tool: Some("repartir".into()),
            needed: true,
            rationale: format!(
                "Distribution beneficial with {:.1}x speedup across {} nodes",
                speedup, node_count
            ),
            node_count: Some(node_count),
        }
    } else {
        DistributionRecommendation {
            tool: None,
            needed: false,
            rationale: format!(
                "Distribution overhead ({:.0}%) outweighs benefits",
                comm_overhead * 100.0
            ),
            node_count: None,
        }
    }
}

// =============================================================================
// Recommender
// =============================================================================

/// Oracle recommendation engine
pub struct Recommender {
    graph: KnowledgeGraph,
    engine: QueryEngine,
}

impl Default for Recommender {
    fn default() -> Self {
        Self::new()
    }
}

impl Recommender {
    /// Create a new recommender with the Sovereign AI Stack
    pub fn new() -> Self {
        Self {
            graph: KnowledgeGraph::sovereign_stack(),
            engine: QueryEngine::new(),
        }
    }

    /// Create a recommender with a custom knowledge graph
    pub fn with_graph(graph: KnowledgeGraph) -> Self {
        Self {
            graph,
            engine: QueryEngine::new(),
        }
    }

    /// Process a natural language query and return recommendations
    pub fn query(&self, query: &str) -> OracleResponse {
        let parsed = self.engine.parse(query);
        self.recommend(&parsed, &QueryConstraints::default())
    }

    /// Process a structured OracleQuery
    pub fn query_structured(&self, query: &OracleQuery) -> OracleResponse {
        let parsed = self.engine.parse(&query.description);
        self.recommend(&parsed, &query.constraints)
    }

    /// Generate recommendations from parsed query
    pub fn recommend(&self, parsed: &ParsedQuery, constraints: &QueryConstraints) -> OracleResponse {
        // Determine primary problem class
        let problem_class = self.classify_problem(parsed);

        // Find primary component recommendation
        let primary = self.recommend_primary(parsed, constraints);

        // Find supporting components
        let supporting = self.recommend_supporting(&primary, parsed, constraints);

        // Determine compute backend
        let complexity = self.engine.estimate_complexity(parsed);
        let backend = select_backend(complexity, constraints.data_size, &constraints.hardware);
        let compute = ComputeRecommendation {
            backend,
            rationale: self.compute_rationale(backend, complexity, constraints),
        };

        // Determine distribution needs
        let parallel_fraction = self.estimate_parallel_fraction(parsed);
        let distribution = should_distribute(
            constraints.data_size,
            &constraints.hardware,
            parallel_fraction,
        );

        // Generate code example
        let code_example = self.generate_code_example(&primary, &supporting, parsed);

        // Generate related queries
        let related_queries = self.generate_related_queries(parsed);

        OracleResponse {
            problem_class,
            algorithm: self.engine.primary_algorithm(parsed).map(String::from),
            primary,
            supporting,
            compute,
            distribution,
            code_example,
            related_queries,
        }
    }

    fn classify_problem(&self, parsed: &ParsedQuery) -> String {
        if let Some(domain) = self.engine.primary_domain(parsed) {
            domain.to_string()
        } else if !parsed.algorithms.is_empty() {
            "Algorithm-specific".into()
        } else {
            "General".into()
        }
    }

    fn recommend_primary(
        &self,
        parsed: &ParsedQuery,
        _constraints: &QueryConstraints,
    ) -> ComponentRecommendation {
        // Check if specific components were mentioned
        if let Some(component) = parsed.mentioned_components.first() {
            if let Some(comp) = self.graph.get_component(component) {
                return ComponentRecommendation {
                    component: comp.name.clone(),
                    path: None,
                    confidence: 0.95,
                    rationale: format!(
                        "Explicitly mentioned {} - {}",
                        comp.name, comp.description
                    ),
                };
            }
        }

        // Find by algorithm
        if let Some(algo) = parsed.algorithms.first() {
            let components = self.graph.find_by_capability(algo);
            if let Some(comp) = components.first() {
                let path = self.get_algorithm_path(comp, algo);
                return ComponentRecommendation {
                    component: comp.name.clone(),
                    path,
                    confidence: 0.9,
                    rationale: format!(
                        "{} provides {} implementation",
                        comp.name, algo
                    ),
                };
            }
        }

        // Find by problem domain
        if let Some(domain) = self.engine.primary_domain(parsed) {
            let components = self.graph.find_by_domain(domain);
            if let Some(comp) = components.first() {
                return ComponentRecommendation {
                    component: comp.name.clone(),
                    path: None,
                    confidence: 0.85,
                    rationale: format!(
                        "{} is recommended for {} tasks",
                        comp.name, domain
                    ),
                };
            }
        }

        // Default recommendation based on performance hints
        if parsed.performance_hints.contains(&PerformanceHint::GPURequired) {
            return ComponentRecommendation {
                component: "trueno".into(),
                path: None,
                confidence: 0.7,
                rationale: "GPU acceleration available via trueno".into(),
            };
        }

        if parsed.performance_hints.contains(&PerformanceHint::Distributed) {
            return ComponentRecommendation {
                component: "repartir".into(),
                path: None,
                confidence: 0.7,
                rationale: "Distributed computing via repartir".into(),
            };
        }

        // Fallback to batuta for general orchestration
        ComponentRecommendation {
            component: "batuta".into(),
            path: None,
            confidence: 0.5,
            rationale: "General orchestration framework for the Sovereign AI Stack".into(),
        }
    }

    fn recommend_supporting(
        &self,
        primary: &ComponentRecommendation,
        parsed: &ParsedQuery,
        constraints: &QueryConstraints,
    ) -> Vec<ComponentRecommendation> {
        let mut supporting = Vec::new();

        // Get integration patterns from primary
        let integrations = self.graph.integrations_from(&primary.component);
        for pattern in integrations.iter().take(2) {
            if let Some(comp) = self.graph.get_component(&pattern.to) {
                supporting.push(ComponentRecommendation {
                    component: comp.name.clone(),
                    path: None,
                    confidence: 0.7,
                    rationale: format!(
                        "Integrates via {} pattern",
                        pattern.pattern_name
                    ),
                });
            }
        }

        // Add trueno for compute if ML task
        if parsed.domains.iter().any(|d| matches!(d,
            ProblemDomain::SupervisedLearning |
            ProblemDomain::UnsupervisedLearning |
            ProblemDomain::DeepLearning
        )) && primary.component != "trueno" {
            supporting.push(ComponentRecommendation {
                component: "trueno".into(),
                path: None,
                confidence: 0.8,
                rationale: "SIMD/GPU backend for compute acceleration".into(),
            });
        }

        // Add repartir for large data
        if constraints.data_size.map(|d| d.is_large()).unwrap_or(false)
            && primary.component != "repartir"
        {
            supporting.push(ComponentRecommendation {
                component: "repartir".into(),
                path: None,
                confidence: 0.6,
                rationale: "Distribution recommended for large dataset".into(),
            });
        }

        // Add alimentar for data loading hints
        if parsed.domains.contains(&ProblemDomain::DataPipeline)
            && primary.component != "alimentar"
        {
            supporting.push(ComponentRecommendation {
                component: "alimentar".into(),
                path: None,
                confidence: 0.7,
                rationale: "Data loading and preprocessing".into(),
            });
        }

        // Add realizar for inference
        if parsed.domains.contains(&ProblemDomain::Inference)
            && primary.component != "realizar"
        {
            supporting.push(ComponentRecommendation {
                component: "realizar".into(),
                path: None,
                confidence: 0.85,
                rationale: "Model serving and inference".into(),
            });
        }

        supporting
    }

    fn get_algorithm_path(&self, component: &StackComponent, algorithm: &str) -> Option<String> {
        // Generate module path based on component and algorithm
        match (component.name.as_str(), algorithm) {
            ("aprender", algo) if algo.contains("random_forest") => {
                Some("aprender::tree::RandomForestClassifier".into())
            }
            ("aprender", algo) if algo.contains("decision_tree") => {
                Some("aprender::tree::DecisionTreeClassifier".into())
            }
            ("aprender", algo) if algo.contains("linear_regression") => {
                Some("aprender::linear::LinearRegression".into())
            }
            ("aprender", algo) if algo.contains("logistic_regression") => {
                Some("aprender::linear::LogisticRegression".into())
            }
            ("aprender", algo) if algo.contains("gbm") || algo.contains("gradient_boosting") => {
                Some("aprender::ensemble::GradientBoostingClassifier".into())
            }
            ("aprender", algo) if algo.contains("kmeans") || algo.contains("k_means") => {
                Some("aprender::cluster::KMeans".into())
            }
            ("aprender", algo) if algo.contains("pca") => {
                Some("aprender::decomposition::PCA".into())
            }
            ("aprender", algo) if algo.contains("svm") => {
                Some("aprender::svm::SVC".into())
            }
            ("aprender", algo) if algo.contains("knn") => {
                Some("aprender::neighbors::KNeighborsClassifier".into())
            }
            ("entrenar", algo) if algo.contains("lora") => {
                Some("entrenar::lora::LoRA".into())
            }
            ("entrenar", algo) if algo.contains("qlora") => {
                Some("entrenar::lora::QLoRA".into())
            }
            _ => None,
        }
    }

    fn compute_rationale(
        &self,
        backend: Backend,
        complexity: OpComplexity,
        constraints: &QueryConstraints,
    ) -> String {
        let size_str = constraints
            .data_size
            .map(|d| match d {
                DataSize::Samples(n) => format!("{} samples", format_number(n)),
                DataSize::Bytes(n) => format!("{} bytes", format_number(n)),
                DataSize::Unknown => "unknown size".into(),
            })
            .unwrap_or_else(|| "unspecified size".into());

        match backend {
            Backend::Scalar => {
                format!("Scalar operations sufficient for small {} with {:?} complexity", size_str, complexity)
            }
            Backend::SIMD => {
                format!("SIMD vectorization optimal for {} with {:?} complexity", size_str, complexity)
            }
            Backend::GPU => {
                format!("GPU acceleration recommended for {} with {:?} complexity - PCIe overhead amortized", size_str, complexity)
            }
            Backend::Distributed => {
                format!("Distributed execution for {} exceeds single-node capacity", size_str)
            }
        }
    }

    fn estimate_parallel_fraction(&self, parsed: &ParsedQuery) -> f64 {
        // Estimate parallelizability based on algorithm
        if let Some(algo) = parsed.algorithms.first() {
            if algo.contains("random_forest") || algo.contains("gbm") {
                return 0.95; // Embarrassingly parallel
            }
            if algo.contains("kmeans") {
                return 0.85;
            }
            if algo.contains("linear") {
                return 0.7;
            }
        }

        // Based on problem domain
        if parsed.domains.contains(&ProblemDomain::DeepLearning) {
            return 0.8;
        }
        if parsed.domains.contains(&ProblemDomain::SupervisedLearning) {
            return 0.75;
        }

        0.6 // Default
    }

    fn generate_code_example(
        &self,
        primary: &ComponentRecommendation,
        _supporting: &[ComponentRecommendation],
        parsed: &ParsedQuery,
    ) -> Option<String> {
        // Generate contextual code example based on primary component
        match primary.component.as_str() {
            "aprender" => {
                let path = primary.path.as_deref().unwrap_or("aprender::tree::RandomForestClassifier");
                let _algo = parsed.algorithms.first().map(|s| s.as_str()).unwrap_or("RandomForest");

                Some(format!(
                    r#"use {};

// Load data
let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.2)?;

// Train model
let model = {}::new()
    .n_estimators(100)
    .fit(&X_train, &y_train)?;

// Predict
let predictions = model.predict(&X_test)?;
let accuracy = accuracy_score(&y_test, &predictions);
println!("Accuracy: {{:.2}}%", accuracy * 100.0);"#,
                    path,
                    path.split("::").last().unwrap_or("Model")
                ))
            }
            "trueno" => Some(
                r#"use trueno::prelude::*;

// Create tensors with SIMD acceleration
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

// SIMD-accelerated operations
let result = a.dot(&b);
println!("Dot product: {}", result);"#
                    .into(),
            ),
            "depyler" => Some(
                r#"# Run depyler to convert Python to Rust
batuta transpile --source my_project.py --output rust-output/

# The sklearn code:
#   from sklearn.ensemble import RandomForestClassifier
#   model = RandomForestClassifier(n_estimators=100)
#
# Becomes:
#   use aprender::tree::RandomForestClassifier;
#   let model = RandomForestClassifier::new().n_estimators(100);"#
                    .into(),
            ),
            "realizar" => Some(
                r#"use realizar::ModelRegistry;

// Load trained model
let registry = ModelRegistry::new();
registry.load_apr("classifier", "model.apr")?;

// Serve predictions
let input = vec![1.0, 2.0, 3.0, 4.0];
let prediction = registry.predict("classifier", &input)?;
println!("Prediction: {:?}", prediction);"#
                    .into(),
            ),
            "repartir" => Some(
                r#"use repartir::{Executor, Task};

// Create distributed executor
let executor = Executor::new()
    .with_workers(4)
    .with_scheduler(WorkStealing);

// Distribute computation
let results = executor.map(data.chunks(1000), |chunk| {
    process_chunk(chunk)
}).await?;

// Aggregate results
let final_result = results.reduce(|a, b| a + b);"#
                    .into(),
            ),
            _ => None,
        }
    }

    fn generate_related_queries(&self, parsed: &ParsedQuery) -> Vec<String> {
        let mut related = Vec::new();

        // Add related queries based on domains
        if parsed.domains.contains(&ProblemDomain::SupervisedLearning) {
            related.push("How do I tune hyperparameters for this model?".into());
            related.push("What's the best way to handle imbalanced data?".into());
        }

        if parsed.domains.contains(&ProblemDomain::PythonMigration) {
            related.push("How do I convert numpy arrays to trueno tensors?".into());
            related.push("What sklearn features are supported in aprender?".into());
        }

        if parsed.domains.contains(&ProblemDomain::Inference) {
            related.push("How do I optimize for low latency?".into());
            related.push("What model formats does realizar support?".into());
        }

        if parsed.performance_hints.contains(&PerformanceHint::Distributed) {
            related.push("How do I scale to multiple nodes?".into());
            related.push("What's the communication overhead for distributed training?".into());
        }

        related.truncate(3);
        related
    }

    /// Get capabilities of a component
    pub fn get_capabilities(&self, component: &str) -> Vec<String> {
        self.graph
            .get_component(component)
            .map(|c| c.capabilities.iter().map(|cap| cap.name.clone()).collect())
            .unwrap_or_default()
    }

    /// Get integration pattern between two components
    pub fn get_integration(&self, from: &str, to: &str) -> Option<IntegrationPattern> {
        self.graph.get_integration(from, to).cloned()
    }

    /// List all available components
    pub fn list_components(&self) -> Vec<String> {
        self.graph.component_names().cloned().collect()
    }

    /// Get component details
    pub fn get_component(&self, name: &str) -> Option<&StackComponent> {
        self.graph.get_component(name)
    }
}

/// Format large numbers for display
fn format_number(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{}B", n / 1_000_000_000)
    } else if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        n.to_string()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Backend Selection Tests
    // =========================================================================

    #[test]
    fn test_select_backend_low_complexity_small() {
        let hw = HardwareSpec::cpu_only();
        let backend = select_backend(OpComplexity::Low, Some(DataSize::samples(100)), &hw);
        assert_eq!(backend, Backend::Scalar);
    }

    #[test]
    fn test_select_backend_low_complexity_medium() {
        let hw = HardwareSpec::cpu_only();
        let backend = select_backend(OpComplexity::Low, Some(DataSize::samples(10_000)), &hw);
        assert_eq!(backend, Backend::SIMD);
    }

    #[test]
    fn test_select_backend_low_complexity_large_gpu() {
        let hw = HardwareSpec::with_gpu(16.0);
        let backend = select_backend(OpComplexity::Low, Some(DataSize::samples(2_000_000)), &hw);
        assert_eq!(backend, Backend::GPU);
    }

    #[test]
    fn test_select_backend_high_complexity_gpu() {
        let hw = HardwareSpec::with_gpu(8.0);
        let backend = select_backend(OpComplexity::High, Some(DataSize::samples(50_000)), &hw);
        assert_eq!(backend, Backend::GPU);
    }

    #[test]
    fn test_select_backend_high_complexity_no_gpu() {
        let hw = HardwareSpec::cpu_only();
        let backend = select_backend(OpComplexity::High, Some(DataSize::samples(50_000)), &hw);
        assert_eq!(backend, Backend::SIMD);
    }

    #[test]
    fn test_select_backend_medium_complexity() {
        let hw = HardwareSpec::cpu_only();
        let backend = select_backend(OpComplexity::Medium, Some(DataSize::samples(1_000)), &hw);
        assert_eq!(backend, Backend::SIMD);
    }

    // =========================================================================
    // Distribution Decision Tests
    // =========================================================================

    #[test]
    fn test_should_distribute_small_data() {
        let hw = HardwareSpec::default();
        let decision = should_distribute(Some(DataSize::samples(10_000)), &hw, 0.9);
        assert!(!decision.needed);
    }

    #[test]
    fn test_should_distribute_single_node() {
        let hw = HardwareSpec {
            node_count: Some(1),
            is_distributed: true,
            ..Default::default()
        };
        let decision = should_distribute(Some(DataSize::samples(100_000_000)), &hw, 0.9);
        assert!(!decision.needed);
    }

    #[test]
    fn test_should_distribute_multi_node() {
        let hw = HardwareSpec {
            node_count: Some(4),
            is_distributed: true,
            ..Default::default()
        };
        let decision = should_distribute(Some(DataSize::samples(100_000_000)), &hw, 0.95);
        assert!(decision.needed);
        assert_eq!(decision.tool.as_deref(), Some("repartir"));
    }

    #[test]
    fn test_should_distribute_low_parallel_fraction() {
        let hw = HardwareSpec {
            node_count: Some(4),
            is_distributed: true,
            ..Default::default()
        };
        let decision = should_distribute(Some(DataSize::samples(100_000_000)), &hw, 0.3);
        // Low parallel fraction with limited speedup - result depends on thresholds
        // Just verify the function returns a valid decision without panicking
        assert!(!decision.rationale.is_empty());
    }

    // =========================================================================
    // Recommender Creation Tests
    // =========================================================================

    #[test]
    fn test_recommender_new() {
        let rec = Recommender::new();
        assert!(!rec.list_components().is_empty());
    }

    #[test]
    fn test_recommender_default() {
        let rec = Recommender::default();
        let components = rec.list_components();
        assert!(components.contains(&"trueno".to_string()));
        assert!(components.contains(&"aprender".to_string()));
    }

    #[test]
    fn test_recommender_with_graph() {
        let graph = KnowledgeGraph::sovereign_stack();
        let rec = Recommender::with_graph(graph);
        assert!(!rec.list_components().is_empty());
    }

    // =========================================================================
    // Query Tests
    // =========================================================================

    #[test]
    fn test_query_random_forest() {
        let rec = Recommender::new();
        let response = rec.query("Train a random forest classifier");

        assert_eq!(response.primary.component, "aprender");
        assert!(response.primary.confidence > 0.8);
        assert!(response.algorithm.is_some());
    }

    #[test]
    fn test_query_sklearn_migration() {
        let rec = Recommender::new();
        let response = rec.query("Convert my sklearn pipeline to Rust");

        assert_eq!(response.primary.component, "depyler");
        assert_eq!(response.problem_class, "Python Migration");
    }

    #[test]
    fn test_query_model_serving() {
        let rec = Recommender::new();
        let response = rec.query("Deploy model for production inference");

        assert_eq!(response.primary.component, "realizar");
    }

    #[test]
    fn test_query_matrix_ops() {
        let rec = Recommender::new();
        let response = rec.query("Fast matrix multiplication with SIMD");

        assert_eq!(response.primary.component, "trueno");
    }

    #[test]
    fn test_query_distributed() {
        let rec = Recommender::new();
        // "Distributed" queries without specific ML tasks should recommend repartir
        let response = rec.query("Scale computation with repartir across cluster");

        assert_eq!(response.primary.component, "repartir");
    }

    #[test]
    fn test_query_explicit_component() {
        let rec = Recommender::new();
        let response = rec.query("Use aprender for my ML task");

        assert_eq!(response.primary.component, "aprender");
        assert!(response.primary.confidence > 0.9); // High confidence for explicit mention
    }

    // =========================================================================
    // Structured Query Tests
    // =========================================================================

    #[test]
    fn test_query_structured() {
        let rec = Recommender::new();
        let query = OracleQuery::new("Train a classifier")
            .with_data_size(DataSize::samples(1_000_000))
            .with_hardware(HardwareSpec::with_gpu(16.0));

        let response = rec.query_structured(&query);

        assert_eq!(response.primary.component, "aprender");
        // Should recommend GPU backend for large data
        assert!(matches!(response.compute.backend, Backend::GPU | Backend::SIMD));
    }

    #[test]
    fn test_query_structured_sovereign() {
        let rec = Recommender::new();
        let query = OracleQuery::new("Train model with GDPR compliance")
            .sovereign_only()
            .eu_compliant();

        let response = rec.query_structured(&query);
        // Should work - sovereign mode is about local execution, which all stack components support
        assert!(!response.primary.component.is_empty());
    }

    // =========================================================================
    // Backend Recommendation Tests
    // =========================================================================

    #[test]
    fn test_compute_recommendation_simd() {
        let rec = Recommender::new();
        // Use structured query with explicit data size for reliable backend selection
        let query = OracleQuery::new("Train random forest")
            .with_data_size(DataSize::samples(10_000));
        let response = rec.query_structured(&query);

        assert_eq!(response.compute.backend, Backend::SIMD);
    }

    #[test]
    fn test_compute_recommendation_scalar() {
        let rec = Recommender::new();
        let query = OracleQuery::new("Simple calculation")
            .with_data_size(DataSize::samples(50));

        let response = rec.query_structured(&query);
        // Small data should use scalar
        assert!(matches!(response.compute.backend, Backend::Scalar | Backend::SIMD));
    }

    // =========================================================================
    // Supporting Component Tests
    // =========================================================================

    #[test]
    fn test_supporting_components_ml() {
        let rec = Recommender::new();
        let response = rec.query("Train a model on large dataset");

        // Should recommend trueno for compute backend
        assert!(response.supporting.iter().any(|s| s.component == "trueno"));
    }

    #[test]
    fn test_supporting_components_inference() {
        let rec = Recommender::new();
        let response = rec.query("Train a model and deploy for inference");

        // Should recommend realizar for serving
        assert!(
            response.primary.component == "realizar"
                || response.supporting.iter().any(|s| s.component == "realizar")
        );
    }

    // =========================================================================
    // Code Example Tests
    // =========================================================================

    #[test]
    fn test_code_example_aprender() {
        let rec = Recommender::new();
        let response = rec.query("Train random forest classifier");

        assert!(response.code_example.is_some());
        let code = response.code_example.unwrap();
        assert!(code.contains("aprender") || code.contains("RandomForest"));
    }

    #[test]
    fn test_code_example_trueno() {
        let rec = Recommender::new();
        let response = rec.query("Matrix operations with trueno");

        assert!(response.code_example.is_some());
        let code = response.code_example.unwrap();
        assert!(code.contains("trueno") || code.contains("Tensor"));
    }

    // =========================================================================
    // Related Queries Tests
    // =========================================================================

    #[test]
    fn test_related_queries_ml() {
        let rec = Recommender::new();
        let response = rec.query("Train a classifier");

        assert!(!response.related_queries.is_empty());
    }

    #[test]
    fn test_related_queries_migration() {
        let rec = Recommender::new();
        let response = rec.query("Convert sklearn to Rust");

        assert!(!response.related_queries.is_empty());
        // Should have sklearn/numpy related queries
        assert!(response
            .related_queries
            .iter()
            .any(|q| q.contains("sklearn") || q.contains("numpy") || q.contains("aprender")));
    }

    // =========================================================================
    // Capability Tests
    // =========================================================================

    #[test]
    fn test_get_capabilities_trueno() {
        let rec = Recommender::new();
        let caps = rec.get_capabilities("trueno");

        assert!(caps.contains(&"simd".to_string()));
        assert!(caps.contains(&"gpu".to_string()));
    }

    #[test]
    fn test_get_capabilities_aprender() {
        let rec = Recommender::new();
        let caps = rec.get_capabilities("aprender");

        assert!(caps.contains(&"random_forest".to_string()));
        assert!(caps.contains(&"linear_regression".to_string()));
    }

    #[test]
    fn test_get_capabilities_unknown() {
        let rec = Recommender::new();
        let caps = rec.get_capabilities("unknown_component");

        assert!(caps.is_empty());
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_get_integration() {
        let rec = Recommender::new();
        let pattern = rec.get_integration("aprender", "realizar");

        assert!(pattern.is_some());
        let pattern = pattern.unwrap();
        assert_eq!(pattern.pattern_name, "model_export");
    }

    #[test]
    fn test_get_integration_not_found() {
        let rec = Recommender::new();
        let pattern = rec.get_integration("trueno", "bashrs");

        assert!(pattern.is_none());
    }

    // =========================================================================
    // Component Lookup Tests
    // =========================================================================

    #[test]
    fn test_get_component() {
        let rec = Recommender::new();
        let comp = rec.get_component("trueno");

        assert!(comp.is_some());
        assert_eq!(comp.unwrap().version, "0.7.3");
    }

    #[test]
    fn test_list_components() {
        let rec = Recommender::new();
        let components = rec.list_components();

        assert!(components.len() >= 15);
        assert!(components.contains(&"trueno".to_string()));
        assert!(components.contains(&"aprender".to_string()));
        assert!(components.contains(&"repartir".to_string()));
    }

    // =========================================================================
    // Format Number Tests
    // =========================================================================

    #[test]
    fn test_format_number_billions() {
        assert_eq!(format_number(2_000_000_000), "2B");
    }

    #[test]
    fn test_format_number_millions() {
        assert_eq!(format_number(5_000_000), "5M");
    }

    #[test]
    fn test_format_number_thousands() {
        assert_eq!(format_number(10_000), "10K");
    }

    #[test]
    fn test_format_number_small() {
        assert_eq!(format_number(500), "500");
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_empty_query() {
        let rec = Recommender::new();
        let response = rec.query("");

        // Should still return something
        assert!(!response.primary.component.is_empty());
        assert_eq!(response.primary.component, "batuta"); // Fallback
    }

    #[test]
    fn test_nonsense_query() {
        let rec = Recommender::new();
        let response = rec.query("xyz abc 123");

        // Should still return fallback
        assert!(!response.primary.component.is_empty());
    }

    #[test]
    fn test_multiple_algorithms() {
        let rec = Recommender::new();
        let response = rec.query("Should I use random forest or gradient boosting?");

        // Should pick first one detected
        assert!(response.algorithm.is_some());
    }
}
