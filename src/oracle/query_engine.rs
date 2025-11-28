//! Query Engine for Oracle Mode
//!
//! Parses natural language queries and extracts structured information
//! for the recommendation engine.

use super::types::*;
use std::collections::HashSet;

// =============================================================================
// Query Parser
// =============================================================================

/// Parsed query with extracted information
#[derive(Debug, Clone)]
pub struct ParsedQuery {
    /// Original query text
    pub original: String,
    /// Detected problem domains
    pub domains: Vec<ProblemDomain>,
    /// Detected algorithms/techniques
    pub algorithms: Vec<String>,
    /// Extracted keywords
    pub keywords: Vec<String>,
    /// Detected data size indicators
    pub data_size: Option<DataSize>,
    /// Performance requirements detected
    pub performance_hints: Vec<PerformanceHint>,
    /// Component mentions
    pub mentioned_components: Vec<String>,
}

/// Performance hint extracted from query
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceHint {
    LowLatency,
    HighThroughput,
    LowMemory,
    GPURequired,
    Distributed,
    EdgeDeployment,
    Sovereign,
}

/// Query parser for natural language queries
#[derive(Debug, Default)]
pub struct QueryParser {
    /// Known algorithm keywords
    algorithm_keywords: HashSet<String>,
    /// Problem domain keywords
    domain_keywords: Vec<(String, ProblemDomain)>,
    /// Component names
    component_names: HashSet<String>,
}

impl QueryParser {
    /// Create a new query parser
    pub fn new() -> Self {
        let mut parser = Self::default();
        parser.initialize_keywords();
        parser
    }

    fn initialize_keywords(&mut self) {
        // Algorithm keywords
        self.algorithm_keywords.extend([
            "random_forest", "random forest", "randomforest",
            "linear_regression", "linear regression", "linearregression",
            "logistic_regression", "logistic regression", "logisticregression",
            "decision_tree", "decision tree", "decisiontree",
            "gradient_boosting", "gradient boosting", "gbm", "xgboost", "lightgbm",
            "naive_bayes", "naive bayes", "naivebayes",
            "knn", "k-nearest", "nearest neighbor",
            "svm", "support vector", "supportvector",
            "kmeans", "k-means", "clustering",
            "pca", "principal component", "dimensionality reduction",
            "dbscan", "density clustering",
            "neural network", "deep learning", "transformer", "llm",
            "lora", "qlora", "fine-tuning", "fine tuning", "finetuning",
        ].map(String::from));

        // Domain keywords
        self.domain_keywords = vec![
            // Supervised Learning
            ("classify".into(), ProblemDomain::SupervisedLearning),
            ("classification".into(), ProblemDomain::SupervisedLearning),
            ("predict".into(), ProblemDomain::SupervisedLearning),
            ("regression".into(), ProblemDomain::SupervisedLearning),
            ("train".into(), ProblemDomain::SupervisedLearning),
            ("supervised".into(), ProblemDomain::SupervisedLearning),
            // Unsupervised Learning
            ("cluster".into(), ProblemDomain::UnsupervisedLearning),
            ("clustering".into(), ProblemDomain::UnsupervisedLearning),
            ("unsupervised".into(), ProblemDomain::UnsupervisedLearning),
            ("anomaly".into(), ProblemDomain::UnsupervisedLearning),
            ("outlier".into(), ProblemDomain::UnsupervisedLearning),
            // Deep Learning
            ("neural".into(), ProblemDomain::DeepLearning),
            ("deep learning".into(), ProblemDomain::DeepLearning),
            ("transformer".into(), ProblemDomain::DeepLearning),
            ("llm".into(), ProblemDomain::DeepLearning),
            ("fine-tune".into(), ProblemDomain::DeepLearning),
            ("lora".into(), ProblemDomain::DeepLearning),
            // Inference
            ("serve".into(), ProblemDomain::Inference),
            ("serving".into(), ProblemDomain::Inference),
            ("inference".into(), ProblemDomain::Inference),
            ("deploy".into(), ProblemDomain::Inference),
            ("production".into(), ProblemDomain::Inference),
            // Linear Algebra
            ("matrix".into(), ProblemDomain::LinearAlgebra),
            ("tensor".into(), ProblemDomain::LinearAlgebra),
            ("vector".into(), ProblemDomain::LinearAlgebra),
            ("linear algebra".into(), ProblemDomain::LinearAlgebra),
            ("simd".into(), ProblemDomain::LinearAlgebra),
            // Vector Search
            ("similarity".into(), ProblemDomain::VectorSearch),
            ("embedding".into(), ProblemDomain::VectorSearch),
            ("vector search".into(), ProblemDomain::VectorSearch),
            ("nearest neighbor".into(), ProblemDomain::VectorSearch),
            // Graph Analytics
            ("graph".into(), ProblemDomain::GraphAnalytics),
            ("pagerank".into(), ProblemDomain::GraphAnalytics),
            ("pathfinding".into(), ProblemDomain::GraphAnalytics),
            ("community".into(), ProblemDomain::GraphAnalytics),
            // Python Migration
            ("python".into(), ProblemDomain::PythonMigration),
            ("sklearn".into(), ProblemDomain::PythonMigration),
            ("scikit".into(), ProblemDomain::PythonMigration),
            ("numpy".into(), ProblemDomain::PythonMigration),
            ("pandas".into(), ProblemDomain::PythonMigration),
            ("pytorch".into(), ProblemDomain::PythonMigration),
            // C Migration
            ("c code".into(), ProblemDomain::CMigration),
            ("c++".into(), ProblemDomain::CMigration),
            ("cpp".into(), ProblemDomain::CMigration),
            // Shell Migration
            ("bash".into(), ProblemDomain::ShellMigration),
            ("shell".into(), ProblemDomain::ShellMigration),
            ("script".into(), ProblemDomain::ShellMigration),
            // Distribution
            ("distributed".into(), ProblemDomain::DistributedCompute),
            ("parallel".into(), ProblemDomain::DistributedCompute),
            ("multi-node".into(), ProblemDomain::DistributedCompute),
            ("cluster".into(), ProblemDomain::DistributedCompute),
            // Data Pipeline
            ("data loading".into(), ProblemDomain::DataPipeline),
            ("csv".into(), ProblemDomain::DataPipeline),
            ("parquet".into(), ProblemDomain::DataPipeline),
            ("etl".into(), ProblemDomain::DataPipeline),
            // Model Serving
            ("lambda".into(), ProblemDomain::ModelServing),
            ("serverless".into(), ProblemDomain::ModelServing),
            ("container".into(), ProblemDomain::ModelServing),
            ("edge".into(), ProblemDomain::ModelServing),
            // Quality
            ("test".into(), ProblemDomain::Testing),
            ("coverage".into(), ProblemDomain::Testing),
            ("mutation".into(), ProblemDomain::Testing),
            ("profile".into(), ProblemDomain::Profiling),
            ("trace".into(), ProblemDomain::Profiling),
            ("syscall".into(), ProblemDomain::Profiling),
            ("validate".into(), ProblemDomain::Validation),
            ("quality".into(), ProblemDomain::Validation),
        ];

        // Component names
        self.component_names.extend([
            "trueno", "trueno-db", "trueno-graph", "trueno-viz",
            "aprender", "entrenar", "realizar",
            "depyler", "decy", "bashrs", "ruchy",
            "batuta", "repartir",
            "certeza", "pmat", "renacer",
            "alimentar",
        ].map(String::from));
    }

    /// Parse a natural language query
    pub fn parse(&self, query: &str) -> ParsedQuery {
        let lower = query.to_lowercase();

        ParsedQuery {
            original: query.to_string(),
            domains: self.extract_domains(&lower),
            algorithms: self.extract_algorithms(&lower),
            keywords: self.extract_keywords(&lower),
            data_size: self.extract_data_size(&lower),
            performance_hints: self.extract_performance_hints(&lower),
            mentioned_components: self.extract_components(&lower),
        }
    }

    fn extract_domains(&self, query: &str) -> Vec<ProblemDomain> {
        let mut domains = Vec::new();
        let mut seen = HashSet::new();

        for (keyword, domain) in &self.domain_keywords {
            if query.contains(keyword) && !seen.contains(domain) {
                domains.push(*domain);
                seen.insert(*domain);
            }
        }

        domains
    }

    fn extract_algorithms(&self, query: &str) -> Vec<String> {
        let mut algorithms = Vec::new();

        for algo in &self.algorithm_keywords {
            if query.contains(algo) {
                // Normalize algorithm name
                let normalized = algo
                    .replace([' ', '-'], "_")
                    .to_lowercase();
                if !algorithms.contains(&normalized) {
                    algorithms.push(normalized);
                }
            }
        }

        algorithms
    }

    fn extract_keywords(&self, query: &str) -> Vec<String> {
        // Extract significant keywords (words > 3 chars, not stopwords)
        let stopwords: HashSet<_> = [
            "the", "and", "for", "with", "how", "what", "can", "does",
            "want", "need", "use", "using", "have", "this", "that",
            "from", "into", "about", "which", "when", "where", "should",
        ].iter().map(|s| s.to_string()).collect();

        query
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() > 3 && !stopwords.contains(*w))
            .map(String::from)
            .collect()
    }

    fn extract_data_size(&self, query: &str) -> Option<DataSize> {
        // Look for patterns like "1M samples", "100K rows", "1GB data"
        // Note: regex patterns defined for reference but simple string matching used below
        // for patterns like: r"(\d+)\s*[mM]\s*(samples?|rows?|records?|items?)"

        for (suffix, multiplier) in [
            ("m samples", 1_000_000),
            ("m rows", 1_000_000),
            ("k samples", 1_000),
            ("k rows", 1_000),
            ("million", 1_000_000),
            ("thousand", 1_000),
            ("billion", 1_000_000_000),
        ] {
            if let Some(idx) = query.find(suffix) {
                // Look for number before suffix
                let before = &query[..idx];
                if let Some(num_str) = before.split_whitespace().last() {
                    if let Ok(num) = num_str.parse::<u64>() {
                        return Some(DataSize::samples(num * multiplier));
                    }
                }
            }
        }

        // Look for "large" / "small" / "huge" indicators
        if query.contains("large") || query.contains("huge") || query.contains("big") {
            return Some(DataSize::samples(1_000_000));
        }
        if query.contains("small") || query.contains("tiny") {
            return Some(DataSize::samples(1_000));
        }

        None
    }

    fn extract_performance_hints(&self, query: &str) -> Vec<PerformanceHint> {
        let mut hints = Vec::new();

        if query.contains("fast") || query.contains("low latency") || query.contains("<") && query.contains("ms") {
            hints.push(PerformanceHint::LowLatency);
        }
        if query.contains("throughput") || query.contains("high volume") {
            hints.push(PerformanceHint::HighThroughput);
        }
        if query.contains("memory") && (query.contains("low") || query.contains("efficient")) {
            hints.push(PerformanceHint::LowMemory);
        }
        if query.contains("gpu") {
            hints.push(PerformanceHint::GPURequired);
        }
        if query.contains("distributed") || query.contains("multi-node") || query.contains("cluster") {
            hints.push(PerformanceHint::Distributed);
        }
        if query.contains("edge") || query.contains("embedded") || query.contains("iot") {
            hints.push(PerformanceHint::EdgeDeployment);
        }
        if query.contains("sovereign") || query.contains("gdpr") || query.contains("local only")
            || query.contains("eu ai act") || query.contains("on-premise")
        {
            hints.push(PerformanceHint::Sovereign);
        }

        hints
    }

    fn extract_components(&self, query: &str) -> Vec<String> {
        self.component_names
            .iter()
            .filter(|name| query.contains(name.as_str()))
            .cloned()
            .collect()
    }
}

// =============================================================================
// Query Engine
// =============================================================================

/// Query engine that processes queries and generates responses
pub struct QueryEngine {
    parser: QueryParser,
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryEngine {
    /// Create a new query engine
    pub fn new() -> Self {
        Self {
            parser: QueryParser::new(),
        }
    }

    /// Parse a query string
    pub fn parse(&self, query: &str) -> ParsedQuery {
        self.parser.parse(query)
    }

    /// Determine the primary problem domain from a parsed query
    pub fn primary_domain(&self, parsed: &ParsedQuery) -> Option<ProblemDomain> {
        parsed.domains.first().copied()
    }

    /// Determine the primary algorithm mentioned
    pub fn primary_algorithm<'a>(&self, parsed: &'a ParsedQuery) -> Option<&'a str> {
        parsed.algorithms.first().map(|s| s.as_str())
    }

    /// Check if the query requires GPU
    pub fn requires_gpu(&self, parsed: &ParsedQuery) -> bool {
        parsed.performance_hints.contains(&PerformanceHint::GPURequired)
    }

    /// Check if the query requires distributed computing
    pub fn requires_distribution(&self, parsed: &ParsedQuery) -> bool {
        parsed.performance_hints.contains(&PerformanceHint::Distributed)
            || parsed.data_size.map(|s| s.is_large()).unwrap_or(false)
    }

    /// Check if the query requires sovereign/local execution
    pub fn requires_sovereign(&self, parsed: &ParsedQuery) -> bool {
        parsed.performance_hints.contains(&PerformanceHint::Sovereign)
    }

    /// Estimate operation complexity from parsed query
    pub fn estimate_complexity(&self, parsed: &ParsedQuery) -> OpComplexity {
        // Matrix operations are high complexity
        if parsed.keywords.iter().any(|k| k.contains("matrix") || k.contains("matmul")) {
            return OpComplexity::High;
        }

        // Training/deep learning is high complexity
        if parsed.domains.contains(&ProblemDomain::DeepLearning) {
            return OpComplexity::High;
        }

        // Graph analytics is medium-high
        if parsed.domains.contains(&ProblemDomain::GraphAnalytics) {
            return OpComplexity::Medium;
        }

        // Most ML algorithms are medium
        if parsed.domains.contains(&ProblemDomain::SupervisedLearning)
            || parsed.domains.contains(&ProblemDomain::UnsupervisedLearning)
        {
            return OpComplexity::Medium;
        }

        // Default to low
        OpComplexity::Low
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // QueryParser Tests
    // =========================================================================

    #[test]
    fn test_parser_new() {
        let parser = QueryParser::new();
        assert!(!parser.algorithm_keywords.is_empty());
        assert!(!parser.domain_keywords.is_empty());
        assert!(!parser.component_names.is_empty());
    }

    #[test]
    fn test_parse_basic() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Train a random forest classifier");

        assert_eq!(parsed.original, "Train a random forest classifier");
        assert!(!parsed.domains.is_empty());
        assert!(!parsed.algorithms.is_empty());
    }

    // =========================================================================
    // Domain Extraction Tests
    // =========================================================================

    #[test]
    fn test_extract_supervised_learning() {
        let parser = QueryParser::new();
        let parsed = parser.parse("I want to train a classification model");

        assert!(parsed.domains.contains(&ProblemDomain::SupervisedLearning));
    }

    #[test]
    fn test_extract_unsupervised_learning() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Help me cluster my data for anomaly detection");

        assert!(parsed.domains.contains(&ProblemDomain::UnsupervisedLearning));
    }

    #[test]
    fn test_extract_deep_learning() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Fine-tune a transformer with LoRA");

        assert!(parsed.domains.contains(&ProblemDomain::DeepLearning));
    }

    #[test]
    fn test_extract_inference() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Deploy model for production inference");

        assert!(parsed.domains.contains(&ProblemDomain::Inference));
    }

    #[test]
    fn test_extract_python_migration() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Convert my sklearn pipeline to Rust");

        assert!(parsed.domains.contains(&ProblemDomain::PythonMigration));
    }

    #[test]
    fn test_extract_linear_algebra() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Fast matrix multiplication with SIMD");

        assert!(parsed.domains.contains(&ProblemDomain::LinearAlgebra));
    }

    #[test]
    fn test_extract_graph_analytics() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Run pagerank on my graph");

        assert!(parsed.domains.contains(&ProblemDomain::GraphAnalytics));
    }

    #[test]
    fn test_extract_multiple_domains() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Train a classifier on python sklearn data");

        assert!(parsed.domains.len() >= 2);
        assert!(parsed.domains.contains(&ProblemDomain::SupervisedLearning));
        assert!(parsed.domains.contains(&ProblemDomain::PythonMigration));
    }

    // =========================================================================
    // Algorithm Extraction Tests
    // =========================================================================

    #[test]
    fn test_extract_random_forest() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Train a random forest on my data");

        assert!(parsed.algorithms.iter().any(|a| a.contains("random_forest")));
    }

    #[test]
    fn test_extract_gradient_boosting() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Use gradient boosting for regression");

        assert!(parsed.algorithms.iter().any(|a| a.contains("gradient_boosting") || a == "gbm"));
    }

    #[test]
    fn test_extract_kmeans() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Cluster with k-means algorithm");

        assert!(parsed.algorithms.iter().any(|a| a.contains("kmeans") || a.contains("k_means")));
    }

    #[test]
    fn test_extract_lora() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Fine-tune with LoRA");

        assert!(parsed.algorithms.iter().any(|a| a.contains("lora")));
    }

    // =========================================================================
    // Data Size Extraction Tests
    // =========================================================================

    #[test]
    fn test_extract_data_size_million() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Train on 1 million samples");

        assert!(parsed.data_size.is_some());
        let size = parsed.data_size.unwrap();
        assert!(size.is_large());
    }

    #[test]
    fn test_extract_data_size_1m() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Process 5m rows of data");

        assert!(parsed.data_size.is_some());
    }

    #[test]
    fn test_extract_data_size_thousand() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Test on 10 thousand samples");

        assert!(parsed.data_size.is_some());
        let size = parsed.data_size.unwrap();
        assert!(!size.is_large());
    }

    #[test]
    fn test_extract_data_size_large_indicator() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Handle large dataset");

        assert!(parsed.data_size.is_some());
        assert!(parsed.data_size.unwrap().is_large());
    }

    #[test]
    fn test_extract_data_size_small_indicator() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Small dataset for testing");

        assert!(parsed.data_size.is_some());
        assert!(!parsed.data_size.unwrap().is_large());
    }

    // =========================================================================
    // Performance Hints Tests
    // =========================================================================

    #[test]
    fn test_extract_low_latency() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Need fast inference with low latency");

        assert!(parsed.performance_hints.contains(&PerformanceHint::LowLatency));
    }

    #[test]
    fn test_extract_gpu_required() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Train model on GPU");

        assert!(parsed.performance_hints.contains(&PerformanceHint::GPURequired));
    }

    #[test]
    fn test_extract_distributed() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Distributed training on multi-node cluster");

        assert!(parsed.performance_hints.contains(&PerformanceHint::Distributed));
    }

    #[test]
    fn test_extract_edge_deployment() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Deploy model to edge devices");

        assert!(parsed.performance_hints.contains(&PerformanceHint::EdgeDeployment));
    }

    #[test]
    fn test_extract_sovereign() {
        let parser = QueryParser::new();
        let parsed = parser.parse("GDPR compliant, sovereign execution");

        assert!(parsed.performance_hints.contains(&PerformanceHint::Sovereign));
    }

    #[test]
    fn test_extract_eu_ai_act() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Must comply with EU AI Act");

        assert!(parsed.performance_hints.contains(&PerformanceHint::Sovereign));
    }

    // =========================================================================
    // Component Extraction Tests
    // =========================================================================

    #[test]
    fn test_extract_component_trueno() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Use trueno for tensor operations");

        assert!(parsed.mentioned_components.contains(&"trueno".to_string()));
    }

    #[test]
    fn test_extract_component_aprender() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Train with aprender random forest");

        assert!(parsed.mentioned_components.contains(&"aprender".to_string()));
    }

    #[test]
    fn test_extract_multiple_components() {
        let parser = QueryParser::new();
        let parsed = parser.parse("Use depyler to convert sklearn to aprender");

        assert!(parsed.mentioned_components.contains(&"depyler".to_string()));
        assert!(parsed.mentioned_components.contains(&"aprender".to_string()));
    }

    // =========================================================================
    // QueryEngine Tests
    // =========================================================================

    #[test]
    fn test_query_engine_new() {
        let engine = QueryEngine::new();
        let parsed = engine.parse("Test query");
        assert!(!parsed.original.is_empty());
    }

    #[test]
    fn test_query_engine_default() {
        let engine = QueryEngine::default();
        let parsed = engine.parse("Test");
        assert_eq!(parsed.original, "Test");
    }

    #[test]
    fn test_primary_domain() {
        let engine = QueryEngine::new();
        let parsed = engine.parse("Train a classifier");

        let domain = engine.primary_domain(&parsed);
        assert!(domain.is_some());
        assert_eq!(domain.unwrap(), ProblemDomain::SupervisedLearning);
    }

    #[test]
    fn test_primary_algorithm() {
        let engine = QueryEngine::new();
        let parsed = engine.parse("Use random forest");

        let algo = engine.primary_algorithm(&parsed);
        assert!(algo.is_some());
        assert!(algo.unwrap().contains("random_forest"));
    }

    #[test]
    fn test_requires_gpu() {
        let engine = QueryEngine::new();

        let parsed = engine.parse("Train on GPU");
        assert!(engine.requires_gpu(&parsed));

        let parsed = engine.parse("Simple CPU training");
        assert!(!engine.requires_gpu(&parsed));
    }

    #[test]
    fn test_requires_distribution() {
        let engine = QueryEngine::new();

        let parsed = engine.parse("Distributed training");
        assert!(engine.requires_distribution(&parsed));

        let parsed = engine.parse("Train on 1 billion samples");
        assert!(engine.requires_distribution(&parsed));

        let parsed = engine.parse("Small local training");
        assert!(!engine.requires_distribution(&parsed));
    }

    #[test]
    fn test_requires_sovereign() {
        let engine = QueryEngine::new();

        let parsed = engine.parse("GDPR compliant local execution");
        assert!(engine.requires_sovereign(&parsed));

        let parsed = engine.parse("Cloud training");
        assert!(!engine.requires_sovereign(&parsed));
    }

    #[test]
    fn test_estimate_complexity_high() {
        let engine = QueryEngine::new();

        let parsed = engine.parse("Matrix multiplication");
        assert_eq!(engine.estimate_complexity(&parsed), OpComplexity::High);

        let parsed = engine.parse("Deep learning training");
        assert_eq!(engine.estimate_complexity(&parsed), OpComplexity::High);
    }

    #[test]
    fn test_estimate_complexity_medium() {
        let engine = QueryEngine::new();

        let parsed = engine.parse("Train a classifier");
        assert_eq!(engine.estimate_complexity(&parsed), OpComplexity::Medium);

        let parsed = engine.parse("Graph pagerank");
        assert_eq!(engine.estimate_complexity(&parsed), OpComplexity::Medium);
    }

    #[test]
    fn test_estimate_complexity_low() {
        let engine = QueryEngine::new();

        let parsed = engine.parse("Simple data loading");
        assert_eq!(engine.estimate_complexity(&parsed), OpComplexity::Low);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_query_parsing() {
        let engine = QueryEngine::new();
        let parsed = engine.parse(
            "I need to train a random forest on 1 million samples with GPU acceleration"
        );

        // Should detect supervised learning
        assert!(parsed.domains.contains(&ProblemDomain::SupervisedLearning));

        // Should detect random forest
        assert!(parsed.algorithms.iter().any(|a| a.contains("random_forest")));

        // Should detect large data
        assert!(parsed.data_size.is_some());
        assert!(parsed.data_size.unwrap().is_large());

        // Should detect GPU requirement
        assert!(parsed.performance_hints.contains(&PerformanceHint::GPURequired));
    }

    #[test]
    fn test_sklearn_migration_query() {
        let engine = QueryEngine::new();
        let parsed = engine.parse(
            "Convert my sklearn pipeline with RandomForest to Rust aprender"
        );

        assert!(parsed.domains.contains(&ProblemDomain::PythonMigration));
        assert!(parsed.algorithms.iter().any(|a| a.contains("random")));
        assert!(parsed.mentioned_components.contains(&"aprender".to_string()));
    }

    #[test]
    fn test_inference_query() {
        let engine = QueryEngine::new();
        let parsed = engine.parse(
            "Deploy model to AWS Lambda with <10ms latency"
        );

        assert!(parsed.domains.contains(&ProblemDomain::Inference));
        assert!(parsed.domains.contains(&ProblemDomain::ModelServing));
        assert!(parsed.performance_hints.contains(&PerformanceHint::LowLatency));
    }
}
