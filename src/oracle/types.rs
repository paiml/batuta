//! Oracle Mode type definitions
//!
//! Based on Oracle Mode Specification v1.0 (BATUTA-ORACLE-001)

use serde::{Deserialize, Serialize};

// =============================================================================
// Stack Layer Definitions
// =============================================================================

/// Layer in the Sovereign AI Stack hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StackLayer {
    /// Layer 0: Compute primitives (trueno, trueno-db, trueno-graph)
    Primitives,
    /// Layer 1: ML algorithms (aprender)
    MlAlgorithms,
    /// Layer 2: Training & inference (entrenar, realizar)
    MlPipeline,
    /// Layer 3: Transpilers (depyler, decy, bashrs)
    Transpilers,
    /// Layer 4: Orchestration (batuta, repartir)
    Orchestration,
    /// Layer 5: Quality & profiling (certeza, pmat, renacer)
    Quality,
    /// Layer 6: Data loading (alimentar)
    Data,
}

impl StackLayer {
    /// Get numeric layer index
    pub fn index(&self) -> u8 {
        match self {
            StackLayer::Primitives => 0,
            StackLayer::MlAlgorithms => 1,
            StackLayer::MlPipeline => 2,
            StackLayer::Transpilers => 3,
            StackLayer::Orchestration => 4,
            StackLayer::Quality => 5,
            StackLayer::Data => 6,
        }
    }

    /// Get all layers in order
    pub fn all() -> Vec<StackLayer> {
        vec![
            StackLayer::Primitives,
            StackLayer::MlAlgorithms,
            StackLayer::MlPipeline,
            StackLayer::Transpilers,
            StackLayer::Orchestration,
            StackLayer::Quality,
            StackLayer::Data,
        ]
    }
}

impl std::fmt::Display for StackLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StackLayer::Primitives => write!(f, "Compute Primitives"),
            StackLayer::MlAlgorithms => write!(f, "ML Algorithms"),
            StackLayer::MlPipeline => write!(f, "Training & Inference"),
            StackLayer::Transpilers => write!(f, "Transpilers"),
            StackLayer::Orchestration => write!(f, "Orchestration"),
            StackLayer::Quality => write!(f, "Quality & Profiling"),
            StackLayer::Data => write!(f, "Data Loading"),
        }
    }
}

// =============================================================================
// Capability Definitions
// =============================================================================

/// Category of capability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityCategory {
    Compute,
    Storage,
    MachineLearning,
    Transpilation,
    Validation,
    Profiling,
    Distribution,
}

/// A capability provided by a component
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,
    pub category: CapabilityCategory,
    pub description: Option<String>,
}

impl Capability {
    pub fn new(name: impl Into<String>, category: CapabilityCategory) -> Self {
        Self {
            name: name.into(),
            category,
            description: None,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

// =============================================================================
// Component Definitions
// =============================================================================

/// A component in the Sovereign AI Stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackComponent {
    /// Component name (e.g., "trueno", "aprender")
    pub name: String,
    /// Component version
    pub version: String,
    /// Stack layer
    pub layer: StackLayer,
    /// Description of component purpose
    pub description: String,
    /// Capabilities provided
    pub capabilities: Vec<Capability>,
    /// Crates.io package name (if different from name)
    pub crate_name: Option<String>,
    /// Academic references
    pub references: Vec<Citation>,
}

impl StackComponent {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        layer: StackLayer,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            layer,
            description: description.into(),
            capabilities: Vec::new(),
            crate_name: None,
            references: Vec::new(),
        }
    }

    pub fn with_capability(mut self, cap: Capability) -> Self {
        self.capabilities.push(cap);
        self
    }

    pub fn with_capabilities(mut self, caps: Vec<Capability>) -> Self {
        self.capabilities.extend(caps);
        self
    }

    pub fn has_capability(&self, name: &str) -> bool {
        self.capabilities.iter().any(|c| c.name == name)
    }
}

/// Academic citation/reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub id: u32,
    pub authors: String,
    pub year: u16,
    pub title: String,
    pub venue: Option<String>,
}

// =============================================================================
// Query Types
// =============================================================================

/// Hardware specification for queries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub has_gpu: bool,
    pub gpu_memory_gb: Option<f32>,
    pub cpu_cores: Option<u32>,
    pub ram_gb: Option<f32>,
    pub is_distributed: bool,
    pub node_count: Option<u32>,
}

impl HardwareSpec {
    pub fn cpu_only() -> Self {
        Self {
            has_gpu: false,
            ..Default::default()
        }
    }

    pub fn with_gpu(memory_gb: f32) -> Self {
        Self {
            has_gpu: true,
            gpu_memory_gb: Some(memory_gb),
            ..Default::default()
        }
    }

    pub fn has_gpu(&self) -> bool {
        self.has_gpu
    }
}

/// Data size specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataSize {
    /// Number of samples/rows
    Samples(u64),
    /// Size in bytes
    Bytes(u64),
    /// Unknown/unspecified
    Unknown,
}

impl DataSize {
    pub fn samples(n: u64) -> Self {
        DataSize::Samples(n)
    }

    pub fn bytes(n: u64) -> Self {
        DataSize::Bytes(n)
    }

    /// Get sample count if available
    pub fn as_samples(&self) -> Option<u64> {
        match self {
            DataSize::Samples(n) => Some(*n),
            _ => None,
        }
    }

    /// Estimate if this is "large" data (>100K samples or >1GB)
    pub fn is_large(&self) -> bool {
        match self {
            DataSize::Samples(n) => *n > 100_000,
            DataSize::Bytes(n) => *n > 1_000_000_000,
            DataSize::Unknown => false,
        }
    }
}

/// Optimization target for queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OptimizationTarget {
    /// Optimize for execution speed
    #[default]
    Speed,
    /// Optimize for memory efficiency
    Memory,
    /// Optimize for power efficiency
    Power,
    /// Balance all factors
    Balanced,
}

/// Query constraints
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryConstraints {
    /// Maximum latency requirement (ms)
    pub max_latency_ms: Option<u64>,
    /// Data size
    pub data_size: Option<DataSize>,
    /// Must run locally (no cloud)
    pub sovereign_only: bool,
    /// EU AI Act compliance required
    pub eu_compliant: bool,
    /// Available hardware
    pub hardware: HardwareSpec,
}

/// Query preferences
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryPreferences {
    /// Optimization target
    pub optimize_for: OptimizationTarget,
    /// Preference for simpler solutions (0.0-1.0)
    pub simplicity_weight: f32,
    /// Existing stack components to integrate with
    pub existing_components: Vec<String>,
}

/// Oracle query structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleQuery {
    /// Problem description in natural language
    pub description: String,
    /// Constraints
    pub constraints: QueryConstraints,
    /// Preferences
    pub preferences: QueryPreferences,
}

impl OracleQuery {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            constraints: QueryConstraints::default(),
            preferences: QueryPreferences::default(),
        }
    }

    pub fn with_constraints(mut self, constraints: QueryConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn with_preferences(mut self, preferences: QueryPreferences) -> Self {
        self.preferences = preferences;
        self
    }

    pub fn with_data_size(mut self, size: DataSize) -> Self {
        self.constraints.data_size = Some(size);
        self
    }

    pub fn with_hardware(mut self, hardware: HardwareSpec) -> Self {
        self.constraints.hardware = hardware;
        self
    }

    pub fn sovereign_only(mut self) -> Self {
        self.constraints.sovereign_only = true;
        self
    }

    pub fn eu_compliant(mut self) -> Self {
        self.constraints.eu_compliant = true;
        self
    }
}

// =============================================================================
// Response Types
// =============================================================================

/// Compute backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub enum Backend {
    Scalar,
    SIMD,
    GPU,
    Distributed,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Scalar => write!(f, "Scalar"),
            Backend::SIMD => write!(f, "SIMD"),
            Backend::GPU => write!(f, "GPU"),
            Backend::Distributed => write!(f, "Distributed"),
        }
    }
}

/// Operation complexity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpComplexity {
    /// O(n) - element-wise operations
    Low,
    /// O(n log n) to O(n²) - reductions, sorts
    Medium,
    /// O(n²) to O(n³) - matrix operations
    High,
}

/// Compute recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeRecommendation {
    pub backend: Backend,
    pub rationale: String,
}

/// Distribution recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionRecommendation {
    pub tool: Option<String>,
    pub needed: bool,
    pub rationale: String,
    pub node_count: Option<u32>,
}

/// Component recommendation with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentRecommendation {
    /// Component name
    pub component: String,
    /// Specific module/function path (e.g., "aprender::tree::RandomForest")
    pub path: Option<String>,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Reason for recommendation
    pub rationale: String,
}

/// Complete Oracle response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleResponse {
    /// Problem classification
    pub problem_class: String,
    /// Detected algorithm/approach
    pub algorithm: Option<String>,
    /// Primary recommendation
    pub primary: ComponentRecommendation,
    /// Supporting components
    pub supporting: Vec<ComponentRecommendation>,
    /// Compute backend recommendation
    pub compute: ComputeRecommendation,
    /// Distribution recommendation
    pub distribution: DistributionRecommendation,
    /// Example code snippet
    pub code_example: Option<String>,
    /// Related queries for follow-up
    pub related_queries: Vec<String>,
}

impl OracleResponse {
    pub fn new(problem_class: impl Into<String>, primary: ComponentRecommendation) -> Self {
        Self {
            problem_class: problem_class.into(),
            algorithm: None,
            primary,
            supporting: Vec::new(),
            compute: ComputeRecommendation {
                backend: Backend::SIMD,
                rationale: "Default SIMD backend".into(),
            },
            distribution: DistributionRecommendation {
                tool: None,
                needed: false,
                rationale: "Single-node sufficient".into(),
                node_count: None,
            },
            code_example: None,
            related_queries: Vec::new(),
        }
    }

    pub fn with_algorithm(mut self, algo: impl Into<String>) -> Self {
        self.algorithm = Some(algo.into());
        self
    }

    pub fn with_supporting(mut self, rec: ComponentRecommendation) -> Self {
        self.supporting.push(rec);
        self
    }

    pub fn with_compute(mut self, compute: ComputeRecommendation) -> Self {
        self.compute = compute;
        self
    }

    pub fn with_distribution(mut self, dist: DistributionRecommendation) -> Self {
        self.distribution = dist;
        self
    }

    pub fn with_code_example(mut self, code: impl Into<String>) -> Self {
        self.code_example = Some(code.into());
        self
    }
}

// =============================================================================
// Problem Domain Classification
// =============================================================================

/// Problem domain for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProblemDomain {
    // ML domains
    SupervisedLearning,
    UnsupervisedLearning,
    DeepLearning,
    Inference,
    // Compute domains
    LinearAlgebra,
    VectorSearch,
    GraphAnalytics,
    // Transpilation domains
    PythonMigration,
    CMigration,
    ShellMigration,
    // Infrastructure domains
    DistributedCompute,
    DataPipeline,
    ModelServing,
    // Quality domains
    Testing,
    Profiling,
    Validation,
}

impl std::fmt::Display for ProblemDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProblemDomain::SupervisedLearning => write!(f, "Supervised Learning"),
            ProblemDomain::UnsupervisedLearning => write!(f, "Unsupervised Learning"),
            ProblemDomain::DeepLearning => write!(f, "Deep Learning"),
            ProblemDomain::Inference => write!(f, "Model Inference"),
            ProblemDomain::LinearAlgebra => write!(f, "Linear Algebra"),
            ProblemDomain::VectorSearch => write!(f, "Vector Search"),
            ProblemDomain::GraphAnalytics => write!(f, "Graph Analytics"),
            ProblemDomain::PythonMigration => write!(f, "Python Migration"),
            ProblemDomain::CMigration => write!(f, "C/C++ Migration"),
            ProblemDomain::ShellMigration => write!(f, "Shell Migration"),
            ProblemDomain::DistributedCompute => write!(f, "Distributed Computing"),
            ProblemDomain::DataPipeline => write!(f, "Data Pipeline"),
            ProblemDomain::ModelServing => write!(f, "Model Serving"),
            ProblemDomain::Testing => write!(f, "Testing"),
            ProblemDomain::Profiling => write!(f, "Profiling"),
            ProblemDomain::Validation => write!(f, "Validation"),
        }
    }
}

// =============================================================================
// Integration Patterns
// =============================================================================

/// Integration pattern between components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationPattern {
    pub from: String,
    pub to: String,
    pub pattern_name: String,
    pub description: String,
    pub code_template: Option<String>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // StackLayer Tests
    // =========================================================================

    #[test]
    fn test_stack_layer_index() {
        assert_eq!(StackLayer::Primitives.index(), 0);
        assert_eq!(StackLayer::MlAlgorithms.index(), 1);
        assert_eq!(StackLayer::MlPipeline.index(), 2);
        assert_eq!(StackLayer::Transpilers.index(), 3);
        assert_eq!(StackLayer::Orchestration.index(), 4);
        assert_eq!(StackLayer::Quality.index(), 5);
        assert_eq!(StackLayer::Data.index(), 6);
    }

    #[test]
    fn test_stack_layer_all() {
        let layers = StackLayer::all();
        assert_eq!(layers.len(), 7);
        assert_eq!(layers[0], StackLayer::Primitives);
        assert_eq!(layers[6], StackLayer::Data);
    }

    #[test]
    fn test_stack_layer_display() {
        assert_eq!(StackLayer::Primitives.to_string(), "Compute Primitives");
        assert_eq!(StackLayer::MlAlgorithms.to_string(), "ML Algorithms");
        assert_eq!(StackLayer::Transpilers.to_string(), "Transpilers");
    }

    #[test]
    fn test_stack_layer_serialization() {
        let layer = StackLayer::MlAlgorithms;
        let json = serde_json::to_string(&layer).unwrap();
        let parsed: StackLayer = serde_json::from_str(&json).unwrap();
        assert_eq!(layer, parsed);
    }

    // =========================================================================
    // Capability Tests
    // =========================================================================

    #[test]
    fn test_capability_new() {
        let cap = Capability::new("simd", CapabilityCategory::Compute);
        assert_eq!(cap.name, "simd");
        assert_eq!(cap.category, CapabilityCategory::Compute);
        assert!(cap.description.is_none());
    }

    #[test]
    fn test_capability_with_description() {
        let cap = Capability::new("vector_ops", CapabilityCategory::Compute)
            .with_description("SIMD-accelerated vector operations");
        assert_eq!(cap.description.as_deref(), Some("SIMD-accelerated vector operations"));
    }

    #[test]
    fn test_capability_equality() {
        let cap1 = Capability::new("simd", CapabilityCategory::Compute);
        let cap2 = Capability::new("simd", CapabilityCategory::Compute);
        let cap3 = Capability::new("gpu", CapabilityCategory::Compute);
        assert_eq!(cap1, cap2);
        assert_ne!(cap1, cap3);
    }

    // =========================================================================
    // StackComponent Tests
    // =========================================================================

    #[test]
    fn test_stack_component_new() {
        let comp = StackComponent::new(
            "trueno",
            "0.7.3",
            StackLayer::Primitives,
            "SIMD-accelerated tensor operations",
        );
        assert_eq!(comp.name, "trueno");
        assert_eq!(comp.version, "0.7.3");
        assert_eq!(comp.layer, StackLayer::Primitives);
        assert!(comp.capabilities.is_empty());
    }

    #[test]
    fn test_stack_component_with_capability() {
        let comp = StackComponent::new("trueno", "0.7.3", StackLayer::Primitives, "SIMD tensors")
            .with_capability(Capability::new("simd", CapabilityCategory::Compute))
            .with_capability(Capability::new("gpu", CapabilityCategory::Compute));
        assert_eq!(comp.capabilities.len(), 2);
        assert!(comp.has_capability("simd"));
        assert!(comp.has_capability("gpu"));
        assert!(!comp.has_capability("tpu"));
    }

    #[test]
    fn test_stack_component_with_capabilities() {
        let caps = vec![
            Capability::new("vector_ops", CapabilityCategory::Compute),
            Capability::new("matrix_ops", CapabilityCategory::Compute),
        ];
        let comp = StackComponent::new("trueno", "0.7.3", StackLayer::Primitives, "SIMD tensors")
            .with_capabilities(caps);
        assert_eq!(comp.capabilities.len(), 2);
    }

    #[test]
    fn test_stack_component_serialization() {
        let comp = StackComponent::new("aprender", "0.12.0", StackLayer::MlAlgorithms, "ML library")
            .with_capability(Capability::new("random_forest", CapabilityCategory::MachineLearning));
        let json = serde_json::to_string(&comp).unwrap();
        let parsed: StackComponent = serde_json::from_str(&json).unwrap();
        assert_eq!(comp.name, parsed.name);
        assert_eq!(comp.version, parsed.version);
        assert_eq!(comp.capabilities.len(), parsed.capabilities.len());
    }

    // =========================================================================
    // HardwareSpec Tests
    // =========================================================================

    #[test]
    fn test_hardware_spec_default() {
        let hw = HardwareSpec::default();
        assert!(!hw.has_gpu);
        assert!(hw.gpu_memory_gb.is_none());
        assert!(!hw.is_distributed);
    }

    #[test]
    fn test_hardware_spec_cpu_only() {
        let hw = HardwareSpec::cpu_only();
        assert!(!hw.has_gpu());
    }

    #[test]
    fn test_hardware_spec_with_gpu() {
        let hw = HardwareSpec::with_gpu(16.0);
        assert!(hw.has_gpu());
        assert_eq!(hw.gpu_memory_gb, Some(16.0));
    }

    // =========================================================================
    // DataSize Tests
    // =========================================================================

    #[test]
    fn test_data_size_samples() {
        let size = DataSize::samples(1_000_000);
        assert_eq!(size.as_samples(), Some(1_000_000));
        assert!(size.is_large());
    }

    #[test]
    fn test_data_size_bytes() {
        let size = DataSize::bytes(2_000_000_000);
        assert!(size.is_large());
        assert!(size.as_samples().is_none());
    }

    #[test]
    fn test_data_size_small() {
        let size = DataSize::samples(1000);
        assert!(!size.is_large());
    }

    #[test]
    fn test_data_size_unknown() {
        let size = DataSize::Unknown;
        assert!(!size.is_large());
        assert!(size.as_samples().is_none());
    }

    // =========================================================================
    // OracleQuery Tests
    // =========================================================================

    #[test]
    fn test_oracle_query_new() {
        let query = OracleQuery::new("Train a random forest on 1M samples");
        assert_eq!(query.description, "Train a random forest on 1M samples");
        assert!(!query.constraints.sovereign_only);
        assert!(!query.constraints.eu_compliant);
    }

    #[test]
    fn test_oracle_query_with_data_size() {
        let query = OracleQuery::new("classification task")
            .with_data_size(DataSize::samples(1_000_000));
        assert_eq!(query.constraints.data_size, Some(DataSize::Samples(1_000_000)));
    }

    #[test]
    fn test_oracle_query_sovereign_only() {
        let query = OracleQuery::new("GDPR compliant training").sovereign_only();
        assert!(query.constraints.sovereign_only);
    }

    #[test]
    fn test_oracle_query_eu_compliant() {
        let query = OracleQuery::new("EU AI Act compliant").eu_compliant();
        assert!(query.constraints.eu_compliant);
    }

    #[test]
    fn test_oracle_query_with_hardware() {
        let query = OracleQuery::new("GPU training")
            .with_hardware(HardwareSpec::with_gpu(24.0));
        assert!(query.constraints.hardware.has_gpu());
    }

    #[test]
    fn test_oracle_query_serialization() {
        let query = OracleQuery::new("Test query")
            .with_data_size(DataSize::samples(10000))
            .sovereign_only();
        let json = serde_json::to_string(&query).unwrap();
        let parsed: OracleQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(query.description, parsed.description);
        assert_eq!(query.constraints.sovereign_only, parsed.constraints.sovereign_only);
    }

    // =========================================================================
    // Backend Tests
    // =========================================================================

    #[test]
    fn test_backend_display() {
        assert_eq!(Backend::Scalar.to_string(), "Scalar");
        assert_eq!(Backend::SIMD.to_string(), "SIMD");
        assert_eq!(Backend::GPU.to_string(), "GPU");
        assert_eq!(Backend::Distributed.to_string(), "Distributed");
    }

    #[test]
    fn test_backend_serialization() {
        let backend = Backend::GPU;
        let json = serde_json::to_string(&backend).unwrap();
        let parsed: Backend = serde_json::from_str(&json).unwrap();
        assert_eq!(backend, parsed);
    }

    // =========================================================================
    // OracleResponse Tests
    // =========================================================================

    #[test]
    fn test_oracle_response_new() {
        let rec = ComponentRecommendation {
            component: "aprender".into(),
            path: Some("aprender::tree::RandomForest".into()),
            confidence: 0.95,
            rationale: "Random forest is ideal for tabular data".into(),
        };
        let response = OracleResponse::new("supervised_learning", rec);
        assert_eq!(response.problem_class, "supervised_learning");
        assert_eq!(response.primary.component, "aprender");
        assert!(!response.distribution.needed);
    }

    #[test]
    fn test_oracle_response_with_algorithm() {
        let rec = ComponentRecommendation {
            component: "aprender".into(),
            path: None,
            confidence: 0.9,
            rationale: "Test".into(),
        };
        let response = OracleResponse::new("ml", rec)
            .with_algorithm("random_forest");
        assert_eq!(response.algorithm.as_deref(), Some("random_forest"));
    }

    #[test]
    fn test_oracle_response_with_supporting() {
        let primary = ComponentRecommendation {
            component: "aprender".into(),
            path: None,
            confidence: 0.9,
            rationale: "Primary".into(),
        };
        let supporting = ComponentRecommendation {
            component: "trueno".into(),
            path: None,
            confidence: 0.8,
            rationale: "Backend compute".into(),
        };
        let response = OracleResponse::new("ml", primary)
            .with_supporting(supporting);
        assert_eq!(response.supporting.len(), 1);
        assert_eq!(response.supporting[0].component, "trueno");
    }

    #[test]
    fn test_oracle_response_with_code_example() {
        let rec = ComponentRecommendation {
            component: "aprender".into(),
            path: None,
            confidence: 0.9,
            rationale: "Test".into(),
        };
        let code = r#"use aprender::tree::RandomForest;
let model = RandomForest::new();"#;
        let response = OracleResponse::new("ml", rec)
            .with_code_example(code);
        assert!(response.code_example.is_some());
        assert!(response.code_example.unwrap().contains("RandomForest"));
    }

    // =========================================================================
    // ProblemDomain Tests
    // =========================================================================

    #[test]
    fn test_problem_domain_display() {
        assert_eq!(ProblemDomain::SupervisedLearning.to_string(), "Supervised Learning");
        assert_eq!(ProblemDomain::PythonMigration.to_string(), "Python Migration");
        assert_eq!(ProblemDomain::GraphAnalytics.to_string(), "Graph Analytics");
    }

    #[test]
    fn test_problem_domain_serialization() {
        let domain = ProblemDomain::DeepLearning;
        let json = serde_json::to_string(&domain).unwrap();
        let parsed: ProblemDomain = serde_json::from_str(&json).unwrap();
        assert_eq!(domain, parsed);
    }

    // =========================================================================
    // OptimizationTarget Tests
    // =========================================================================

    #[test]
    fn test_optimization_target_default() {
        let target = OptimizationTarget::default();
        assert_eq!(target, OptimizationTarget::Speed);
    }

    #[test]
    fn test_optimization_target_serialization() {
        let target = OptimizationTarget::Memory;
        let json = serde_json::to_string(&target).unwrap();
        let parsed: OptimizationTarget = serde_json::from_str(&json).unwrap();
        assert_eq!(target, parsed);
    }

    // =========================================================================
    // QueryConstraints Tests
    // =========================================================================

    #[test]
    fn test_query_constraints_default() {
        let constraints = QueryConstraints::default();
        assert!(constraints.max_latency_ms.is_none());
        assert!(constraints.data_size.is_none());
        assert!(!constraints.sovereign_only);
        assert!(!constraints.eu_compliant);
    }

    // =========================================================================
    // QueryPreferences Tests
    // =========================================================================

    #[test]
    fn test_query_preferences_default() {
        let prefs = QueryPreferences::default();
        assert_eq!(prefs.optimize_for, OptimizationTarget::Speed);
        assert_eq!(prefs.simplicity_weight, 0.0);
        assert!(prefs.existing_components.is_empty());
    }

    // =========================================================================
    // IntegrationPattern Tests
    // =========================================================================

    #[test]
    fn test_integration_pattern() {
        let pattern = IntegrationPattern {
            from: "aprender".into(),
            to: "realizar".into(),
            pattern_name: "model_export".into(),
            description: "Export trained model for serving".into(),
            code_template: Some("model.export_apr(\"model.apr\")".into()),
        };
        assert_eq!(pattern.from, "aprender");
        assert_eq!(pattern.to, "realizar");
        assert!(pattern.code_template.is_some());
    }

    #[test]
    fn test_integration_pattern_serialization() {
        let pattern = IntegrationPattern {
            from: "depyler".into(),
            to: "aprender".into(),
            pattern_name: "sklearn_convert".into(),
            description: "Convert sklearn to aprender".into(),
            code_template: None,
        };
        let json = serde_json::to_string(&pattern).unwrap();
        let parsed: IntegrationPattern = serde_json::from_str(&json).unwrap();
        assert_eq!(pattern.from, parsed.from);
        assert_eq!(pattern.to, parsed.to);
    }
}
