//! Knowledge Graph for Sovereign AI Stack
//!
//! Provides component registry, capability indexing, and relationship mapping
//! between all stack components.

use super::types::*;
use std::collections::HashMap;

// =============================================================================
// Knowledge Graph
// =============================================================================

/// Knowledge graph containing all stack components and their relationships
#[derive(Debug, Clone)]
pub struct KnowledgeGraph {
    /// All registered components
    components: HashMap<String, StackComponent>,
    /// Capability to component index
    capability_index: HashMap<String, Vec<String>>,
    /// Problem domain to capability mapping
    domain_capabilities: HashMap<ProblemDomain, Vec<String>>,
    /// Integration patterns between components
    integrations: Vec<IntegrationPattern>,
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph
    pub fn new() -> Self {
        let mut graph = Self {
            components: HashMap::new(),
            capability_index: HashMap::new(),
            domain_capabilities: HashMap::new(),
            integrations: Vec::new(),
        };
        graph.initialize_domain_mappings();
        graph
    }

    /// Create a knowledge graph pre-populated with the Sovereign AI Stack
    pub fn sovereign_stack() -> Self {
        let mut graph = Self::new();
        graph.register_sovereign_stack();
        graph.register_integration_patterns();
        graph
    }

    /// Initialize problem domain to capability mappings
    fn initialize_domain_mappings(&mut self) {
        use ProblemDomain::*;

        self.domain_capabilities.insert(
            SupervisedLearning,
            vec![
                "linear_regression".into(),
                "logistic_regression".into(),
                "decision_tree".into(),
                "random_forest".into(),
                "gbm".into(),
                "naive_bayes".into(),
                "knn".into(),
                "svm".into(),
            ],
        );

        self.domain_capabilities.insert(
            UnsupervisedLearning,
            vec![
                "kmeans".into(),
                "pca".into(),
                "dbscan".into(),
                "hierarchical".into(),
            ],
        );

        self.domain_capabilities.insert(
            DeepLearning,
            vec![
                "autograd".into(),
                "lora".into(),
                "qlora".into(),
                "quantization".into(),
            ],
        );

        self.domain_capabilities.insert(
            Inference,
            vec![
                "model_serving".into(),
                "batching".into(),
                "moe_routing".into(),
            ],
        );

        self.domain_capabilities.insert(
            LinearAlgebra,
            vec![
                "vector_ops".into(),
                "matrix_ops".into(),
                "simd".into(),
                "gpu".into(),
            ],
        );

        self.domain_capabilities.insert(
            VectorSearch,
            vec![
                "vector_store".into(),
                "similarity_search".into(),
                "knn_search".into(),
            ],
        );

        self.domain_capabilities.insert(
            GraphAnalytics,
            vec![
                "pathfinding".into(),
                "centrality".into(),
                "community_detection".into(),
            ],
        );

        self.domain_capabilities.insert(
            PythonMigration,
            vec![
                "type_inference".into(),
                "sklearn_to_aprender".into(),
                "numpy_to_trueno".into(),
            ],
        );

        self.domain_capabilities.insert(
            CMigration,
            vec!["ownership_inference".into(), "unsafe_elimination".into()],
        );

        self.domain_capabilities.insert(
            ShellMigration,
            vec!["script_conversion".into(), "cli_generation".into()],
        );

        self.domain_capabilities.insert(
            DistributedCompute,
            vec![
                "work_stealing".into(),
                "cpu_executor".into(),
                "gpu_executor".into(),
                "remote_executor".into(),
            ],
        );

        self.domain_capabilities.insert(
            DataPipeline,
            vec!["csv".into(), "parquet".into(), "json".into(), "streaming".into()],
        );

        self.domain_capabilities.insert(
            ModelServing,
            vec![
                "model_serving".into(),
                "lambda".into(),
                "container".into(),
                "edge".into(),
            ],
        );

        self.domain_capabilities.insert(
            Testing,
            vec![
                "coverage_check".into(),
                "mutation_testing".into(),
                "tdg_scoring".into(),
            ],
        );

        self.domain_capabilities.insert(
            Profiling,
            vec![
                "syscall_trace".into(),
                "flamegraph".into(),
                "golden_trace_comparison".into(),
            ],
        );

        self.domain_capabilities.insert(
            Validation,
            vec![
                "privacy_audit".into(),
                "quality_gates".into(),
                "complexity_analysis".into(),
            ],
        );
    }

    /// Register all Sovereign AI Stack components
    fn register_sovereign_stack(&mut self) {
        // Layer 0: Compute Primitives
        self.register_trueno();
        self.register_trueno_db();
        self.register_trueno_graph();
        self.register_trueno_viz();

        // Layer 1: ML Algorithms
        self.register_aprender();

        // Layer 2: Training & Inference
        self.register_entrenar();
        self.register_realizar();

        // Layer 3: Transpilers
        self.register_depyler();
        self.register_decy();
        self.register_bashrs();
        self.register_ruchy();

        // Layer 4: Orchestration
        self.register_batuta();
        self.register_repartir();
        self.register_pforge();

        // Layer 5: Quality
        self.register_certeza();
        self.register_pmat();
        self.register_renacer();

        // Layer 6: Data & MLOps
        self.register_alimentar();
        self.register_pacha();
    }

    fn register_trueno(&mut self) {
        let component = StackComponent::new(
            "trueno",
            "0.7.3",
            StackLayer::Primitives,
            "SIMD-accelerated tensor operations with GPU support",
        )
        .with_capabilities(vec![
            Capability::new("vector_ops", CapabilityCategory::Compute)
                .with_description("SIMD-accelerated vector operations"),
            Capability::new("matrix_ops", CapabilityCategory::Compute)
                .with_description("High-performance matrix multiplication"),
            Capability::new("simd", CapabilityCategory::Compute)
                .with_description("SIMD auto-vectorization support"),
            Capability::new("gpu", CapabilityCategory::Compute)
                .with_description("GPU acceleration for large operations"),
        ]);
        self.register_component(component);
    }

    fn register_trueno_db(&mut self) {
        let component = StackComponent::new(
            "trueno-db",
            "0.3.3",
            StackLayer::Primitives,
            "GPU-first embedded analytics database with SIMD fallback",
        )
        .with_capabilities(vec![
            Capability::new("vector_store", CapabilityCategory::Storage)
                .with_description("Efficient vector storage"),
            Capability::new("similarity_search", CapabilityCategory::Storage)
                .with_description("Fast similarity search with HNSW"),
            Capability::new("persistence", CapabilityCategory::Storage)
                .with_description("Durable data persistence"),
            Capability::new("knn_search", CapabilityCategory::Storage)
                .with_description("K-nearest neighbor queries"),
        ]);
        self.register_component(component);
    }

    fn register_trueno_graph(&mut self) {
        let component = StackComponent::new(
            "trueno-graph",
            "0.1.1",
            StackLayer::Primitives,
            "GPU-first embedded graph database for code analysis",
        )
        .with_capabilities(vec![
            Capability::new("pathfinding", CapabilityCategory::Compute)
                .with_description("Dijkstra and A* pathfinding"),
            Capability::new("centrality", CapabilityCategory::Compute)
                .with_description("PageRank and betweenness centrality"),
            Capability::new("community_detection", CapabilityCategory::Compute)
                .with_description("Label propagation clustering"),
        ]);
        self.register_component(component);
    }

    fn register_trueno_viz(&mut self) {
        let component = StackComponent::new(
            "trueno-viz",
            "0.1.1",
            StackLayer::Primitives,
            "SIMD/GPU/WASM-accelerated visualization",
        )
        .with_capabilities(vec![
            Capability::new("visualization", CapabilityCategory::Compute)
                .with_description("High-performance data visualization"),
        ]);
        self.register_component(component);
    }

    fn register_aprender(&mut self) {
        let component = StackComponent::new(
            "aprender",
            "0.12.0",
            StackLayer::MlAlgorithms,
            "Next-generation machine learning library in pure Rust",
        )
        .with_capabilities(vec![
            // Supervised learning
            Capability::new("linear_regression", CapabilityCategory::MachineLearning),
            Capability::new("logistic_regression", CapabilityCategory::MachineLearning),
            Capability::new("decision_tree", CapabilityCategory::MachineLearning),
            Capability::new("random_forest", CapabilityCategory::MachineLearning),
            Capability::new("gbm", CapabilityCategory::MachineLearning)
                .with_description("Gradient boosting machines"),
            Capability::new("naive_bayes", CapabilityCategory::MachineLearning),
            Capability::new("knn", CapabilityCategory::MachineLearning)
                .with_description("K-nearest neighbors"),
            Capability::new("svm", CapabilityCategory::MachineLearning)
                .with_description("Support vector machines"),
            // Unsupervised learning
            Capability::new("kmeans", CapabilityCategory::MachineLearning),
            Capability::new("pca", CapabilityCategory::MachineLearning)
                .with_description("Principal component analysis"),
            Capability::new("dbscan", CapabilityCategory::MachineLearning),
            Capability::new("hierarchical", CapabilityCategory::MachineLearning),
            // Preprocessing
            Capability::new("standard_scaler", CapabilityCategory::MachineLearning),
            Capability::new("minmax_scaler", CapabilityCategory::MachineLearning),
            Capability::new("label_encoder", CapabilityCategory::MachineLearning),
            // Model selection
            Capability::new("train_test_split", CapabilityCategory::MachineLearning),
            Capability::new("cross_validate", CapabilityCategory::MachineLearning),
            Capability::new("grid_search", CapabilityCategory::MachineLearning),
        ]);
        self.register_component(component);
    }

    fn register_entrenar(&mut self) {
        let component = StackComponent::new(
            "entrenar",
            "0.2.1",
            StackLayer::MlPipeline,
            "Training library with autograd, LoRA, and quantization",
        )
        .with_capabilities(vec![
            Capability::new("autograd", CapabilityCategory::MachineLearning)
                .with_description("Automatic differentiation"),
            Capability::new("lora", CapabilityCategory::MachineLearning)
                .with_description("Low-rank adaptation"),
            Capability::new("qlora", CapabilityCategory::MachineLearning)
                .with_description("Quantized LoRA"),
            Capability::new("quantization", CapabilityCategory::MachineLearning)
                .with_description("Model quantization (int8, int4)"),
            Capability::new("model_merge", CapabilityCategory::MachineLearning),
            Capability::new("distillation", CapabilityCategory::MachineLearning),
        ]);
        self.register_component(component);
    }

    fn register_realizar(&mut self) {
        let component = StackComponent::new(
            "realizar",
            "0.2.1",
            StackLayer::MlPipeline,
            "Pure Rust ML inference engine - GGUF, safetensors, transformer serving",
        )
        .with_capabilities(vec![
            // Model formats
            Capability::new("gguf", CapabilityCategory::MachineLearning)
                .with_description("GGUF model format support"),
            Capability::new("safetensors", CapabilityCategory::MachineLearning)
                .with_description("Safetensors model loading"),
            Capability::new("apr_format", CapabilityCategory::MachineLearning)
                .with_description("Native .apr model format"),
            // Transformer inference
            Capability::new("transformer_serving", CapabilityCategory::MachineLearning)
                .with_description("LLM/transformer inference runtime"),
            Capability::new("kv_cache", CapabilityCategory::MachineLearning)
                .with_description("KV-cache for efficient generation"),
            Capability::new("continuous_batching", CapabilityCategory::MachineLearning)
                .with_description("Dynamic request batching"),
            // Production features
            Capability::new("model_serving", CapabilityCategory::MachineLearning)
                .with_description("Production model serving"),
            Capability::new("moe_routing", CapabilityCategory::MachineLearning)
                .with_description("Mixture-of-experts routing"),
            Capability::new("circuit_breaker", CapabilityCategory::MachineLearning)
                .with_description("Fault tolerance"),
            // Deployment targets
            Capability::new("lambda", CapabilityCategory::Distribution)
                .with_description("AWS Lambda (53,000x faster cold start)"),
            Capability::new("container", CapabilityCategory::Distribution)
                .with_description("Container deployment"),
            Capability::new("edge", CapabilityCategory::Distribution)
                .with_description("Edge deployment"),
        ]);
        self.register_component(component);
    }

    fn register_depyler(&mut self) {
        let component = StackComponent::new(
            "depyler",
            "0.1.0",
            StackLayer::Transpilers,
            "Python to Rust transpiler with ML oracle",
        )
        .with_capabilities(vec![
            Capability::new("type_inference", CapabilityCategory::Transpilation)
                .with_description("Python type inference"),
            Capability::new("sklearn_to_aprender", CapabilityCategory::Transpilation)
                .with_description("sklearn to aprender conversion"),
            Capability::new("numpy_to_trueno", CapabilityCategory::Transpilation)
                .with_description("NumPy to trueno conversion"),
        ]);
        self.register_component(component);
    }

    fn register_decy(&mut self) {
        let component = StackComponent::new(
            "decy",
            "0.1.0",
            StackLayer::Transpilers,
            "C/C++ to Rust transpiler",
        )
        .with_capabilities(vec![
            Capability::new("ownership_inference", CapabilityCategory::Transpilation)
                .with_description("Infer Rust ownership from C code"),
            Capability::new("unsafe_elimination", CapabilityCategory::Transpilation)
                .with_description("Eliminate unsafe blocks where possible"),
        ]);
        self.register_component(component);
    }

    fn register_bashrs(&mut self) {
        let component = StackComponent::new(
            "bashrs",
            "6.41.0",
            StackLayer::Transpilers,
            "Rust-to-Shell transpiler for deterministic bootstrap scripts",
        )
        .with_capabilities(vec![
            Capability::new("rust_to_shell", CapabilityCategory::Transpilation)
                .with_description("Transpile Rust to portable shell scripts"),
            Capability::new("bootstrap_scripts", CapabilityCategory::Transpilation)
                .with_description("Generate deterministic bootstrap scripts"),
            Capability::new("cross_platform_shell", CapabilityCategory::Transpilation)
                .with_description("Generate POSIX-compliant shell code"),
        ]);
        self.register_component(component);
    }

    fn register_ruchy(&mut self) {
        let component = StackComponent::new(
            "ruchy",
            "3.213.0",
            StackLayer::Transpilers,
            "Systems scripting language that transpiles to idiomatic Rust",
        )
        .with_capabilities(vec![
            Capability::new("script_to_rust", CapabilityCategory::Transpilation)
                .with_description("Transpile scripts to idiomatic Rust"),
            Capability::new("shell_semantics", CapabilityCategory::Transpilation)
                .with_description("Shell-like semantics with Rust safety"),
            Capability::new("wasm_target", CapabilityCategory::Transpilation)
                .with_description("Compile to WebAssembly"),
            Capability::new("extreme_tdd", CapabilityCategory::Validation)
                .with_description("Built-in extreme TDD methodology"),
        ]);
        self.register_component(component);
    }

    fn register_batuta(&mut self) {
        let component = StackComponent::new(
            "batuta",
            "0.1.0",
            StackLayer::Orchestration,
            "Workflow orchestrator for transpilation pipelines",
        )
        .with_capabilities(vec![
            Capability::new("analysis", CapabilityCategory::Validation),
            Capability::new("transpilation", CapabilityCategory::Transpilation),
            Capability::new("optimization", CapabilityCategory::Compute),
            Capability::new("validation", CapabilityCategory::Validation),
            Capability::new("deployment", CapabilityCategory::Distribution),
        ]);
        self.register_component(component);
    }

    fn register_repartir(&mut self) {
        let component = StackComponent::new(
            "repartir",
            "1.0.0",
            StackLayer::Orchestration,
            "Sovereign AI-grade distributed computing primitives",
        )
        .with_capabilities(vec![
            Capability::new("work_stealing", CapabilityCategory::Distribution)
                .with_description("Work-stealing scheduler"),
            Capability::new("cpu_executor", CapabilityCategory::Distribution)
                .with_description("CPU task execution"),
            Capability::new("gpu_executor", CapabilityCategory::Distribution)
                .with_description("GPU task execution"),
            Capability::new("remote_executor", CapabilityCategory::Distribution)
                .with_description("Remote/distributed execution"),
        ]);
        self.register_component(component);
    }

    fn register_certeza(&mut self) {
        let component = StackComponent::new(
            "certeza",
            "0.1.0",
            StackLayer::Quality,
            "Quality validation framework",
        )
        .with_capabilities(vec![
            Capability::new("coverage_check", CapabilityCategory::Validation),
            Capability::new("mutation_testing", CapabilityCategory::Validation),
            Capability::new("tdg_scoring", CapabilityCategory::Validation)
                .with_description("Technical debt gauge scoring"),
            Capability::new("privacy_audit", CapabilityCategory::Validation),
        ]);
        self.register_component(component);
    }

    fn register_pmat(&mut self) {
        let component = StackComponent::new(
            "pmat",
            "2.205.0",
            StackLayer::Quality,
            "Zero-config AI context generation and code quality toolkit",
        )
        .with_capabilities(vec![
            Capability::new("complexity_analysis", CapabilityCategory::Validation)
                .with_description("Cyclomatic and cognitive complexity analysis"),
            Capability::new("satd_detection", CapabilityCategory::Validation)
                .with_description("Self-admitted technical debt detection"),
            Capability::new("quality_gates", CapabilityCategory::Validation)
                .with_description("Tiered quality enforcement"),
            Capability::new("repo_health", CapabilityCategory::Validation)
                .with_description("Repository health scoring (0-125)"),
            Capability::new("mcp_server", CapabilityCategory::Distribution)
                .with_description("MCP protocol server for AI agents"),
            Capability::new("multi_language", CapabilityCategory::Validation)
                .with_description("17+ programming language support"),
        ]);
        self.register_component(component);
    }

    fn register_pforge(&mut self) {
        let component = StackComponent::new(
            "pforge",
            "0.1.2",
            StackLayer::Orchestration,
            "Zero-boilerplate MCP server framework with rust-mcp-sdk",
        )
        .with_capabilities(vec![
            // MCP server generation
            Capability::new("mcp_codegen", CapabilityCategory::Transpilation)
                .with_description("Declarative MCP server code generation"),
            Capability::new("mcp_runtime", CapabilityCategory::Distribution)
                .with_description("MCP protocol runtime via rust-mcp-sdk"),
            Capability::new("zero_boilerplate", CapabilityCategory::Transpilation)
                .with_description("Macro-based boilerplate elimination"),
            // Agent capabilities
            Capability::new("tool_orchestration", CapabilityCategory::Distribution)
                .with_description("AI agent tool-use orchestration"),
            Capability::new("resource_provider", CapabilityCategory::Distribution)
                .with_description("MCP resource exposure for agents"),
            Capability::new("prompt_templates", CapabilityCategory::Distribution)
                .with_description("Reusable prompt template serving"),
            // Quality
            Capability::new("extreme_tdd", CapabilityCategory::Validation)
                .with_description("Built-in extreme TDD methodology"),
        ]);
        self.register_component(component);
    }

    fn register_renacer(&mut self) {
        let component = StackComponent::new(
            "renacer",
            "0.6.5",
            StackLayer::Quality,
            "Pure Rust system call tracer with source-aware correlation",
        )
        .with_capabilities(vec![
            Capability::new("syscall_trace", CapabilityCategory::Profiling)
                .with_description("System call tracing"),
            Capability::new("flamegraph", CapabilityCategory::Profiling)
                .with_description("Flamegraph generation"),
            Capability::new("golden_trace_comparison", CapabilityCategory::Profiling)
                .with_description("Compare traces for semantic equivalence"),
        ]);
        self.register_component(component);
    }

    fn register_alimentar(&mut self) {
        let component = StackComponent::new(
            "alimentar",
            "0.1.0",
            StackLayer::Data,
            "Data loading and preprocessing in pure Rust",
        )
        .with_capabilities(vec![
            Capability::new("csv", CapabilityCategory::Storage),
            Capability::new("parquet", CapabilityCategory::Storage),
            Capability::new("json", CapabilityCategory::Storage),
            Capability::new("streaming", CapabilityCategory::Storage)
                .with_description("Streaming data loading"),
        ]);
        self.register_component(component);
    }

    fn register_pacha(&mut self) {
        let component = StackComponent::new(
            "pacha",
            "0.1.0",
            StackLayer::Data,
            "Model, Data and Recipe Registry for MLOps",
        )
        .with_capabilities(vec![
            // Model registry
            Capability::new("model_versioning", CapabilityCategory::Storage)
                .with_description("Semantic versioned model artifacts"),
            Capability::new("model_lineage", CapabilityCategory::Storage)
                .with_description("Model lineage tracking"),
            Capability::new("artifact_storage", CapabilityCategory::Storage)
                .with_description("Artifact storage and retrieval"),
            // Data registry
            Capability::new("dataset_versioning", CapabilityCategory::Storage)
                .with_description("Dataset version control"),
            Capability::new("data_lineage", CapabilityCategory::Storage)
                .with_description("Data lineage and provenance"),
            // Recipe registry
            Capability::new("recipe_management", CapabilityCategory::Storage)
                .with_description("Training recipe versioning"),
            Capability::new("experiment_tracking", CapabilityCategory::Validation)
                .with_description("Experiment metrics and comparison"),
        ]);
        self.register_component(component);
    }

    /// Register integration patterns
    fn register_integration_patterns(&mut self) {
        // ML Pipeline patterns
        self.integrations.push(IntegrationPattern {
            from: "aprender".into(),
            to: "realizar".into(),
            pattern_name: "model_export".into(),
            description: "Export trained model for serving".into(),
            code_template: Some(
                r#"// Train with aprender
let model = RandomForest::new().fit(&X, &y)?;
// Export for realizar
model.save_apr("model.apr")?;"#
                    .into(),
            ),
        });

        self.integrations.push(IntegrationPattern {
            from: "aprender".into(),
            to: "trueno".into(),
            pattern_name: "tensor_backend".into(),
            description: "Use trueno as compute backend for aprender".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "entrenar".into(),
            to: "realizar".into(),
            pattern_name: "training_to_inference".into(),
            description: "Export fine-tuned model for inference".into(),
            code_template: None,
        });

        // Transpilation patterns
        self.integrations.push(IntegrationPattern {
            from: "depyler".into(),
            to: "aprender".into(),
            pattern_name: "sklearn_convert".into(),
            description: "Convert sklearn code to aprender".into(),
            code_template: Some(
                r#"// depyler converts:
// from sklearn.ensemble import RandomForestClassifier
// to:
use aprender::tree::RandomForestClassifier;"#
                    .into(),
            ),
        });

        self.integrations.push(IntegrationPattern {
            from: "depyler".into(),
            to: "trueno".into(),
            pattern_name: "numpy_convert".into(),
            description: "Convert numpy operations to trueno".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "batuta".into(),
            to: "depyler".into(),
            pattern_name: "orchestrated_transpilation".into(),
            description: "Batuta orchestrates depyler for Python projects".into(),
            code_template: None,
        });

        // Distribution patterns
        self.integrations.push(IntegrationPattern {
            from: "repartir".into(),
            to: "entrenar".into(),
            pattern_name: "distributed_training".into(),
            description: "Distribute training across nodes".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "repartir".into(),
            to: "trueno".into(),
            pattern_name: "distributed_compute".into(),
            description: "Distribute tensor operations".into(),
            code_template: None,
        });

        // Data patterns
        self.integrations.push(IntegrationPattern {
            from: "alimentar".into(),
            to: "aprender".into(),
            pattern_name: "data_loading".into(),
            description: "Load data for ML training".into(),
            code_template: Some(
                r#"use alimentar::CsvReader;
use aprender::Dataset;

let data = CsvReader::from_path("data.csv")?.load()?;
let dataset = Dataset::from(data);"#
                    .into(),
            ),
        });

        // Ruchy patterns
        self.integrations.push(IntegrationPattern {
            from: "ruchy".into(),
            to: "batuta".into(),
            pattern_name: "scripted_orchestration".into(),
            description: "Use ruchy scripts to drive batuta pipelines".into(),
            code_template: Some(
                r#"// ruchy script for pipeline automation
let result = batuta::analyze("./project")?;
let transpiled = batuta::transpile(&result)?;
batuta::validate(&transpiled)?;"#
                    .into(),
            ),
        });

        self.integrations.push(IntegrationPattern {
            from: "ruchy".into(),
            to: "renacer".into(),
            pattern_name: "scripted_tracing".into(),
            description: "Automate syscall tracing with ruchy scripts".into(),
            code_template: None,
        });

        // Bashrs patterns
        self.integrations.push(IntegrationPattern {
            from: "bashrs".into(),
            to: "batuta".into(),
            pattern_name: "bootstrap_generation".into(),
            description: "Generate shell bootstrap scripts from Rust".into(),
            code_template: Some(
                r#"// Generate portable install script
bashrs::transpile! {
    check_rust_installed();
    cargo_install("batuta");
    batuta_init("./project");
}"#
                    .into(),
            ),
        });

        self.integrations.push(IntegrationPattern {
            from: "bashrs".into(),
            to: "repartir".into(),
            pattern_name: "cluster_bootstrap".into(),
            description: "Generate shell scripts for cluster node setup".into(),
            code_template: None,
        });

        // Quality patterns
        self.integrations.push(IntegrationPattern {
            from: "renacer".into(),
            to: "certeza".into(),
            pattern_name: "trace_validation".into(),
            description: "Validate performance with golden traces".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "pmat".into(),
            to: "batuta".into(),
            pattern_name: "quality_gating".into(),
            description: "Gate pipeline on quality metrics".into(),
            code_template: None,
        });
    }

    /// Register a component and update indices
    pub fn register_component(&mut self, component: StackComponent) {
        let name = component.name.clone();

        // Update capability index
        for cap in &component.capabilities {
            self.capability_index
                .entry(cap.name.clone())
                .or_default()
                .push(name.clone());
        }

        self.components.insert(name, component);
    }

    /// Get a component by name
    pub fn get_component(&self, name: &str) -> Option<&StackComponent> {
        self.components.get(name)
    }

    /// Get all components
    pub fn components(&self) -> impl Iterator<Item = &StackComponent> {
        self.components.values()
    }

    /// Get all component names
    pub fn component_names(&self) -> impl Iterator<Item = &String> {
        self.components.keys()
    }

    /// Get components in a specific layer
    pub fn components_in_layer(&self, layer: StackLayer) -> Vec<&StackComponent> {
        self.components
            .values()
            .filter(|c| c.layer == layer)
            .collect()
    }

    /// Find components with a specific capability
    pub fn find_by_capability(&self, capability: &str) -> Vec<&StackComponent> {
        self.capability_index
            .get(capability)
            .map(|names| {
                names
                    .iter()
                    .filter_map(|n| self.components.get(n))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find components for a problem domain
    pub fn find_by_domain(&self, domain: ProblemDomain) -> Vec<&StackComponent> {
        self.domain_capabilities
            .get(&domain)
            .map(|caps| {
                caps.iter()
                    .flat_map(|cap| self.find_by_capability(cap))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    /// Get integration patterns from a component
    pub fn integrations_from(&self, component: &str) -> Vec<&IntegrationPattern> {
        self.integrations
            .iter()
            .filter(|p| p.from == component)
            .collect()
    }

    /// Get integration patterns to a component
    pub fn integrations_to(&self, component: &str) -> Vec<&IntegrationPattern> {
        self.integrations
            .iter()
            .filter(|p| p.to == component)
            .collect()
    }

    /// Get integration pattern between two components
    pub fn get_integration(&self, from: &str, to: &str) -> Option<&IntegrationPattern> {
        self.integrations
            .iter()
            .find(|p| p.from == from && p.to == to)
    }

    /// Get all capabilities in the graph
    pub fn all_capabilities(&self) -> impl Iterator<Item = &String> {
        self.capability_index.keys()
    }

    /// Get total number of components
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Get total number of capabilities
    pub fn capability_count(&self) -> usize {
        self.capability_index.len()
    }

    /// Get total number of integration patterns
    pub fn integration_count(&self) -> usize {
        self.integrations.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Basic Knowledge Graph Tests
    // =========================================================================

    #[test]
    fn test_knowledge_graph_new() {
        let graph = KnowledgeGraph::new();
        assert_eq!(graph.component_count(), 0);
        assert_eq!(graph.integration_count(), 0);
    }

    #[test]
    fn test_knowledge_graph_default() {
        let graph = KnowledgeGraph::default();
        assert_eq!(graph.component_count(), 0);
    }

    #[test]
    fn test_knowledge_graph_sovereign_stack() {
        let graph = KnowledgeGraph::sovereign_stack();
        // Should have all 16 components
        assert!(graph.component_count() >= 15);
        // Should have integration patterns
        assert!(graph.integration_count() > 0);
    }

    // =========================================================================
    // Component Registration Tests
    // =========================================================================

    #[test]
    fn test_register_component() {
        let mut graph = KnowledgeGraph::new();
        let comp = StackComponent::new("test", "1.0.0", StackLayer::Primitives, "Test component")
            .with_capability(Capability::new("test_cap", CapabilityCategory::Compute));

        graph.register_component(comp);

        assert_eq!(graph.component_count(), 1);
        assert!(graph.get_component("test").is_some());
    }

    #[test]
    fn test_get_component() {
        let graph = KnowledgeGraph::sovereign_stack();

        let trueno = graph.get_component("trueno");
        assert!(trueno.is_some());
        let trueno = trueno.unwrap();
        assert_eq!(trueno.name, "trueno");
        assert_eq!(trueno.version, "0.7.3");
        assert_eq!(trueno.layer, StackLayer::Primitives);
    }

    #[test]
    fn test_get_component_not_found() {
        let graph = KnowledgeGraph::sovereign_stack();
        assert!(graph.get_component("nonexistent").is_none());
    }

    #[test]
    fn test_component_names() {
        let graph = KnowledgeGraph::sovereign_stack();
        let names: Vec<_> = graph.component_names().collect();
        assert!(names.contains(&&"trueno".to_string()));
        assert!(names.contains(&&"aprender".to_string()));
        assert!(names.contains(&&"repartir".to_string()));
    }

    // =========================================================================
    // Layer Query Tests
    // =========================================================================

    #[test]
    fn test_components_in_layer_primitives() {
        let graph = KnowledgeGraph::sovereign_stack();
        let primitives = graph.components_in_layer(StackLayer::Primitives);

        assert!(primitives.len() >= 3); // trueno, trueno-db, trueno-graph, trueno-viz
        assert!(primitives.iter().any(|c| c.name == "trueno"));
        assert!(primitives.iter().any(|c| c.name == "trueno-db"));
        assert!(primitives.iter().any(|c| c.name == "trueno-graph"));
    }

    #[test]
    fn test_components_in_layer_ml_algorithms() {
        let graph = KnowledgeGraph::sovereign_stack();
        let ml = graph.components_in_layer(StackLayer::MlAlgorithms);

        assert_eq!(ml.len(), 1);
        assert_eq!(ml[0].name, "aprender");
    }

    #[test]
    fn test_components_in_layer_transpilers() {
        let graph = KnowledgeGraph::sovereign_stack();
        let transpilers = graph.components_in_layer(StackLayer::Transpilers);

        assert_eq!(transpilers.len(), 4);
        let names: Vec<_> = transpilers.iter().map(|c| &c.name).collect();
        assert!(names.contains(&&"depyler".to_string()));
        assert!(names.contains(&&"decy".to_string()));
        assert!(names.contains(&&"bashrs".to_string()));
        assert!(names.contains(&&"ruchy".to_string()));
    }

    // =========================================================================
    // Capability Query Tests
    // =========================================================================

    #[test]
    fn test_find_by_capability_simd() {
        let graph = KnowledgeGraph::sovereign_stack();
        let simd_components = graph.find_by_capability("simd");

        assert!(!simd_components.is_empty());
        assert!(simd_components.iter().any(|c| c.name == "trueno"));
    }

    #[test]
    fn test_find_by_capability_random_forest() {
        let graph = KnowledgeGraph::sovereign_stack();
        let rf_components = graph.find_by_capability("random_forest");

        assert!(!rf_components.is_empty());
        assert!(rf_components.iter().any(|c| c.name == "aprender"));
    }

    #[test]
    fn test_find_by_capability_model_serving() {
        let graph = KnowledgeGraph::sovereign_stack();
        let serving = graph.find_by_capability("model_serving");

        assert!(!serving.is_empty());
        assert!(serving.iter().any(|c| c.name == "realizar"));
    }

    #[test]
    fn test_find_by_capability_not_found() {
        let graph = KnowledgeGraph::sovereign_stack();
        let result = graph.find_by_capability("nonexistent_capability");
        assert!(result.is_empty());
    }

    // =========================================================================
    // Domain Query Tests
    // =========================================================================

    #[test]
    fn test_find_by_domain_supervised_learning() {
        let graph = KnowledgeGraph::sovereign_stack();
        let components = graph.find_by_domain(ProblemDomain::SupervisedLearning);

        assert!(!components.is_empty());
        assert!(components.iter().any(|c| c.name == "aprender"));
    }

    #[test]
    fn test_find_by_domain_linear_algebra() {
        let graph = KnowledgeGraph::sovereign_stack();
        let components = graph.find_by_domain(ProblemDomain::LinearAlgebra);

        assert!(!components.is_empty());
        assert!(components.iter().any(|c| c.name == "trueno"));
    }

    #[test]
    fn test_find_by_domain_python_migration() {
        let graph = KnowledgeGraph::sovereign_stack();
        let components = graph.find_by_domain(ProblemDomain::PythonMigration);

        assert!(!components.is_empty());
        assert!(components.iter().any(|c| c.name == "depyler"));
    }

    #[test]
    fn test_find_by_domain_distributed_compute() {
        let graph = KnowledgeGraph::sovereign_stack();
        let components = graph.find_by_domain(ProblemDomain::DistributedCompute);

        assert!(!components.is_empty());
        assert!(components.iter().any(|c| c.name == "repartir"));
    }

    // =========================================================================
    // Integration Pattern Tests
    // =========================================================================

    #[test]
    fn test_integrations_from() {
        let graph = KnowledgeGraph::sovereign_stack();
        let patterns = graph.integrations_from("aprender");

        assert!(!patterns.is_empty());
        assert!(patterns.iter().any(|p| p.to == "realizar"));
    }

    #[test]
    fn test_integrations_to() {
        let graph = KnowledgeGraph::sovereign_stack();
        let patterns = graph.integrations_to("aprender");

        assert!(!patterns.is_empty());
        assert!(patterns.iter().any(|p| p.from == "depyler"));
    }

    #[test]
    fn test_get_integration() {
        let graph = KnowledgeGraph::sovereign_stack();
        let pattern = graph.get_integration("aprender", "realizar");

        assert!(pattern.is_some());
        let pattern = pattern.unwrap();
        assert_eq!(pattern.pattern_name, "model_export");
    }

    #[test]
    fn test_get_integration_not_found() {
        let graph = KnowledgeGraph::sovereign_stack();
        let pattern = graph.get_integration("trueno", "bashrs");
        assert!(pattern.is_none());
    }

    #[test]
    fn test_integration_has_code_template() {
        let graph = KnowledgeGraph::sovereign_stack();
        let pattern = graph.get_integration("aprender", "realizar").unwrap();
        assert!(pattern.code_template.is_some());
    }

    // =========================================================================
    // Component Detail Tests
    // =========================================================================

    #[test]
    fn test_trueno_capabilities() {
        let graph = KnowledgeGraph::sovereign_stack();
        let trueno = graph.get_component("trueno").unwrap();

        assert!(trueno.has_capability("simd"));
        assert!(trueno.has_capability("gpu"));
        assert!(trueno.has_capability("vector_ops"));
        assert!(trueno.has_capability("matrix_ops"));
    }

    #[test]
    fn test_aprender_capabilities() {
        let graph = KnowledgeGraph::sovereign_stack();
        let aprender = graph.get_component("aprender").unwrap();

        // Supervised
        assert!(aprender.has_capability("random_forest"));
        assert!(aprender.has_capability("linear_regression"));
        assert!(aprender.has_capability("gbm"));
        // Unsupervised
        assert!(aprender.has_capability("kmeans"));
        assert!(aprender.has_capability("pca"));
        // Preprocessing
        assert!(aprender.has_capability("standard_scaler"));
    }

    #[test]
    fn test_repartir_capabilities() {
        let graph = KnowledgeGraph::sovereign_stack();
        let repartir = graph.get_component("repartir").unwrap();

        assert!(repartir.has_capability("work_stealing"));
        assert!(repartir.has_capability("cpu_executor"));
        assert!(repartir.has_capability("gpu_executor"));
    }

    #[test]
    fn test_realizar_capabilities() {
        let graph = KnowledgeGraph::sovereign_stack();
        let realizar = graph.get_component("realizar").unwrap();

        assert!(realizar.has_capability("model_serving"));
        assert!(realizar.has_capability("gguf"));
        assert!(realizar.has_capability("safetensors"));
        assert!(realizar.has_capability("transformer_serving"));
        assert!(realizar.has_capability("continuous_batching"));
        assert!(realizar.has_capability("lambda"));
    }

    // =========================================================================
    // Statistics Tests
    // =========================================================================

    #[test]
    fn test_all_capabilities() {
        let graph = KnowledgeGraph::sovereign_stack();
        let caps: Vec<_> = graph.all_capabilities().collect();

        assert!(caps.len() > 30); // Many capabilities registered
        assert!(caps.contains(&&"simd".to_string()));
        assert!(caps.contains(&&"random_forest".to_string()));
    }

    #[test]
    fn test_capability_count() {
        let graph = KnowledgeGraph::sovereign_stack();
        assert!(graph.capability_count() > 30);
    }

    // =========================================================================
    // Version Tests
    // =========================================================================

    #[test]
    fn test_component_versions() {
        let graph = KnowledgeGraph::sovereign_stack();

        assert_eq!(graph.get_component("trueno").unwrap().version, "0.7.3");
        assert_eq!(graph.get_component("trueno-db").unwrap().version, "0.3.3");
        assert_eq!(graph.get_component("trueno-graph").unwrap().version, "0.1.1");
        assert_eq!(graph.get_component("aprender").unwrap().version, "0.12.0");
        assert_eq!(graph.get_component("repartir").unwrap().version, "1.0.0");
        assert_eq!(graph.get_component("renacer").unwrap().version, "0.6.5");
    }
}
