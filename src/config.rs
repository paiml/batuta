use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Batuta project configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatutaConfig {
    /// Configuration file version
    pub version: String,

    /// Project metadata
    pub project: ProjectConfig,

    /// Source code configuration
    pub source: SourceConfig,

    /// Transpilation settings
    pub transpilation: TranspilationConfig,

    /// Optimization settings
    pub optimization: OptimizationConfig,

    /// Validation settings
    pub validation: ValidationConfig,

    /// Build settings
    pub build: BuildConfig,
}

impl Default for BatutaConfig {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            project: ProjectConfig::default(),
            source: SourceConfig::default(),
            transpilation: TranspilationConfig::default(),
            optimization: OptimizationConfig::default(),
            validation: ValidationConfig::default(),
            build: BuildConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    /// Project name
    pub name: String,

    /// Project description
    pub description: Option<String>,

    /// Primary language of source project
    pub primary_language: Option<String>,

    /// Authors
    pub authors: Vec<String>,

    /// License
    pub license: Option<String>,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            name: "untitled".to_string(),
            description: None,
            primary_language: None,
            authors: vec![],
            license: Some("MIT".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    /// Source code directory (relative to config file)
    pub path: PathBuf,

    /// Files/directories to exclude
    pub exclude: Vec<String>,

    /// Files/directories to include (overrides exclude)
    pub include: Vec<String>,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("."),
            exclude: vec![
                ".git".to_string(),
                "target".to_string(),
                "build".to_string(),
                "dist".to_string(),
                "node_modules".to_string(),
                "__pycache__".to_string(),
                "*.pyc".to_string(),
                ".venv".to_string(),
                "venv".to_string(),
            ],
            include: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranspilationConfig {
    /// Output directory for generated Rust code
    pub output_dir: PathBuf,

    /// Enable incremental compilation
    pub incremental: bool,

    /// Enable caching
    pub cache: bool,

    /// Generate Ruchy instead of pure Rust
    pub use_ruchy: bool,

    /// Ruchy strictness level (permissive, gradual, strict)
    pub ruchy_strictness: Option<String>,

    /// Specific modules to transpile (empty = all)
    pub modules: Vec<String>,

    /// Tool-specific settings
    pub decy: DecyConfig,
    pub depyler: DepylerConfig,
    pub bashrs: BashrsConfig,
}

impl Default for TranspilationConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./rust-output"),
            incremental: true,
            cache: true,
            use_ruchy: false,
            ruchy_strictness: Some("gradual".to_string()),
            modules: vec![],
            decy: DecyConfig::default(),
            depyler: DepylerConfig::default(),
            bashrs: BashrsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecyConfig {
    /// Enable ownership inference
    pub ownership_inference: bool,

    /// Generate actionable diagnostics
    pub actionable_diagnostics: bool,

    /// Use StaticFixer integration
    pub use_static_fixer: bool,
}

impl Default for DecyConfig {
    fn default() -> Self {
        Self {
            ownership_inference: true,
            actionable_diagnostics: true,
            use_static_fixer: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepylerConfig {
    /// Enable type inference
    pub type_inference: bool,

    /// Convert NumPy to Trueno
    pub numpy_to_trueno: bool,

    /// Convert sklearn to Aprender
    pub sklearn_to_aprender: bool,

    /// Convert PyTorch to Realizar
    pub pytorch_to_realizar: bool,
}

impl Default for DepylerConfig {
    fn default() -> Self {
        Self {
            type_inference: true,
            numpy_to_trueno: true,
            sklearn_to_aprender: true,
            pytorch_to_realizar: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BashrsConfig {
    /// Target shell compatibility
    pub target_shell: String,

    /// Generate CLI using clap
    pub use_clap: bool,
}

impl Default for BashrsConfig {
    fn default() -> Self {
        Self {
            target_shell: "bash".to_string(),
            use_clap: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Optimization profile (fast, balanced, aggressive)
    pub profile: String,

    /// Enable SIMD vectorization
    pub enable_simd: bool,

    /// Enable GPU acceleration
    pub enable_gpu: bool,

    /// GPU dispatch threshold (matrix size)
    pub gpu_threshold: usize,

    /// Use Mixture-of-Experts routing
    pub use_moe_routing: bool,

    /// Trueno backend preferences
    pub trueno: TruenoConfig,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            profile: "balanced".to_string(),
            enable_simd: true,
            enable_gpu: false,
            gpu_threshold: 500,
            use_moe_routing: false,
            trueno: TruenoConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruenoConfig {
    /// Preferred backends in priority order
    pub backends: Vec<String>,

    /// Enable adaptive threshold learning
    pub adaptive_thresholds: bool,

    /// Initial CPU threshold
    pub cpu_threshold: usize,
}

impl Default for TruenoConfig {
    fn default() -> Self {
        Self {
            backends: vec!["simd".to_string(), "cpu".to_string()],
            adaptive_thresholds: false,
            cpu_threshold: 500,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable syscall tracing
    pub trace_syscalls: bool,

    /// Run original test suite
    pub run_original_tests: bool,

    /// Generate diff output
    pub diff_output: bool,

    /// Run benchmarks
    pub benchmark: bool,

    /// Renacer configuration
    pub renacer: RenacerConfig,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            trace_syscalls: true,
            run_original_tests: true,
            diff_output: true,
            benchmark: false,
            renacer: RenacerConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenacerConfig {
    /// Syscalls to trace (empty = all)
    pub trace_syscalls: Vec<String>,

    /// Output format
    pub output_format: String,
}

impl Default for RenacerConfig {
    fn default() -> Self {
        Self {
            trace_syscalls: vec![],
            output_format: "json".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Build in release mode
    pub release: bool,

    /// Target platform (empty = native)
    pub target: Option<String>,

    /// Enable WebAssembly build
    pub wasm: bool,

    /// Additional cargo flags
    pub cargo_flags: Vec<String>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            release: true,
            target: None,
            wasm: false,
            cargo_flags: vec![],
        }
    }
}

impl BatutaConfig {
    /// Load configuration from TOML file
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to TOML file
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Create a new config from project analysis
    pub fn from_analysis(analysis: &crate::types::ProjectAnalysis) -> Self {
        let mut config = Self::default();

        // Set project name from directory
        if let Some(name) = analysis.root_path.file_name() {
            config.project.name = name.to_string_lossy().to_string();
        }

        // Set primary language
        if let Some(lang) = &analysis.primary_language {
            config.project.primary_language = Some(format!("{}", lang));
        }

        // Configure transpilation based on detected dependencies
        if analysis.has_ml_dependencies() {
            config.transpilation.depyler.numpy_to_trueno = true;
            config.transpilation.depyler.sklearn_to_aprender = true;
            config.transpilation.depyler.pytorch_to_realizar = true;
        }

        config
    }
}
