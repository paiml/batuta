use anyhow::{Context as AnyhowContext, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[cfg(feature = "native")]
use tracing::{debug, info, warn};

#[cfg(feature = "native")]
use walkdir::WalkDir;

// Stub macros for WASM build
#[cfg(not(feature = "native"))]
macro_rules! info {
    ($($arg:tt)*) => {{}};
}

#[cfg(not(feature = "native"))]
macro_rules! debug {
    ($($arg:tt)*) => {{}};
}

#[cfg(not(feature = "native"))]
macro_rules! warn {
    ($($arg:tt)*) => {{}};
}

use crate::numpy_converter::{NumPyConverter, NumPyOp};
use crate::pytorch_converter::{PyTorchConverter, PyTorchOperation};
use crate::sklearn_converter::{SklearnAlgorithm, SklearnConverter};

/// Context passed between pipeline stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineContext {
    /// Input path (source project)
    pub input_path: PathBuf,

    /// Output path (transpiled Rust project)
    pub output_path: PathBuf,

    /// Primary language detected
    pub primary_language: Option<crate::types::Language>,

    /// Transpiled file mappings (source -> output)
    pub file_mappings: Vec<(PathBuf, PathBuf)>,

    /// Optimization passes applied
    pub optimizations: Vec<String>,

    /// Validation results
    pub validation_results: Vec<ValidationResult>,

    /// Metadata accumulated during pipeline
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl PipelineContext {
    pub fn new(input_path: PathBuf, output_path: PathBuf) -> Self {
        Self {
            input_path,
            output_path,
            primary_language: None,
            file_mappings: Vec::new(),
            optimizations: Vec::new(),
            validation_results: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Get final output artifacts
    pub fn output(&self) -> PipelineOutput {
        PipelineOutput {
            output_path: self.output_path.clone(),
            file_mappings: self.file_mappings.clone(),
            optimizations: self.optimizations.clone(),
            validation_passed: self.validation_results.iter().all(|v| v.passed),
        }
    }
}

/// Validation result from a pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub stage: String,
    pub passed: bool,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

/// Final output from the pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineOutput {
    pub output_path: PathBuf,
    pub file_mappings: Vec<(PathBuf, PathBuf)>,
    pub optimizations: Vec<String>,
    pub validation_passed: bool,
}

/// Validation strategy for pipeline stages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationStrategy {
    /// Stop on first error (Jidoka principle)
    StopOnError,
    /// Continue on errors but collect them
    ContinueOnError,
    /// Skip validation
    None,
}

/// Trait for pipeline stages
#[async_trait::async_trait]
pub trait PipelineStage: Send + Sync {
    /// Name of this stage
    fn name(&self) -> &str;

    /// Execute this stage
    async fn execute(&self, ctx: PipelineContext) -> Result<PipelineContext>;

    /// Validate the output of this stage
    fn validate(&self, _ctx: &PipelineContext) -> Result<ValidationResult> {
        // Default: always pass
        Ok(ValidationResult {
            stage: self.name().to_string(),
            passed: true,
            message: "No validation configured".to_string(),
            details: None,
        })
    }
}

/// Main transpilation pipeline
pub struct TranspilationPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
    validation: ValidationStrategy,
}

impl TranspilationPipeline {
    pub fn new(validation: ValidationStrategy) -> Self {
        Self {
            stages: Vec::new(),
            validation,
        }
    }

    /// Add a stage to the pipeline
    pub fn add_stage(mut self, stage: Box<dyn PipelineStage>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Run the complete pipeline
    pub async fn run(&self, input: &Path, output: &Path) -> Result<PipelineOutput> {
        info!("Starting pipeline with {} stages", self.stages.len());

        let mut ctx = PipelineContext::new(input.to_path_buf(), output.to_path_buf());

        for (idx, stage) in self.stages.iter().enumerate() {
            info!("Running stage {}/{}: {}", idx + 1, self.stages.len(), stage.name());

            // Execute stage
            ctx = stage.execute(ctx).await
                .with_context(|| format!("Stage '{}' failed", stage.name()))?;

            // Validate if strategy requires it
            if self.validation != ValidationStrategy::None {
                debug!("Validating stage: {}", stage.name());
                let validation_result = stage.validate(&ctx)?;
                ctx.validation_results.push(validation_result.clone());

                if !validation_result.passed && self.validation == ValidationStrategy::StopOnError {
                    anyhow::bail!(
                        "Validation failed for stage '{}': {}",
                        stage.name(),
                        validation_result.message
                    );
                }
            }
        }

        info!("Pipeline completed successfully");
        Ok(ctx.output())
    }
}

/// Analysis stage - detects languages and dependencies
pub struct AnalysisStage;

#[async_trait::async_trait]
impl PipelineStage for AnalysisStage {
    fn name(&self) -> &str {
        "Analysis"
    }

    async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        info!("Analyzing project at {:?}", ctx.input_path);

        let analysis = crate::analyzer::analyze_project(
            &ctx.input_path,
            false,  // TDG - skip for pipeline
            true,   // languages
            true,   // dependencies
        )?;

        ctx.primary_language = analysis.primary_language;
        ctx.metadata.insert(
            "total_files".to_string(),
            serde_json::json!(analysis.total_files),
        );
        ctx.metadata.insert(
            "total_lines".to_string(),
            serde_json::json!(analysis.total_lines),
        );

        Ok(ctx)
    }

    fn validate(&self, ctx: &PipelineContext) -> Result<ValidationResult> {
        let passed = ctx.primary_language.is_some();
        Ok(ValidationResult {
            stage: self.name().to_string(),
            passed,
            message: if passed {
                format!("Language detected: {:?}", ctx.primary_language)
            } else {
                "Could not detect primary language".to_string()
            },
            details: None,
        })
    }
}

/// Transpilation stage - converts source to Rust
pub struct TranspilationStage {
    incremental: bool,
    cache: bool,
    numpy_converter: Option<NumPyConverter>,
    sklearn_converter: Option<SklearnConverter>,
    pytorch_converter: Option<PyTorchConverter>,
}

impl TranspilationStage {
    pub fn new(incremental: bool, cache: bool) -> Self {
        Self {
            incremental,
            cache,
            numpy_converter: Some(NumPyConverter::new()),
            sklearn_converter: Some(SklearnConverter::new()),
            pytorch_converter: Some(PyTorchConverter::new()),
        }
    }

    /// Analyze Python source for NumPy usage and provide conversion guidance
    #[cfg(feature = "native")]
    fn analyze_numpy_usage(&self, input_path: &Path) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        if let Some(converter) = &self.numpy_converter {
            // Walk Python files looking for NumPy imports
            for entry in walkdir::WalkDir::new(input_path)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if let Some(ext) = entry.path().extension() {
                    if ext == "py" {
                        // Read file and check for numpy imports
                        if let Ok(content) = std::fs::read_to_string(entry.path()) {
                            if content.contains("import numpy") || content.contains("from numpy") {
                                info!("  Found NumPy usage in: {}", entry.path().display());

                                // Analyze common NumPy operations
                                let operations = vec![
                                    ("np.add", NumPyOp::Add),
                                    ("np.subtract", NumPyOp::Subtract),
                                    ("np.multiply", NumPyOp::Multiply),
                                    ("np.dot", NumPyOp::Dot),
                                    ("np.sum", NumPyOp::Sum),
                                    ("np.array", NumPyOp::Array),
                                ];

                                for (pattern, op) in operations {
                                    if content.contains(pattern) {
                                        if let Some(trueno_op) = converter.convert(&op) {
                                            recommendations.push(format!(
                                                "{}: {} → {}",
                                                entry.path().display(),
                                                pattern,
                                                trueno_op.code_template
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(recommendations)
    }

    /// Analyze Python source for sklearn usage and provide conversion guidance
    #[cfg(feature = "native")]
    fn analyze_sklearn_usage(&self, input_path: &Path) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        if let Some(converter) = &self.sklearn_converter {
            // Walk Python files looking for sklearn imports
            for entry in WalkDir::new(input_path)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if let Some(ext) = entry.path().extension() {
                    if ext == "py" {
                        // Read file and check for sklearn imports
                        if let Ok(content) = std::fs::read_to_string(entry.path()) {
                            if content.contains("import sklearn") || content.contains("from sklearn") {
                                info!("  Found sklearn usage in: {}", entry.path().display());

                                // Analyze common sklearn algorithms
                                let algorithms = vec![
                                    ("LinearRegression", SklearnAlgorithm::LinearRegression),
                                    ("LogisticRegression", SklearnAlgorithm::LogisticRegression),
                                    ("KMeans", SklearnAlgorithm::KMeans),
                                    ("DecisionTreeClassifier", SklearnAlgorithm::DecisionTreeClassifier),
                                    ("RandomForestClassifier", SklearnAlgorithm::RandomForestClassifier),
                                    ("StandardScaler", SklearnAlgorithm::StandardScaler),
                                    ("train_test_split", SklearnAlgorithm::TrainTestSplit),
                                ];

                                for (pattern, alg) in algorithms {
                                    if content.contains(pattern) {
                                        if let Some(aprender_alg) = converter.convert(&alg) {
                                            recommendations.push(format!(
                                                "{}: {} ({}) → {}",
                                                entry.path().display(),
                                                pattern,
                                                alg.sklearn_module(),
                                                aprender_alg.code_template
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(recommendations)
    }

    /// Analyze Python source for PyTorch usage and provide conversion guidance
    #[cfg(feature = "native")]
    fn analyze_pytorch_usage(&self, input_path: &Path) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        if let Some(converter) = &self.pytorch_converter {
            // Walk Python files looking for PyTorch/transformers imports
            for entry in WalkDir::new(input_path)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if let Some(ext) = entry.path().extension() {
                    if ext == "py" {
                        // Read file and check for PyTorch imports
                        if let Ok(content) = std::fs::read_to_string(entry.path()) {
                            if content.contains("import torch") || content.contains("from torch")
                                || content.contains("from transformers") {
                                info!("  Found PyTorch usage in: {}", entry.path().display());

                                // Analyze common PyTorch operations
                                let operations = vec![
                                    ("torch.load", PyTorchOperation::LoadModel),
                                    ("from_pretrained", PyTorchOperation::LoadModel),
                                    ("AutoTokenizer", PyTorchOperation::LoadTokenizer),
                                    (".forward(", PyTorchOperation::Forward),
                                    (".generate(", PyTorchOperation::Generate),
                                    ("nn.Linear", PyTorchOperation::Linear),
                                    ("MultiheadAttention", PyTorchOperation::Attention),
                                    ("tokenizer.encode", PyTorchOperation::Encode),
                                    ("tokenizer.decode", PyTorchOperation::Decode),
                                ];

                                for (pattern, op) in operations {
                                    if content.contains(pattern) {
                                        if let Some(realizar_op) = converter.convert(&op) {
                                            recommendations.push(format!(
                                                "{}: {} ({}) → {}",
                                                entry.path().display(),
                                                pattern,
                                                op.pytorch_module(),
                                                realizar_op.code_template
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(recommendations)
    }
}

#[async_trait::async_trait]
impl PipelineStage for TranspilationStage {
    fn name(&self) -> &str {
        "Transpilation"
    }

    async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        info!("Transpiling {} to Rust", ctx.primary_language.as_ref()
            .map(|l| format!("{}", l))
            .unwrap_or_else(|| "unknown".to_string())
        );

        // Create output directory structure
        std::fs::create_dir_all(&ctx.output_path)?;
        std::fs::create_dir_all(ctx.output_path.join("src"))?;

        // If Python project, analyze NumPy usage
        #[cfg(feature = "native")]
        if let Some(crate::types::Language::Python) = ctx.primary_language {
            info!("Analyzing NumPy usage for conversion guidance");
            match self.analyze_numpy_usage(&ctx.input_path) {
                Ok(recommendations) => {
                    if !recommendations.is_empty() {
                        info!("Found {} NumPy operations to convert:", recommendations.len());
                        for rec in &recommendations {
                            info!("  - {}", rec);
                        }

                        ctx.metadata.insert(
                            "numpy_conversions".to_string(),
                            serde_json::json!(recommendations),
                        );

                        ctx.metadata.insert(
                            "numpy_detected".to_string(),
                            serde_json::json!(true),
                        );
                    } else {
                        info!("No NumPy usage detected");
                        ctx.metadata.insert(
                            "numpy_detected".to_string(),
                            serde_json::json!(false),
                        );
                    }
                }
                Err(e) => {
                    warn!("NumPy analysis failed: {}", e);
                }
            }

            // Also analyze sklearn usage
            info!("Analyzing sklearn usage for conversion guidance");
            match self.analyze_sklearn_usage(&ctx.input_path) {
                Ok(recommendations) => {
                    if !recommendations.is_empty() {
                        info!("Found {} sklearn algorithms to convert:", recommendations.len());
                        for rec in &recommendations {
                            info!("  - {}", rec);
                        }

                        ctx.metadata.insert(
                            "sklearn_conversions".to_string(),
                            serde_json::json!(recommendations),
                        );

                        ctx.metadata.insert(
                            "sklearn_detected".to_string(),
                            serde_json::json!(true),
                        );
                    } else {
                        info!("No sklearn usage detected");
                        ctx.metadata.insert(
                            "sklearn_detected".to_string(),
                            serde_json::json!(false),
                        );
                    }
                }
                Err(e) => {
                    warn!("sklearn analysis failed: {}", e);
                }
            }

            // Also analyze PyTorch usage
            info!("Analyzing PyTorch usage for conversion guidance");
            match self.analyze_pytorch_usage(&ctx.input_path) {
                Ok(recommendations) => {
                    if !recommendations.is_empty() {
                        info!("Found {} PyTorch operations to convert:", recommendations.len());
                        for rec in &recommendations {
                            info!("  - {}", rec);
                        }

                        ctx.metadata.insert(
                            "pytorch_conversions".to_string(),
                            serde_json::json!(recommendations),
                        );

                        ctx.metadata.insert(
                            "pytorch_detected".to_string(),
                            serde_json::json!(true),
                        );
                    } else {
                        info!("No PyTorch usage detected");
                        ctx.metadata.insert(
                            "pytorch_detected".to_string(),
                            serde_json::json!(false),
                        );
                    }
                }
                Err(e) => {
                    warn!("PyTorch analysis failed: {}", e);
                }
            }
        }

        // Detect available tools
        let tools = crate::tools::ToolRegistry::detect();

        // Get transpiler for primary language and run transpilation
        if let Some(lang) = &ctx.primary_language {
            use crate::types::Language;

            info!("Starting transpilation for language: {}", lang);

            let result = match lang {
                Language::Python => {
                    if tools.depyler.is_some() {
                        info!("Using Depyler for Python transpilation");
                        crate::tools::transpile_python(&ctx.input_path, &ctx.output_path)
                    } else {
                        anyhow::bail!("Depyler not available. Install with: cargo install depyler");
                    }
                }
                Language::Shell => {
                    if tools.bashrs.is_some() {
                        info!("Using Bashrs for Shell transpilation");
                        crate::tools::transpile_shell(&ctx.input_path, &ctx.output_path)
                    } else {
                        anyhow::bail!("Bashrs not available. Install with: cargo install bashrs");
                    }
                }
                Language::C | Language::Cpp => {
                    if tools.decy.is_some() {
                        info!("Using Decy for C/C++ transpilation");
                        crate::tools::transpile_c_cpp(&ctx.input_path, &ctx.output_path)
                    } else {
                        anyhow::bail!("Decy not available. Install with: cargo install decy");
                    }
                }
                _ => {
                    anyhow::bail!("No transpiler available for language: {}", lang);
                }
            };

            match result {
                Ok(output) => {
                    info!("Transpilation completed successfully");
                    info!("Output: {}", output);

                    ctx.metadata.insert(
                        "transpiler".to_string(),
                        serde_json::json!(format!("{}", lang)),
                    );
                    ctx.metadata.insert(
                        "transpilation_output".to_string(),
                        serde_json::json!(output),
                    );
                }
                Err(e) => {
                    warn!("Transpilation failed: {}", e);
                    anyhow::bail!("Transpilation failed: {}", e);
                }
            }
        } else {
            anyhow::bail!("No primary language detected");
        }

        Ok(ctx)
    }

    fn validate(&self, ctx: &PipelineContext) -> Result<ValidationResult> {
        // Check that output directory exists and has content
        let src_dir = ctx.output_path.join("src");
        let passed = src_dir.exists() && src_dir.read_dir()?.next().is_some();

        Ok(ValidationResult {
            stage: self.name().to_string(),
            passed,
            message: if passed {
                "Transpilation output validated".to_string()
            } else {
                "No transpiled files found".to_string()
            },
            details: None,
        })
    }
}

/// Optimization stage - applies performance optimizations using MoE routing
pub struct OptimizationStage {
    enable_gpu: bool,
    enable_simd: bool,
    gpu_threshold: usize,
    backend_selector: crate::backend::BackendSelector,
}

impl OptimizationStage {
    pub fn new(enable_gpu: bool, enable_simd: bool, gpu_threshold: usize) -> Self {
        Self {
            enable_gpu,
            enable_simd,
            gpu_threshold,
            backend_selector: crate::backend::BackendSelector::new(),
        }
    }

    /// Analyze code and recommend backend optimizations using MoE
    fn analyze_optimizations(&self) -> Vec<String> {
        use crate::backend::OpComplexity;

        let mut recommendations = Vec::new();

        // Example workload analysis - in practice this would analyze the actual code
        let workloads = vec![
            ("Element-wise operations", OpComplexity::Low, 1_000_000),
            ("Vector reductions", OpComplexity::Medium, 50_000),
            ("Matrix multiplications", OpComplexity::High, 100_000),
        ];

        for (name, complexity, size) in workloads {
            let backend = self.backend_selector.select_with_moe(complexity, size);
            recommendations.push(format!("{}: {} backend recommended ({} elements)",
                name, backend, size));
        }

        recommendations
    }
}

#[async_trait::async_trait]
impl PipelineStage for OptimizationStage {
    fn name(&self) -> &str {
        "Optimization"
    }

    async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        info!("Applying optimizations using MoE routing (GPU: {}, SIMD: {})",
            self.enable_gpu, self.enable_simd);

        // Use MoE to analyze and recommend backend optimizations
        let moe_recommendations = self.analyze_optimizations();

        info!("MoE backend recommendations:");
        for rec in &moe_recommendations {
            info!("  - {}", rec);
        }

        // Apply traditional optimizations
        if self.enable_simd {
            ctx.optimizations.push("SIMD vectorization enabled".to_string());
        }

        if self.enable_gpu {
            ctx.optimizations.push(format!("GPU dispatch enabled (threshold: {})", self.gpu_threshold));
        }

        // Add MoE recommendations
        ctx.optimizations.extend(moe_recommendations);

        // Store optimization strategy in metadata
        ctx.metadata.insert(
            "optimizations_applied".to_string(),
            serde_json::json!(ctx.optimizations),
        );

        ctx.metadata.insert(
            "moe_routing_enabled".to_string(),
            serde_json::json!(true),
        );

        Ok(ctx)
    }
}

/// Validation stage - verifies semantic equivalence
pub struct ValidationStage {
    trace_syscalls: bool,
    run_tests: bool,
}

impl ValidationStage {
    pub fn new(trace_syscalls: bool, run_tests: bool) -> Self {
        Self {
            trace_syscalls,
            run_tests,
        }
    }

    /// Trace syscalls from both binaries and compare them for semantic equivalence
    async fn trace_and_compare(
        &self,
        original_binary: &std::path::Path,
        transpiled_binary: &std::path::Path,
    ) -> Result<bool> {
        // Trace original binary
        let original_trace = Self::trace_binary(original_binary)?;

        // Trace transpiled binary
        let transpiled_trace = Self::trace_binary(transpiled_binary)?;

        // Compare traces
        Ok(Self::compare_traces(&original_trace, &transpiled_trace))
    }

    /// Trace a binary using Renacer and capture syscall output
    fn trace_binary(binary: &std::path::Path) -> Result<Vec<String>> {
        use std::process::Command;

        // Run the binary with renacer tracing
        // Note: This is a simplified approach - ideally we'd use the renacer library directly
        let output = Command::new("renacer")
            .arg(binary.to_string_lossy().to_string())
            .output()
            .context("Failed to run renacer")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Renacer failed: {}", stderr);
        }

        // Parse syscalls from output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let syscalls: Vec<String> = stdout
            .lines()
            .filter(|line| !line.starts_with('[')) // Filter out renacer's own messages
            .map(|s| s.to_string())
            .collect();

        Ok(syscalls)
    }

    /// Compare two syscall traces for semantic equivalence
    fn compare_traces(trace1: &[String], trace2: &[String]) -> bool {
        // For now, simple comparison - ideally would normalize and filter
        // non-deterministic syscalls (timestamps, PIDs, etc.)
        if trace1.len() != trace2.len() {
            return false;
        }

        // Compare each syscall (ignoring arguments that may vary)
        for (call1, call2) in trace1.iter().zip(trace2.iter()) {
            // Extract syscall name (before the '(' character)
            let name1 = call1.split('(').next().unwrap_or("");
            let name2 = call2.split('(').next().unwrap_or("");

            if name1 != name2 {
                return false;
            }
        }

        true
    }
}

#[async_trait::async_trait]
impl PipelineStage for ValidationStage {
    fn name(&self) -> &str {
        "Validation"
    }

    async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        info!("Validating semantic equivalence");

        // If syscall tracing is enabled, use Renacer to verify equivalence
        if self.trace_syscalls {
            info!("Tracing syscalls with Renacer");

            // Find original and transpiled binaries
            let original_binary = ctx.input_path.join("original_binary");
            let transpiled_binary = ctx.output_path.join("target/release/transpiled");

            if original_binary.exists() && transpiled_binary.exists() {
                match self.trace_and_compare(&original_binary, &transpiled_binary).await {
                    Ok(equivalent) => {
                        ctx.validation_results.push(ValidationResult {
                            stage: self.name().to_string(),
                            passed: equivalent,
                            message: if equivalent {
                                "Syscall traces match - semantic equivalence verified".to_string()
                            } else {
                                "Syscall traces differ - semantic equivalence NOT verified".to_string()
                            },
                            details: None,
                        });

                        ctx.metadata.insert(
                            "syscall_equivalence".to_string(),
                            serde_json::json!(equivalent),
                        );
                    }
                    Err(e) => {
                        warn!("Syscall tracing failed: {}", e);
                        ctx.validation_results.push(ValidationResult {
                            stage: self.name().to_string(),
                            passed: false,
                            message: format!("Syscall tracing error: {}", e),
                            details: None,
                        });
                    }
                }
            } else {
                info!("Binaries not found for comparison, skipping syscall trace");
            }
        }

        // If run_tests is enabled, run the original test suite
        if self.run_tests {
            info!("Running original test suite");
            // TODO: Implement test suite execution
        }

        ctx.metadata.insert(
            "validation_completed".to_string(),
            serde_json::json!(true),
        );

        Ok(ctx)
    }
}

/// Build stage - compiles to final binary
pub struct BuildStage {
    release: bool,
    target: Option<String>,
    wasm: bool,
}

impl BuildStage {
    pub fn new(release: bool, target: Option<String>, wasm: bool) -> Self {
        Self {
            release,
            target,
            wasm,
        }
    }
}

#[async_trait::async_trait]
impl PipelineStage for BuildStage {
    fn name(&self) -> &str {
        "Build"
    }

    async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        info!("Building Rust project (release: {})", self.release);

        // Check if Cargo.toml exists in output directory
        let cargo_toml = ctx.output_path.join("Cargo.toml");
        if !cargo_toml.exists() {
            anyhow::bail!("No Cargo.toml found in output directory");
        }

        // Build cargo arguments
        let mut args = vec!["build"];

        if self.release {
            args.push("--release");
        }

        if let Some(target) = &self.target {
            args.push("--target");
            args.push(target);
        }

        // For WASM, use special target
        if self.wasm {
            args.push("--target");
            args.push("wasm32-unknown-unknown");
        }

        // Run cargo build
        let output = std::process::Command::new("cargo")
            .args(&args)
            .current_dir(&ctx.output_path)
            .output()
            .context("Failed to run cargo build")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Cargo build failed: {}", stderr);
        }

        ctx.metadata.insert(
            "build_mode".to_string(),
            serde_json::json!(if self.release { "release" } else { "debug" }),
        );

        if self.wasm {
            ctx.metadata.insert(
                "wasm_build".to_string(),
                serde_json::json!(true),
            );
        }

        Ok(ctx)
    }

    fn validate(&self, ctx: &PipelineContext) -> Result<ValidationResult> {
        // Check that build artifacts exist
        let build_dir = if self.release {
            ctx.output_path.join("target/release")
        } else {
            ctx.output_path.join("target/debug")
        };

        let passed = build_dir.exists();

        Ok(ValidationResult {
            stage: self.name().to_string(),
            passed,
            message: if passed {
                "Build artifacts found".to_string()
            } else {
                "Build directory not found".to_string()
            },
            details: None,
        })
    }
}
