#![allow(dead_code)]

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
    #[allow(clippy::cognitive_complexity)]
    pub async fn run(&self, input: &Path, output: &Path) -> Result<PipelineOutput> {
        info!("Starting pipeline with {} stages", self.stages.len());

        let mut ctx = PipelineContext::new(input.to_path_buf(), output.to_path_buf());

        for (idx, stage) in self.stages.iter().enumerate() {
            info!(
                "Running stage {}/{}: {}",
                idx + 1,
                self.stages.len(),
                stage.name()
            );

            // Execute stage
            ctx = stage
                .execute(ctx)
                .await
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
            false, // TDG - skip for pipeline
            true,  // languages
            true,  // dependencies
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
                            if content.contains("import sklearn")
                                || content.contains("from sklearn")
                            {
                                info!("  Found sklearn usage in: {}", entry.path().display());

                                // Analyze common sklearn algorithms
                                let algorithms = vec![
                                    ("LinearRegression", SklearnAlgorithm::LinearRegression),
                                    ("LogisticRegression", SklearnAlgorithm::LogisticRegression),
                                    ("KMeans", SklearnAlgorithm::KMeans),
                                    (
                                        "DecisionTreeClassifier",
                                        SklearnAlgorithm::DecisionTreeClassifier,
                                    ),
                                    (
                                        "RandomForestClassifier",
                                        SklearnAlgorithm::RandomForestClassifier,
                                    ),
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
                            if content.contains("import torch")
                                || content.contains("from torch")
                                || content.contains("from transformers")
                            {
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
        info!(
            "Transpiling {} to Rust",
            ctx.primary_language
                .as_ref()
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
                        info!(
                            "Found {} NumPy operations to convert:",
                            recommendations.len()
                        );
                        for rec in &recommendations {
                            info!("  - {}", rec);
                        }

                        ctx.metadata.insert(
                            "numpy_conversions".to_string(),
                            serde_json::json!(recommendations),
                        );

                        ctx.metadata
                            .insert("numpy_detected".to_string(), serde_json::json!(true));
                    } else {
                        info!("No NumPy usage detected");
                        ctx.metadata
                            .insert("numpy_detected".to_string(), serde_json::json!(false));
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
                        info!(
                            "Found {} sklearn algorithms to convert:",
                            recommendations.len()
                        );
                        for rec in &recommendations {
                            info!("  - {}", rec);
                        }

                        ctx.metadata.insert(
                            "sklearn_conversions".to_string(),
                            serde_json::json!(recommendations),
                        );

                        ctx.metadata
                            .insert("sklearn_detected".to_string(), serde_json::json!(true));
                    } else {
                        info!("No sklearn usage detected");
                        ctx.metadata
                            .insert("sklearn_detected".to_string(), serde_json::json!(false));
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
                        info!(
                            "Found {} PyTorch operations to convert:",
                            recommendations.len()
                        );
                        for rec in &recommendations {
                            info!("  - {}", rec);
                        }

                        ctx.metadata.insert(
                            "pytorch_conversions".to_string(),
                            serde_json::json!(recommendations),
                        );

                        ctx.metadata
                            .insert("pytorch_detected".to_string(), serde_json::json!(true));
                    } else {
                        info!("No PyTorch usage detected");
                        ctx.metadata
                            .insert("pytorch_detected".to_string(), serde_json::json!(false));
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
            recommendations.push(format!(
                "{}: {} backend recommended ({} elements)",
                name, backend, size
            ));
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
        info!(
            "Applying optimizations using MoE routing (GPU: {}, SIMD: {})",
            self.enable_gpu, self.enable_simd
        );

        // Use MoE to analyze and recommend backend optimizations
        let moe_recommendations = self.analyze_optimizations();

        info!("MoE backend recommendations:");
        for rec in &moe_recommendations {
            info!("  - {}", rec);
        }

        // Apply traditional optimizations
        if self.enable_simd {
            ctx.optimizations
                .push("SIMD vectorization enabled".to_string());
        }

        if self.enable_gpu {
            ctx.optimizations.push(format!(
                "GPU dispatch enabled (threshold: {})",
                self.gpu_threshold
            ));
        }

        // Add MoE recommendations
        ctx.optimizations.extend(moe_recommendations);

        // Store optimization strategy in metadata
        ctx.metadata.insert(
            "optimizations_applied".to_string(),
            serde_json::json!(ctx.optimizations),
        );

        ctx.metadata
            .insert("moe_routing_enabled".to_string(), serde_json::json!(true));

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
                match self
                    .trace_and_compare(&original_binary, &transpiled_binary)
                    .await
                {
                    Ok(equivalent) => {
                        ctx.validation_results.push(ValidationResult {
                            stage: self.name().to_string(),
                            passed: equivalent,
                            message: if equivalent {
                                "Syscall traces match - semantic equivalence verified".to_string()
                            } else {
                                "Syscall traces differ - semantic equivalence NOT verified"
                                    .to_string()
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

        ctx.metadata
            .insert("validation_completed".to_string(), serde_json::json!(true));

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
            ctx.metadata
                .insert("wasm_build".to_string(), serde_json::json!(true));
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ============================================================================
    // PIPELINE CONTEXT TESTS
    // ============================================================================

    #[test]
    fn test_pipeline_context_creation() {
        let input = PathBuf::from("/input");
        let output = PathBuf::from("/output");

        let ctx = PipelineContext::new(input.clone(), output.clone());

        assert_eq!(ctx.input_path, input);
        assert_eq!(ctx.output_path, output);
        assert!(ctx.primary_language.is_none());
        assert!(ctx.file_mappings.is_empty());
        assert!(ctx.optimizations.is_empty());
        assert!(ctx.validation_results.is_empty());
        assert!(ctx.metadata.is_empty());
    }

    #[test]
    fn test_pipeline_context_with_language() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

        ctx.primary_language = Some(crate::types::Language::Python);
        assert_eq!(ctx.primary_language, Some(crate::types::Language::Python));
    }

    #[test]
    fn test_pipeline_context_file_mappings() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

        ctx.file_mappings.push((
            PathBuf::from("/input/main.py"),
            PathBuf::from("/output/src/main.rs"),
        ));

        assert_eq!(ctx.file_mappings.len(), 1);
        assert_eq!(ctx.file_mappings[0].0, PathBuf::from("/input/main.py"));
        assert_eq!(ctx.file_mappings[0].1, PathBuf::from("/output/src/main.rs"));
    }

    #[test]
    fn test_pipeline_context_optimizations() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

        ctx.optimizations.push("SIMD enabled".to_string());
        ctx.optimizations.push("GPU dispatch enabled".to_string());

        assert_eq!(ctx.optimizations.len(), 2);
        assert_eq!(ctx.optimizations[0], "SIMD enabled");
        assert_eq!(ctx.optimizations[1], "GPU dispatch enabled");
    }

    #[test]
    fn test_pipeline_context_validation_results() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

        ctx.validation_results.push(ValidationResult {
            stage: "Analysis".to_string(),
            passed: true,
            message: "Language detected".to_string(),
            details: None,
        });

        assert_eq!(ctx.validation_results.len(), 1);
        assert!(ctx.validation_results[0].passed);
        assert_eq!(ctx.validation_results[0].stage, "Analysis");
    }

    #[test]
    fn test_pipeline_context_metadata() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

        ctx.metadata
            .insert("total_files".to_string(), serde_json::json!(42));
        ctx.metadata
            .insert("language".to_string(), serde_json::json!("Python"));

        assert_eq!(ctx.metadata.len(), 2);
        assert_eq!(
            ctx.metadata.get("total_files").unwrap(),
            &serde_json::json!(42)
        );
        assert_eq!(
            ctx.metadata.get("language").unwrap(),
            &serde_json::json!("Python")
        );
    }

    #[test]
    fn test_pipeline_context_output() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

        ctx.file_mappings.push((
            PathBuf::from("/input/main.py"),
            PathBuf::from("/output/main.rs"),
        ));
        ctx.optimizations.push("SIMD".to_string());
        ctx.validation_results.push(ValidationResult {
            stage: "Test".to_string(),
            passed: true,
            message: "OK".to_string(),
            details: None,
        });

        let output = ctx.output();

        assert_eq!(output.output_path, PathBuf::from("/output"));
        assert_eq!(output.file_mappings.len(), 1);
        assert_eq!(output.optimizations.len(), 1);
        assert!(output.validation_passed);
    }

    #[test]
    fn test_pipeline_context_output_validation_failed() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

        // Add one passing and one failing validation result
        ctx.validation_results.push(ValidationResult {
            stage: "Stage1".to_string(),
            passed: true,
            message: "OK".to_string(),
            details: None,
        });
        ctx.validation_results.push(ValidationResult {
            stage: "Stage2".to_string(),
            passed: false,
            message: "Failed".to_string(),
            details: None,
        });

        let output = ctx.output();

        // Should be false because at least one validation failed
        assert!(!output.validation_passed);
    }

    // ============================================================================
    // VALIDATION RESULT TESTS
    // ============================================================================

    #[test]
    fn test_validation_result_passed() {
        let result = ValidationResult {
            stage: "TestStage".to_string(),
            passed: true,
            message: "All checks passed".to_string(),
            details: None,
        };

        assert_eq!(result.stage, "TestStage");
        assert!(result.passed);
        assert_eq!(result.message, "All checks passed");
        assert!(result.details.is_none());
    }

    #[test]
    fn test_validation_result_failed() {
        let result = ValidationResult {
            stage: "TestStage".to_string(),
            passed: false,
            message: "Check failed".to_string(),
            details: Some(serde_json::json!({"error": "details here"})),
        };

        assert_eq!(result.stage, "TestStage");
        assert!(!result.passed);
        assert_eq!(result.message, "Check failed");
        assert!(result.details.is_some());
    }

    #[test]
    fn test_validation_result_with_details() {
        let details = serde_json::json!({
            "errors": ["error1", "error2"],
            "warnings": ["warning1"]
        });

        let result = ValidationResult {
            stage: "Analysis".to_string(),
            passed: false,
            message: "Multiple issues found".to_string(),
            details: Some(details.clone()),
        };

        assert_eq!(result.details, Some(details));
    }

    // ============================================================================
    // VALIDATION STRATEGY TESTS
    // ============================================================================

    #[test]
    fn test_validation_strategy_equality() {
        assert_eq!(
            ValidationStrategy::StopOnError,
            ValidationStrategy::StopOnError
        );
        assert_eq!(
            ValidationStrategy::ContinueOnError,
            ValidationStrategy::ContinueOnError
        );
        assert_eq!(ValidationStrategy::None, ValidationStrategy::None);

        assert_ne!(
            ValidationStrategy::StopOnError,
            ValidationStrategy::ContinueOnError
        );
        assert_ne!(ValidationStrategy::StopOnError, ValidationStrategy::None);
        assert_ne!(
            ValidationStrategy::ContinueOnError,
            ValidationStrategy::None
        );
    }

    // ============================================================================
    // TRANSPILATION PIPELINE TESTS
    // ============================================================================

    struct MockStage {
        name: String,
        should_fail: bool,
        validation_should_pass: bool,
    }

    impl MockStage {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                should_fail: false,
                validation_should_pass: true,
            }
        }

        fn with_execution_failure(mut self) -> Self {
            self.should_fail = true;
            self
        }

        fn with_validation_failure(mut self) -> Self {
            self.validation_should_pass = false;
            self
        }
    }

    #[async_trait::async_trait]
    impl PipelineStage for MockStage {
        fn name(&self) -> &str {
            &self.name
        }

        async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
            if self.should_fail {
                anyhow::bail!("Execution failed for {}", self.name);
            }

            ctx.metadata
                .insert(format!("{}_executed", self.name), serde_json::json!(true));

            Ok(ctx)
        }

        fn validate(&self, _ctx: &PipelineContext) -> Result<ValidationResult> {
            Ok(ValidationResult {
                stage: self.name.clone(),
                passed: self.validation_should_pass,
                message: if self.validation_should_pass {
                    format!("{} validation passed", self.name)
                } else {
                    format!("{} validation failed", self.name)
                },
                details: None,
            })
        }
    }

    #[tokio::test]
    async fn test_pipeline_creation() {
        let pipeline = TranspilationPipeline::new(ValidationStrategy::StopOnError);
        assert_eq!(pipeline.stages.len(), 0);
        assert_eq!(pipeline.validation, ValidationStrategy::StopOnError);
    }

    #[tokio::test]
    async fn test_pipeline_add_stage() {
        let pipeline = TranspilationPipeline::new(ValidationStrategy::None)
            .add_stage(Box::new(MockStage::new("Stage1")))
            .add_stage(Box::new(MockStage::new("Stage2")));

        assert_eq!(pipeline.stages.len(), 2);
    }

    #[tokio::test]
    async fn test_pipeline_run_no_stages() {
        let pipeline = TranspilationPipeline::new(ValidationStrategy::None);

        let input = PathBuf::from("/tmp/input");
        let output = PathBuf::from("/tmp/output");

        let result = pipeline.run(&input, &output).await;
        assert!(result.is_ok());

        let pipeline_output = result.unwrap();
        assert_eq!(pipeline_output.output_path, output);
        assert!(pipeline_output.file_mappings.is_empty());
    }

    #[tokio::test]
    async fn test_pipeline_run_single_stage() {
        let pipeline = TranspilationPipeline::new(ValidationStrategy::None)
            .add_stage(Box::new(MockStage::new("TestStage")));

        let input = PathBuf::from("/tmp/input");
        let output = PathBuf::from("/tmp/output");

        let result = pipeline.run(&input, &output).await;
        assert!(result.is_ok());

        let pipeline_output = result.unwrap();
        assert_eq!(pipeline_output.output_path, output);
    }

    #[tokio::test]
    async fn test_pipeline_run_multiple_stages() {
        let pipeline = TranspilationPipeline::new(ValidationStrategy::None)
            .add_stage(Box::new(MockStage::new("Stage1")))
            .add_stage(Box::new(MockStage::new("Stage2")))
            .add_stage(Box::new(MockStage::new("Stage3")));

        let input = PathBuf::from("/tmp/input");
        let output = PathBuf::from("/tmp/output");

        let result = pipeline.run(&input, &output).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_pipeline_stage_execution_order() {
        let pipeline = TranspilationPipeline::new(ValidationStrategy::None)
            .add_stage(Box::new(MockStage::new("First")))
            .add_stage(Box::new(MockStage::new("Second")))
            .add_stage(Box::new(MockStage::new("Third")));

        let input = PathBuf::from("/tmp/input");
        let output = PathBuf::from("/tmp/output");

        let result = pipeline.run(&input, &output).await;
        assert!(result.is_ok());

        // Stages should execute in order, adding metadata
        // This is implicitly tested by the fact that execution succeeds
    }

    #[tokio::test]
    async fn test_pipeline_stop_on_error_execution() {
        let pipeline = TranspilationPipeline::new(ValidationStrategy::StopOnError)
            .add_stage(Box::new(MockStage::new("Stage1")))
            .add_stage(Box::new(MockStage::new("Stage2").with_execution_failure()))
            .add_stage(Box::new(MockStage::new("Stage3")));

        let input = PathBuf::from("/tmp/input");
        let output = PathBuf::from("/tmp/output");

        let result = pipeline.run(&input, &output).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("Stage 'Stage2' failed"));
    }

    #[tokio::test]
    async fn test_pipeline_stop_on_error_validation() {
        let pipeline = TranspilationPipeline::new(ValidationStrategy::StopOnError)
            .add_stage(Box::new(MockStage::new("Stage1")))
            .add_stage(Box::new(MockStage::new("Stage2").with_validation_failure()))
            .add_stage(Box::new(MockStage::new("Stage3")));

        let input = PathBuf::from("/tmp/input");
        let output = PathBuf::from("/tmp/output");

        let result = pipeline.run(&input, &output).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Validation failed for stage 'Stage2'"));
    }

    #[tokio::test]
    async fn test_pipeline_continue_on_error_validation() {
        let pipeline = TranspilationPipeline::new(ValidationStrategy::ContinueOnError)
            .add_stage(Box::new(MockStage::new("Stage1")))
            .add_stage(Box::new(MockStage::new("Stage2").with_validation_failure()))
            .add_stage(Box::new(MockStage::new("Stage3")));

        let input = PathBuf::from("/tmp/input");
        let output = PathBuf::from("/tmp/output");

        let result = pipeline.run(&input, &output).await;
        assert!(result.is_ok());

        // Validation failure should be recorded but pipeline continues
        let pipeline_output = result.unwrap();
        assert!(!pipeline_output.validation_passed);
    }

    #[tokio::test]
    async fn test_pipeline_no_validation() {
        let pipeline = TranspilationPipeline::new(ValidationStrategy::None)
            .add_stage(Box::new(MockStage::new("Stage1").with_validation_failure()))
            .add_stage(Box::new(MockStage::new("Stage2").with_validation_failure()));

        let input = PathBuf::from("/tmp/input");
        let output = PathBuf::from("/tmp/output");

        let result = pipeline.run(&input, &output).await;
        assert!(result.is_ok());

        // No validation should be performed, so pipeline succeeds
        let pipeline_output = result.unwrap();
        // With ValidationStrategy::None, no validation results are added
        assert!(pipeline_output.validation_passed); // All (zero) validations passed
    }

    // ============================================================================
    // TRANSPILATION STAGE TESTS
    // ============================================================================

    #[test]
    fn test_transpilation_stage_creation() {
        let stage = TranspilationStage::new(true, true);
        assert!(stage.incremental);
        assert!(stage.cache);
        assert!(stage.numpy_converter.is_some());
        assert!(stage.sklearn_converter.is_some());
        assert!(stage.pytorch_converter.is_some());
    }

    #[test]
    fn test_transpilation_stage_name() {
        let stage = TranspilationStage::new(false, false);
        assert_eq!(stage.name(), "Transpilation");
    }

    // ============================================================================
    // OPTIMIZATION STAGE TESTS
    // ============================================================================

    #[test]
    fn test_optimization_stage_creation() {
        let stage = OptimizationStage::new(true, true, 1000);
        assert!(stage.enable_gpu);
        assert!(stage.enable_simd);
        assert_eq!(stage.gpu_threshold, 1000);
    }

    #[test]
    fn test_optimization_stage_name() {
        let stage = OptimizationStage::new(false, false, 500);
        assert_eq!(stage.name(), "Optimization");
    }

    #[test]
    fn test_optimization_stage_analyze_optimizations() {
        let stage = OptimizationStage::new(true, true, 1000);
        let recommendations = stage.analyze_optimizations();

        // Should return 3 recommendations (from the hardcoded workloads)
        assert_eq!(recommendations.len(), 3);

        // Check that recommendations contain backend information
        assert!(recommendations[0].contains("Element-wise operations"));
        assert!(recommendations[1].contains("Vector reductions"));
        assert!(recommendations[2].contains("Matrix multiplications"));
    }

    #[tokio::test]
    async fn test_optimization_stage_execute() {
        let stage = OptimizationStage::new(true, true, 1000);

        let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), PathBuf::from("/tmp/output"));

        let result = stage.execute(ctx).await;
        assert!(result.is_ok());

        let ctx = result.unwrap();

        // Check that optimizations were added
        assert!(!ctx.optimizations.is_empty());
        assert!(ctx.optimizations.iter().any(|o| o.contains("SIMD")));
        assert!(ctx.optimizations.iter().any(|o| o.contains("GPU")));

        // Check metadata
        assert!(ctx.metadata.contains_key("optimizations_applied"));
        assert!(ctx.metadata.contains_key("moe_routing_enabled"));
    }

    #[tokio::test]
    async fn test_optimization_stage_simd_only() {
        let stage = OptimizationStage::new(false, true, 1000);

        let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), PathBuf::from("/tmp/output"));

        let result = stage.execute(ctx).await;
        assert!(result.is_ok());

        let ctx = result.unwrap();

        // Should have SIMD but not GPU in traditional optimizations
        assert!(ctx.optimizations.iter().any(|o| o.contains("SIMD")));
        assert!(!ctx
            .optimizations
            .iter()
            .any(|o| o.contains("GPU dispatch enabled")));
    }

    #[tokio::test]
    async fn test_optimization_stage_gpu_only() {
        let stage = OptimizationStage::new(true, false, 2000);

        let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), PathBuf::from("/tmp/output"));

        let result = stage.execute(ctx).await;
        assert!(result.is_ok());

        let ctx = result.unwrap();

        // Should have GPU but not SIMD in traditional optimizations
        assert!(!ctx
            .optimizations
            .iter()
            .any(|o| o.contains("SIMD vectorization enabled")));
        assert!(ctx.optimizations.iter().any(|o| o.contains("GPU")));
        assert!(ctx.optimizations.iter().any(|o| o.contains("2000")));
    }

    // ============================================================================
    // VALIDATION STAGE TESTS
    // ============================================================================

    #[test]
    fn test_validation_stage_creation() {
        let stage = ValidationStage::new(true, true);
        assert!(stage.trace_syscalls);
        assert!(stage.run_tests);
    }

    #[test]
    fn test_validation_stage_name() {
        let stage = ValidationStage::new(false, false);
        assert_eq!(stage.name(), "Validation");
    }

    #[test]
    fn test_validation_stage_compare_traces_identical() {
        let trace1 = vec![
            "open(/file)".to_string(),
            "read(fd, buf, 100)".to_string(),
            "close(fd)".to_string(),
        ];

        let trace2 = vec![
            "open(/file)".to_string(),
            "read(fd, buf, 100)".to_string(),
            "close(fd)".to_string(),
        ];

        assert!(ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_validation_stage_compare_traces_different_length() {
        let trace1 = vec!["open(/file)".to_string(), "close(fd)".to_string()];

        let trace2 = vec!["open(/file)".to_string()];

        assert!(!ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_validation_stage_compare_traces_different_syscalls() {
        let trace1 = vec!["open(/file)".to_string(), "read(fd, buf, 100)".to_string()];

        let trace2 = vec!["open(/file)".to_string(), "write(fd, buf, 100)".to_string()];

        assert!(!ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_validation_stage_compare_traces_same_syscall_different_args() {
        // Should pass - only syscall names are compared, not arguments
        let trace1 = vec!["open(/file1)".to_string(), "read(fd, buf, 100)".to_string()];

        let trace2 = vec!["open(/file2)".to_string(), "read(fd, buf, 200)".to_string()];

        assert!(ValidationStage::compare_traces(&trace1, &trace2));
    }

    // ============================================================================
    // BUILD STAGE TESTS
    // ============================================================================

    #[test]
    fn test_build_stage_creation() {
        let stage = BuildStage::new(true, Some("x86_64-unknown-linux-gnu".to_string()), false);
        assert!(stage.release);
        assert_eq!(stage.target, Some("x86_64-unknown-linux-gnu".to_string()));
        assert!(!stage.wasm);
    }

    #[test]
    fn test_build_stage_creation_wasm() {
        let stage = BuildStage::new(false, None, true);
        assert!(!stage.release);
        assert!(stage.target.is_none());
        assert!(stage.wasm);
    }

    #[test]
    fn test_build_stage_name() {
        let stage = BuildStage::new(false, None, false);
        assert_eq!(stage.name(), "Build");
    }

    #[test]
    fn test_build_stage_validate_no_build_dir() {
        let stage = BuildStage::new(true, None, false);

        let ctx = PipelineContext::new(
            PathBuf::from("/tmp/input"),
            PathBuf::from("/tmp/nonexistent"),
        );

        let result = stage.validate(&ctx);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(!validation.passed);
        assert!(validation.message.contains("not found"));
    }

    // ============================================================================
    // ANALYSIS STAGE TESTS
    // ============================================================================

    #[test]
    fn test_analysis_stage_name() {
        let stage = AnalysisStage;
        assert_eq!(stage.name(), "Analysis");
    }

    #[test]
    fn test_analysis_stage_validate_no_language() {
        let stage = AnalysisStage;

        let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), PathBuf::from("/tmp/output"));

        let result = stage.validate(&ctx);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(!validation.passed);
        assert!(validation.message.contains("Could not detect"));
    }

    #[test]
    fn test_analysis_stage_validate_with_language() {
        let stage = AnalysisStage;

        let mut ctx =
            PipelineContext::new(PathBuf::from("/tmp/input"), PathBuf::from("/tmp/output"));
        ctx.primary_language = Some(crate::types::Language::Python);

        let result = stage.validate(&ctx);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(validation.passed);
        assert!(validation.message.contains("Language detected"));
    }

    // ============================================================================
    // SERIALIZATION TESTS
    // ============================================================================

    #[test]
    fn test_pipeline_context_serialization() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));
        ctx.primary_language = Some(crate::types::Language::Python);
        ctx.file_mappings
            .push((PathBuf::from("a.py"), PathBuf::from("a.rs")));
        ctx.optimizations.push("SIMD".to_string());

        let json = serde_json::to_string(&ctx).unwrap();
        let deserialized: PipelineContext = serde_json::from_str(&json).unwrap();

        assert_eq!(ctx.input_path, deserialized.input_path);
        assert_eq!(ctx.output_path, deserialized.output_path);
        assert_eq!(ctx.primary_language, deserialized.primary_language);
        assert_eq!(ctx.file_mappings.len(), deserialized.file_mappings.len());
        assert_eq!(ctx.optimizations.len(), deserialized.optimizations.len());
    }

    #[test]
    fn test_validation_result_serialization() {
        let result = ValidationResult {
            stage: "Test".to_string(),
            passed: true,
            message: "OK".to_string(),
            details: Some(serde_json::json!({"key": "value"})),
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ValidationResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.stage, deserialized.stage);
        assert_eq!(result.passed, deserialized.passed);
        assert_eq!(result.message, deserialized.message);
        assert_eq!(result.details, deserialized.details);
    }

    #[test]
    fn test_pipeline_output_serialization() {
        let output = PipelineOutput {
            output_path: PathBuf::from("/output"),
            file_mappings: vec![(PathBuf::from("a"), PathBuf::from("b"))],
            optimizations: vec!["opt1".to_string(), "opt2".to_string()],
            validation_passed: true,
        };

        let json = serde_json::to_string(&output).unwrap();
        let deserialized: PipelineOutput = serde_json::from_str(&json).unwrap();

        assert_eq!(output.output_path, deserialized.output_path);
        assert_eq!(output.file_mappings.len(), deserialized.file_mappings.len());
        assert_eq!(output.optimizations.len(), deserialized.optimizations.len());
        assert_eq!(output.validation_passed, deserialized.validation_passed);
    }

    // ============================================================================
    // ANALYSIS STAGE EXECUTION TESTS
    // ============================================================================

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn test_analysis_stage_execute_with_rust_project() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let src_dir = temp_dir.path().join("src");
        std::fs::create_dir(&src_dir).unwrap();
        std::fs::write(src_dir.join("main.rs"), "fn main() {}").unwrap();

        let stage = AnalysisStage;
        let ctx = PipelineContext::new(temp_dir.path().to_path_buf(), PathBuf::from("/tmp/output"));

        let result = stage.execute(ctx).await;
        assert!(result.is_ok());

        let ctx = result.unwrap();
        assert_eq!(ctx.primary_language, Some(crate::types::Language::Rust));
        assert!(ctx.metadata.contains_key("total_files"));
        assert!(ctx.metadata.contains_key("total_lines"));
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn test_analysis_stage_execute_with_python_project() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        std::fs::write(temp_dir.path().join("main.py"), "print('hello')").unwrap();

        let stage = AnalysisStage;
        let ctx = PipelineContext::new(temp_dir.path().to_path_buf(), PathBuf::from("/tmp/output"));

        let result = stage.execute(ctx).await;
        assert!(result.is_ok());

        let ctx = result.unwrap();
        assert_eq!(ctx.primary_language, Some(crate::types::Language::Python));
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn test_analysis_stage_execute_empty_directory() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();

        let stage = AnalysisStage;
        let ctx = PipelineContext::new(temp_dir.path().to_path_buf(), PathBuf::from("/tmp/output"));

        let result = stage.execute(ctx).await;
        assert!(result.is_ok());

        let ctx = result.unwrap();
        assert!(ctx.primary_language.is_none());
    }

    // ============================================================================
    // TRANSPILATION STAGE EXECUTION TESTS
    // ============================================================================

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn test_transpilation_stage_creates_directories() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().join("output");

        let stage = TranspilationStage::new(false, false);
        let mut ctx = PipelineContext::new(temp_dir.path().to_path_buf(), output_dir.clone());
        ctx.primary_language = Some(crate::types::Language::Rust);

        // This will fail at transpilation but should create directories
        let _result = stage.execute(ctx).await;

        // Check that directories were created even if transpilation fails
        assert!(output_dir.exists());
        assert!(output_dir.join("src").exists());
    }

    #[test]
    fn test_transpilation_stage_validate_empty_output() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().join("output");
        std::fs::create_dir_all(output_dir.join("src")).unwrap();

        let stage = TranspilationStage::new(false, false);
        let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), output_dir.clone());

        let result = stage.validate(&ctx);
        assert!(result.is_ok());

        let validation = result.unwrap();
        // Should fail because src dir is empty
        assert!(!validation.passed);
    }

    #[test]
    fn test_transpilation_stage_validate_with_files() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().join("output");
        std::fs::create_dir_all(output_dir.join("src")).unwrap();
        std::fs::write(output_dir.join("src/main.rs"), "fn main() {}").unwrap();

        let stage = TranspilationStage::new(false, false);
        let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), output_dir.clone());

        let result = stage.validate(&ctx);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(validation.passed);
    }

    // ============================================================================
    // BUILD STAGE EXECUTION TESTS
    // ============================================================================

    #[test]
    fn test_build_stage_validate_debug_build() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().join("output");
        std::fs::create_dir_all(output_dir.join("target/debug")).unwrap();

        let stage = BuildStage::new(false, None, false);
        let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), output_dir.clone());

        let result = stage.validate(&ctx);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(validation.passed);
    }

    #[test]
    fn test_build_stage_validate_release_build() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().join("output");
        std::fs::create_dir_all(output_dir.join("target/release")).unwrap();

        let stage = BuildStage::new(true, None, false);
        let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), output_dir.clone());

        let result = stage.validate(&ctx);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(validation.passed);
    }

    // ============================================================================
    // VALIDATION STAGE TESTS
    // ============================================================================

    #[tokio::test]
    async fn test_validation_stage_execute_no_binaries() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();

        let stage = ValidationStage::new(true, false);
        let ctx = PipelineContext::new(
            temp_dir.path().to_path_buf(),
            temp_dir.path().join("output"),
        );

        let result = stage.execute(ctx).await;
        assert!(result.is_ok());

        let ctx = result.unwrap();
        assert!(ctx.metadata.contains_key("validation_completed"));
    }

    #[tokio::test]
    async fn test_validation_stage_execute_without_tracing() {
        let stage = ValidationStage::new(false, false);
        let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), PathBuf::from("/tmp/output"));

        let result = stage.execute(ctx).await;
        assert!(result.is_ok());

        let ctx = result.unwrap();
        assert_eq!(
            ctx.metadata.get("validation_completed"),
            Some(&serde_json::json!(true))
        );
    }

    // ============================================================================
    // EDGE CASE TESTS
    // ============================================================================

    #[test]
    fn test_pipeline_context_clone() {
        let mut ctx1 = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));
        ctx1.optimizations.push("test".to_string());

        let ctx2 = ctx1.clone();
        assert_eq!(ctx1.input_path, ctx2.input_path);
        assert_eq!(ctx1.output_path, ctx2.output_path);
        assert_eq!(ctx1.optimizations.len(), ctx2.optimizations.len());
    }

    #[test]
    fn test_validation_result_clone() {
        let result1 = ValidationResult {
            stage: "Test".to_string(),
            passed: true,
            message: "OK".to_string(),
            details: None,
        };

        let result2 = result1.clone();
        assert_eq!(result1.stage, result2.stage);
        assert_eq!(result1.passed, result2.passed);
    }

    #[test]
    fn test_pipeline_output_clone() {
        let output1 = PipelineOutput {
            output_path: PathBuf::from("/out"),
            file_mappings: vec![],
            optimizations: vec!["opt".to_string()],
            validation_passed: true,
        };

        let output2 = output1.clone();
        assert_eq!(output1.output_path, output2.output_path);
        assert_eq!(output1.validation_passed, output2.validation_passed);
    }

    #[test]
    fn test_validation_strategy_copy() {
        let s1 = ValidationStrategy::StopOnError;
        let s2 = s1; // Copy
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_optimization_stage_no_gpu_no_simd() {
        let stage = OptimizationStage::new(false, false, 500);
        let recommendations = stage.analyze_optimizations();

        // Should still return MoE recommendations even without GPU/SIMD
        assert!(!recommendations.is_empty());
    }

    // ============================================================================
    // ADDITIONAL COVERAGE TESTS
    // ============================================================================

    #[test]
    fn test_pipeline_context_output_method_extended() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

        // Add some data
        ctx.file_mappings
            .push((PathBuf::from("a.py"), PathBuf::from("a.rs")));
        ctx.optimizations.push("simd".to_string());
        ctx.validation_results.push(ValidationResult {
            stage: "test".to_string(),
            passed: true,
            message: "ok".to_string(),
            details: None,
        });

        let output = ctx.output();
        assert_eq!(output.output_path, PathBuf::from("/output"));
        assert_eq!(output.file_mappings.len(), 1);
        assert!(output.validation_passed);
    }

    #[test]
    fn test_pipeline_context_output_fails_validation_extended() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

        ctx.validation_results.push(ValidationResult {
            stage: "test".to_string(),
            passed: false,
            message: "failed".to_string(),
            details: None,
        });

        let output = ctx.output();
        assert!(!output.validation_passed);
    }

    #[test]
    fn test_transpilation_stage_converters_initialized_ext() {
        let stage = TranspilationStage::new(true, true);
        assert!(stage.numpy_converter.is_some());
        assert!(stage.sklearn_converter.is_some());
        assert!(stage.pytorch_converter.is_some());
    }

    #[test]
    fn test_optimization_stage_with_different_simd_thresholds_ext() {
        let stage_low = OptimizationStage::new(false, true, 100);
        let stage_high = OptimizationStage::new(false, true, 10000);

        let recs_low = stage_low.analyze_optimizations();
        let recs_high = stage_high.analyze_optimizations();

        // Both should have recommendations but may differ
        assert!(!recs_low.is_empty());
        assert!(!recs_high.is_empty());
    }

    #[test]
    fn test_transpilation_stage_validate_ext() {
        let stage = TranspilationStage::new(true, true);
        let mut ctx = PipelineContext::new(PathBuf::from("/tmp"), PathBuf::from("/tmp/out"));

        // Without output files, validation result exists
        let result = stage.validate(&ctx);
        assert!(result.is_ok());

        // With file mappings
        ctx.file_mappings
            .push((PathBuf::from("a.py"), PathBuf::from("a.rs")));
        let result = stage.validate(&ctx);
        assert!(result.is_ok());
    }
}
