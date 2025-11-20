use anyhow::{Context as AnyhowContext, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

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
}

impl TranspilationStage {
    pub fn new(incremental: bool, cache: bool) -> Self {
        Self { incremental, cache }
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

        // Detect available tools
        let tools = crate::tools::ToolRegistry::detect();

        // Get transpiler for primary language
        if let Some(lang) = &ctx.primary_language {
            if let Some(transpiler) = tools.get_transpiler_for_language(lang) {
                // Build arguments
                let input_path_str = ctx.input_path.to_string_lossy().to_string();
                let output_path_str = ctx.output_path.to_string_lossy().to_string();

                let mut args = vec![
                    "--input",
                    &input_path_str,
                    "--output",
                    &output_path_str,
                ];

                if self.incremental {
                    args.push("--incremental");
                }

                if self.cache {
                    args.push("--cache");
                }

                // Run transpiler
                crate::tools::run_tool(&transpiler.name, &args, Some(&ctx.input_path))?;

                ctx.metadata.insert(
                    "transpiler".to_string(),
                    serde_json::json!(transpiler.name),
                );
            } else {
                anyhow::bail!("No transpiler available for language: {}", lang);
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

/// Optimization stage - applies performance optimizations
pub struct OptimizationStage {
    enable_gpu: bool,
    enable_simd: bool,
    gpu_threshold: usize,
}

impl OptimizationStage {
    pub fn new(enable_gpu: bool, enable_simd: bool, gpu_threshold: usize) -> Self {
        Self {
            enable_gpu,
            enable_simd,
            gpu_threshold,
        }
    }
}

#[async_trait::async_trait]
impl PipelineStage for OptimizationStage {
    fn name(&self) -> &str {
        "Optimization"
    }

    async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        info!("Applying optimizations (GPU: {}, SIMD: {})",
            self.enable_gpu, self.enable_simd);

        if self.enable_simd {
            ctx.optimizations.push("SIMD vectorization".to_string());
        }

        if self.enable_gpu {
            ctx.optimizations.push(format!("GPU dispatch (threshold: {})", self.gpu_threshold));
        }

        // TODO: Actual optimization implementation with Trueno integration (BATUTA-007)
        // For now, just record what would be applied

        ctx.metadata.insert(
            "optimizations_applied".to_string(),
            serde_json::json!(ctx.optimizations),
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
}

#[async_trait::async_trait]
impl PipelineStage for ValidationStage {
    fn name(&self) -> &str {
        "Validation"
    }

    async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        info!("Validating semantic equivalence");

        // TODO: Actual validation with Renacer integration (BATUTA-008)
        // For now, mark as validated

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
