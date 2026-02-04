//! Pipeline types and trait definitions.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
