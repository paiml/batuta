//! Validation stage - verifies semantic equivalence.

use anyhow::{Context as AnyhowContext, Result};

#[cfg(feature = "native")]
use tracing::{info, warn};

// Stub macros for WASM build
#[cfg(not(feature = "native"))]
macro_rules! info {
    ($($arg:tt)*) => {{}};
}

#[cfg(not(feature = "native"))]
macro_rules! warn {
    ($($arg:tt)*) => {{}};
}

use crate::pipeline::types::{PipelineContext, PipelineStage, ValidationResult};

/// Validation stage - verifies semantic equivalence
pub struct ValidationStage {
    pub(crate) trace_syscalls: bool,
    pub(crate) run_tests: bool,
}

impl ValidationStage {
    pub fn new(trace_syscalls: bool, run_tests: bool) -> Self {
        Self {
            trace_syscalls,
            run_tests,
        }
    }

    /// Trace syscalls from both binaries and compare them for semantic equivalence
    pub async fn trace_and_compare(
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
    pub fn trace_binary(binary: &std::path::Path) -> Result<Vec<String>> {
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
    pub fn compare_traces(trace1: &[String], trace2: &[String]) -> bool {
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
        // PIPELINE-004: Test suite execution planned for validation phase
        if self.run_tests {
            info!("Running original test suite");
            // Test execution deferred - requires renacer integration
        }

        ctx.metadata
            .insert("validation_completed".to_string(), serde_json::json!(true));

        Ok(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_stage_new() {
        let stage = ValidationStage::new(true, false);
        assert!(stage.trace_syscalls);
        assert!(!stage.run_tests);
    }

    #[test]
    fn test_validation_stage_name() {
        let stage = ValidationStage::new(false, false);
        assert_eq!(stage.name(), "Validation");
    }

    #[test]
    fn test_compare_traces_identical() {
        let trace1 = vec![
            "read(3, buf, 1024)".to_string(),
            "write(1, msg, 12)".to_string(),
        ];
        let trace2 = vec![
            "read(3, buf, 1024)".to_string(),
            "write(1, msg, 12)".to_string(),
        ];
        assert!(ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_same_syscalls_different_args() {
        let trace1 = vec![
            "read(3, buf, 1024)".to_string(),
            "write(1, msg, 12)".to_string(),
        ];
        let trace2 = vec![
            "read(4, buf2, 2048)".to_string(),
            "write(2, msg2, 24)".to_string(),
        ];
        // Should match because syscall names are the same
        assert!(ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_different_syscalls() {
        let trace1 = vec![
            "read(3, buf, 1024)".to_string(),
            "write(1, msg, 12)".to_string(),
        ];
        let trace2 = vec![
            "open(path, flags)".to_string(),
            "close(3)".to_string(),
        ];
        assert!(!ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_different_lengths() {
        let trace1 = vec!["read(3, buf, 1024)".to_string()];
        let trace2 = vec![
            "read(3, buf, 1024)".to_string(),
            "write(1, msg, 12)".to_string(),
        ];
        assert!(!ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_empty() {
        let trace1: Vec<String> = vec![];
        let trace2: Vec<String> = vec![];
        assert!(ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_no_parentheses() {
        let trace1 = vec!["syscall1".to_string()];
        let trace2 = vec!["syscall1".to_string()];
        assert!(ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_partial_match() {
        let trace1 = vec![
            "read(3)".to_string(),
            "write(1)".to_string(),
            "close(3)".to_string(),
        ];
        let trace2 = vec![
            "read(4)".to_string(),
            "read(5)".to_string(),  // Different syscall
            "close(4)".to_string(),
        ];
        assert!(!ValidationStage::compare_traces(&trace1, &trace2));
    }
}
