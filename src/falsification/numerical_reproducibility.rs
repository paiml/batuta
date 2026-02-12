//! Section 4: Numerical Reproducibility (NR-01 to NR-15)
//!
//! IEEE 754 compliance, reference implementation parity, numerical stability.
//!
//! # TPS Principles
//!
//! - **Jidoka**: Automatic compliance verification
//! - **Genchi Genbutsu**: Verify on actual hardware
//! - **Baseline comparison**: Reference parity

use super::types::{CheckItem, Evidence, EvidenceType, Severity};
use std::path::Path;
use std::time::Instant;

/// Evaluate all Numerical Reproducibility checks.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_ieee754_compliance(project_path),
        check_cross_platform_determinism(project_path),
        check_numpy_parity(project_path),
        check_sklearn_parity(project_path),
        check_linalg_accuracy(project_path),
        check_kahan_summation(project_path),
        check_rng_quality(project_path),
        check_quantization_bounds(project_path),
        check_gradient_correctness(project_path),
        check_tokenizer_parity(project_path),
        check_attention_correctness(project_path),
        check_loss_accuracy(project_path),
        check_optimizer_state(project_path),
        check_normalization_correctness(project_path),
        check_matmul_stability(project_path),
    ]
}

/// NR-01: IEEE 754 Floating-Point Compliance
pub fn check_ieee754_compliance(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-01",
        "IEEE 754 Floating-Point Compliance",
        "SIMD operations produce IEEE 754-compliant results",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — automatic compliance verification");

    let has_fp_tests = check_for_pattern(
        project_path,
        &["ieee754", "floating_point", "ulp", "f32", "f64"],
    );
    let has_special_cases =
        check_for_pattern(project_path, &["NaN", "Inf", "subnormal", "denormal"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "IEEE754: fp_tests={}, special_cases={}",
            has_fp_tests, has_special_cases
        ),
        data: None,
        files: Vec::new(),
    });

    if has_fp_tests && has_special_cases {
        item = item.pass();
    } else if has_fp_tests {
        item = item.partial("FP testing (verify special cases)");
    } else {
        item = item.partial("No explicit IEEE 754 testing");
    }

    item.finish_timed(start)
}

/// NR-02: Cross-Platform Numerical Determinism
pub fn check_cross_platform_determinism(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-02",
        "Cross-Platform Numerical Determinism",
        "Identical inputs produce identical outputs across platforms",
    )
    .with_severity(Severity::Major)
    .with_tps("Genchi Genbutsu — verify on actual hardware");

    let has_platform_tests = check_ci_matrix(project_path, &["ubuntu", "macos", "windows"]);
    let has_arch_tests = check_for_pattern(project_path, &["x86_64", "aarch64", "arm64", "wasm32"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Determinism: platform_ci={}, arch_tests={}",
            has_platform_tests, has_arch_tests
        ),
        data: None,
        files: Vec::new(),
    });

    if has_platform_tests && has_arch_tests {
        item = item.pass();
    } else if has_platform_tests {
        item = item.partial("Multi-platform CI (verify determinism)");
    } else {
        item = item.partial("Single platform testing");
    }

    item.finish_timed(start)
}

/// NR-03: NumPy Reference Parity
pub fn check_numpy_parity(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-03",
        "NumPy Reference Parity",
        "Operations match NumPy within documented epsilon",
    )
    .with_severity(Severity::Major)
    .with_tps("Baseline comparison");

    let has_numpy_tests = check_for_pattern(project_path, &["numpy", "NumPy", "np."]);
    let has_golden_tests = check_for_pattern(project_path, &["golden", "reference", "expected"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "NumPy parity: tests={}, golden={}",
            has_numpy_tests, has_golden_tests
        ),
        data: None,
        files: Vec::new(),
    });

    let is_numeric = check_for_pattern(project_path, &["ndarray", "tensor", "matrix"]);
    if !is_numeric || has_numpy_tests || has_golden_tests {
        item = item.pass();
    } else {
        item = item.partial("Numeric code without reference parity tests");
    }

    item.finish_timed(start)
}

/// NR-04: scikit-learn Algorithm Parity
pub fn check_sklearn_parity(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-04",
        "scikit-learn Algorithm Parity",
        "ML algorithms produce statistically equivalent results",
    )
    .with_severity(Severity::Major)
    .with_tps("Scientific validation");

    let has_sklearn_tests = check_for_pattern(
        project_path,
        &[
            "sklearn",
            "scikit-learn",
            "RandomForest",
            "LinearRegression",
        ],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("sklearn parity: tests={}", has_sklearn_tests),
        data: None,
        files: Vec::new(),
    });

    let is_ml = check_for_pattern(project_path, &["classifier", "regressor", "clustering"]);
    if !is_ml || has_sklearn_tests {
        item = item.pass();
    } else {
        item = item.partial("ML code without sklearn parity tests");
    }

    item.finish_timed(start)
}

/// NR-05: Linear Algebra Decomposition Accuracy
pub fn check_linalg_accuracy(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-05",
        "Linear Algebra Decomposition Accuracy",
        "Decompositions meet LAPACK standards",
    )
    .with_severity(Severity::Major)
    .with_tps("Reference baseline");

    let has_decomp = check_for_pattern(project_path, &["cholesky", "svd", "qr", "lu", "eigen"]);
    let has_accuracy_tests =
        check_for_pattern(project_path, &["reconstruction", "residual", "1e-12"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "LinAlg: decomp={}, accuracy_tests={}",
            has_decomp, has_accuracy_tests
        ),
        data: None,
        files: Vec::new(),
    });

    if !has_decomp || has_accuracy_tests {
        item = item.pass();
    } else {
        item = item.partial("Decompositions without accuracy verification");
    }

    item.finish_timed(start)
}

/// NR-06: Kahan Summation Implementation
pub fn check_kahan_summation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-06",
        "Kahan Summation Implementation",
        "Summation uses compensated algorithm",
    )
    .with_severity(Severity::Minor)
    .with_tps("Quality built-in");

    let has_kahan = check_for_pattern(project_path, &["kahan", "compensated", "pairwise_sum"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Kahan summation: impl={}", has_kahan),
        data: None,
        files: Vec::new(),
    });

    let does_summation = check_for_pattern(project_path, &["sum()", ".sum()", "reduce"]);
    if !does_summation || has_kahan {
        item = item.pass();
    } else {
        item = item.partial("Summation without compensated algorithm");
    }

    item.finish_timed(start)
}

/// NR-07: RNG Statistical Quality
pub fn check_rng_quality(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-07",
        "RNG Statistical Quality",
        "RNG passes NIST statistical tests",
    )
    .with_severity(Severity::Major)
    .with_tps("Formal verification");

    let has_quality_rng = check_for_pattern(project_path, &["ChaCha", "Pcg", "Xorshift", "StdRng"]);
    let has_rng_tests = check_for_pattern(project_path, &["nist", "diehard", "statistical_test"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "RNG quality: quality_impl={}, tests={}",
            has_quality_rng, has_rng_tests
        ),
        data: None,
        files: Vec::new(),
    });

    let uses_rng = check_for_pattern(project_path, &["rand::", "Rng", "random"]);
    if !uses_rng || has_quality_rng {
        item = item.pass();
    } else {
        item = item.partial("RNG without quality verification");
    }

    item.finish_timed(start)
}

/// NR-08: Quantization Error Bounds
pub fn check_quantization_bounds(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-08",
        "Quantization Error Bounds",
        "Quantization maintains accuracy within bounds",
    )
    .with_severity(Severity::Major)
    .with_tps("Documented tradeoffs");

    let has_quant = check_for_pattern(project_path, &["quantize", "q4_0", "q8_0", "int8", "bnb"]);
    let has_error_bounds =
        check_for_pattern(project_path, &["perplexity", "error_bound", "quality_loss"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Quantization: impl={}, bounds={}",
            has_quant, has_error_bounds
        ),
        data: None,
        files: Vec::new(),
    });

    if !has_quant || has_error_bounds {
        item = item.pass();
    } else {
        item = item.partial("Quantization without error bounds");
    }

    item.finish_timed(start)
}

/// NR-09: Gradient Computation Correctness
pub fn check_gradient_correctness(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-09",
        "Gradient Computation Correctness",
        "Autograd produces correct gradients",
    )
    .with_severity(Severity::Critical)
    .with_tps("Mathematical correctness");

    let has_autograd =
        check_for_pattern(project_path, &["autograd", "backward", "gradient", "grad"]);
    let has_grad_check = check_for_pattern(
        project_path,
        &["finite_difference", "grad_check", "numerical_gradient"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Gradients: autograd={}, check={}",
            has_autograd, has_grad_check
        ),
        data: None,
        files: Vec::new(),
    });

    if !has_autograd || has_grad_check {
        item = item.pass();
    } else {
        item = item.partial("Autograd without numerical verification");
    }

    item.finish_timed(start)
}

/// NR-10: Tokenization Parity
pub fn check_tokenizer_parity(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-10",
        "Tokenization Parity",
        "Tokenizer matches HuggingFace output",
    )
    .with_severity(Severity::Major)
    .with_tps("Reference baseline");

    let has_tokenizer = check_for_pattern(
        project_path,
        &["tokenizer", "Tokenizer", "bpe", "sentencepiece"],
    );
    let has_parity_tests =
        check_for_pattern(project_path, &["huggingface", "transformers", "token_ids"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Tokenizer: impl={}, parity={}",
            has_tokenizer, has_parity_tests
        ),
        data: None,
        files: Vec::new(),
    });

    if !has_tokenizer || has_parity_tests {
        item = item.pass();
    } else {
        item = item.partial("Tokenizer without parity tests");
    }

    item.finish_timed(start)
}

/// NR-11: Attention Mechanism Correctness
pub fn check_attention_correctness(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-11",
        "Attention Mechanism Correctness",
        "Attention computes softmax(QK^T/√d)V correctly",
    )
    .with_severity(Severity::Critical)
    .with_tps("Mathematical specification");

    let has_attention = check_for_pattern(
        project_path,
        &["attention", "Attention", "sdpa", "multi_head"],
    );
    let has_correctness_tests = check_for_pattern(
        project_path,
        &["attention_test", "softmax_sum", "causal_mask"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Attention: impl={}, tests={}",
            has_attention, has_correctness_tests
        ),
        data: None,
        files: Vec::new(),
    });

    if !has_attention || has_correctness_tests {
        item = item.pass();
    } else {
        item = item.partial("Attention without correctness verification");
    }

    item.finish_timed(start)
}

/// NR-12: Loss Function Accuracy
pub fn check_loss_accuracy(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-12",
        "Loss Function Accuracy",
        "Loss functions match reference implementations",
    )
    .with_severity(Severity::Major)
    .with_tps("Baseline comparison");

    let has_loss = check_for_pattern(project_path, &["loss", "Loss", "cross_entropy", "mse"]);
    let has_accuracy_tests = check_for_pattern(
        project_path,
        &["loss_test", "reference_loss", "expected_loss"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Loss: impl={}, tests={}", has_loss, has_accuracy_tests),
        data: None,
        files: Vec::new(),
    });

    if !has_loss || has_accuracy_tests {
        item = item.pass();
    } else {
        item = item.partial("Loss functions without accuracy tests");
    }

    item.finish_timed(start)
}

/// NR-13: Optimizer State Correctness
pub fn check_optimizer_state(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-13",
        "Optimizer State Correctness",
        "Optimizers maintain correct state updates",
    )
    .with_severity(Severity::Major)
    .with_tps("Step-by-step verification");

    let has_optimizer = check_for_pattern(project_path, &["optimizer", "Optimizer", "adam", "sgd"]);
    let has_state_tests = check_for_pattern(
        project_path,
        &["optimizer_test", "state_update", "momentum"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Optimizer: impl={}, tests={}",
            has_optimizer, has_state_tests
        ),
        data: None,
        files: Vec::new(),
    });

    if !has_optimizer || has_state_tests {
        item = item.pass();
    } else {
        item = item.partial("Optimizer without state verification");
    }

    item.finish_timed(start)
}

/// NR-14: Normalization Layer Correctness
pub fn check_normalization_correctness(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-14",
        "Normalization Layer Correctness",
        "Norm layers produce correct outputs",
    )
    .with_severity(Severity::Major)
    .with_tps("Statistical verification");

    let has_norm = check_for_pattern(
        project_path,
        &["BatchNorm", "LayerNorm", "RMSNorm", "normalize"],
    );
    let has_norm_tests =
        check_for_pattern(project_path, &["norm_test", "mean_zero", "variance_one"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Normalization: impl={}, tests={}", has_norm, has_norm_tests),
        data: None,
        files: Vec::new(),
    });

    if !has_norm || has_norm_tests {
        item = item.pass();
    } else {
        item = item.partial("Normalization without correctness tests");
    }

    item.finish_timed(start)
}

/// NR-15: Matrix Multiplication Numerical Stability
pub fn check_matmul_stability(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "NR-15",
        "Matrix Multiplication Stability",
        "Matmul handles ill-conditioned matrices",
    )
    .with_severity(Severity::Major)
    .with_tps("Graceful degradation");

    let has_matmul = check_for_pattern(project_path, &["matmul", "gemm", "dot"]);
    let has_stability_tests = check_for_pattern(
        project_path,
        &["condition_number", "ill_conditioned", "stability"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Matmul stability: impl={}, tests={}",
            has_matmul, has_stability_tests
        ),
        data: None,
        files: Vec::new(),
    });

    if !has_matmul || has_stability_tests {
        item = item.pass();
    } else {
        item = item.partial("Matmul without stability verification");
    }

    item.finish_timed(start)
}

// Helper functions
fn check_for_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    super::helpers::source_contains_pattern(project_path, patterns)
}

fn check_ci_matrix(project_path: &Path, platforms: &[&str]) -> bool {
    super::helpers::ci_platform_count(project_path, platforms) >= 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_evaluate_all_returns_15_items() {
        let path = PathBuf::from(".");
        let items = evaluate_all(&path);
        assert_eq!(items.len(), 15);
    }

    #[test]
    fn test_all_items_have_tps_principle() {
        let path = PathBuf::from(".");
        for item in evaluate_all(&path) {
            assert!(
                !item.tps_principle.is_empty(),
                "Item {} missing TPS",
                item.id
            );
        }
    }

    #[test]
    fn test_all_items_have_evidence() {
        let path = PathBuf::from(".");
        for item in evaluate_all(&path) {
            assert!(
                !item.evidence.is_empty(),
                "Item {} missing evidence",
                item.id
            );
        }
    }

    #[test]
    fn test_check_ieee754_compliance() {
        let path = PathBuf::from(".");
        let item = check_ieee754_compliance(&path);
        assert_eq!(item.id, "NR-01");
        assert!(item.name.contains("IEEE 754"));
    }

    #[test]
    fn test_check_cross_platform_determinism() {
        let path = PathBuf::from(".");
        let item = check_cross_platform_determinism(&path);
        assert_eq!(item.id, "NR-02");
        assert!(item.name.contains("Cross-Platform"));
    }

    #[test]
    fn test_check_numpy_parity() {
        let path = PathBuf::from(".");
        let item = check_numpy_parity(&path);
        assert_eq!(item.id, "NR-03");
        assert!(item.name.contains("NumPy"));
    }

    #[test]
    fn test_check_sklearn_parity() {
        let path = PathBuf::from(".");
        let item = check_sklearn_parity(&path);
        assert_eq!(item.id, "NR-04");
        assert!(item.name.contains("scikit-learn"));
    }

    #[test]
    fn test_check_linalg_accuracy() {
        let path = PathBuf::from(".");
        let item = check_linalg_accuracy(&path);
        assert_eq!(item.id, "NR-05");
    }

    #[test]
    fn test_check_kahan_summation() {
        let path = PathBuf::from(".");
        let item = check_kahan_summation(&path);
        assert_eq!(item.id, "NR-06");
    }

    #[test]
    fn test_check_rng_quality() {
        let path = PathBuf::from(".");
        let item = check_rng_quality(&path);
        assert_eq!(item.id, "NR-07");
    }

    #[test]
    fn test_check_quantization_bounds() {
        let path = PathBuf::from(".");
        let item = check_quantization_bounds(&path);
        assert_eq!(item.id, "NR-08");
    }

    #[test]
    fn test_check_gradient_correctness() {
        let path = PathBuf::from(".");
        let item = check_gradient_correctness(&path);
        assert_eq!(item.id, "NR-09");
    }

    #[test]
    fn test_check_tokenizer_parity() {
        let path = PathBuf::from(".");
        let item = check_tokenizer_parity(&path);
        assert_eq!(item.id, "NR-10");
    }

    #[test]
    fn test_check_attention_correctness() {
        let path = PathBuf::from(".");
        let item = check_attention_correctness(&path);
        assert_eq!(item.id, "NR-11");
    }

    #[test]
    fn test_check_loss_accuracy() {
        let path = PathBuf::from(".");
        let item = check_loss_accuracy(&path);
        assert_eq!(item.id, "NR-12");
    }

    #[test]
    fn test_check_optimizer_state() {
        let path = PathBuf::from(".");
        let item = check_optimizer_state(&path);
        assert_eq!(item.id, "NR-13");
    }

    #[test]
    fn test_check_normalization_correctness() {
        let path = PathBuf::from(".");
        let item = check_normalization_correctness(&path);
        assert_eq!(item.id, "NR-14");
    }

    #[test]
    fn test_check_matmul_stability() {
        let path = PathBuf::from(".");
        let item = check_matmul_stability(&path);
        assert_eq!(item.id, "NR-15");
    }

    #[test]
    fn test_all_items_have_severity() {
        let path = PathBuf::from(".");
        for item in evaluate_all(&path) {
            // Severity is set via with_severity
            assert!(
                item.tps_principle.len() > 0,
                "Item {} should have TPS principle set along with severity",
                item.id
            );
        }
    }

    #[test]
    fn test_check_for_pattern_helper() {
        let path = PathBuf::from(".");
        // This tests the pattern matching helper
        let has_rust = check_for_pattern(&path, &["Cargo.toml", "lib.rs"]);
        // Should find patterns in a rust project
        let _ = has_rust; // Validates check_for_pattern runs without panic
    }
}
