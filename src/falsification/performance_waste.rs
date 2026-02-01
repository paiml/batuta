//! Section 5: Performance & Waste (Muda) Elimination (PW-01 to PW-15)
//!
//! Toyota Way efficiency principles applied to compute.
//!
//! # TPS Principles
//!
//! - **Muda (Waste)**: Waiting, Transport, Motion, Overprocessing elimination
//! - **Jidoka**: Cost-based backend selection
//! - **Genchi Genbutsu**: Measure actual performance

use super::types::{CheckItem, Evidence, EvidenceType, Severity};
use std::path::Path;
use std::time::Instant;

/// Evaluate all Performance & Waste checks.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_pcie_rule(project_path),
        check_simd_speedup(project_path),
        check_wasm_performance(project_path),
        check_inference_latency(project_path),
        check_batch_efficiency(project_path),
        check_parallel_scaling(project_path),
        check_model_loading(project_path),
        check_startup_time(project_path),
        check_test_suite_time(project_path),
        check_overprocessing(project_path),
        check_zero_copy(project_path),
        check_cache_efficiency(project_path),
        check_cost_model(project_path),
        check_transport_minimization(project_path),
        check_inventory_minimization(project_path),
    ]
}

/// PW-01: 5× PCIe Rule Validation
pub fn check_pcie_rule(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "PW-01",
        "5× PCIe Rule Validation",
        "GPU dispatch beneficial when compute > 5× transfer",
    )
    .with_severity(Severity::Major)
    .with_tps("Cost-based backend selection");

    let has_gpu = check_for_pattern(project_path, &["wgpu", "cuda", "gpu", "GPU"]);
    let has_cost_model = check_for_pattern(
        project_path,
        &["cost_model", "crossover", "dispatch_threshold"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("PCIe rule: gpu={}, cost_model={}", has_gpu, has_cost_model),
        data: None,
        files: Vec::new(),
    });

    if !has_gpu || has_cost_model {
        item = item.pass();
    } else {
        item = item.partial("GPU usage without cost model");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-02: SIMD Speedup Verification
pub fn check_simd_speedup(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "PW-02",
        "SIMD Speedup Verification",
        "AVX2 provides >2× speedup over scalar",
    )
    .with_severity(Severity::Major)
    .with_tps("Muda (Waiting) reduction");

    let has_simd = check_for_pattern(
        project_path,
        &["simd", "avx2", "avx512", "neon", "target_feature"],
    );
    let has_benchmarks = project_path.join("benches").exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("SIMD: impl={}, benchmarks={}", has_simd, has_benchmarks),
        data: None,
        files: Vec::new(),
    });

    if has_simd && has_benchmarks {
        item = item.pass();
    } else if has_simd {
        item = item.partial("SIMD without speedup benchmarks");
    } else {
        item = item.partial("No SIMD optimization");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-03: WASM Performance Ratio
pub fn check_wasm_performance(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "PW-03",
        "WASM Performance Ratio",
        "WASM achieves 50-90% native performance",
    )
    .with_severity(Severity::Major)
    .with_tps("Edge deployment efficiency");

    let has_wasm = check_for_pattern(project_path, &["wasm", "wasm32", "wasm-bindgen"]);
    let has_perf_tests = check_for_pattern(project_path, &["wasm_bench", "native_comparison"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "WASM perf: wasm={}, benchmarks={}",
            has_wasm, has_perf_tests
        ),
        data: None,
        files: Vec::new(),
    });

    if !has_wasm || has_perf_tests {
        item = item.pass();
    } else {
        item = item.partial("WASM without performance comparison");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-04: Inference Latency SLA
pub fn check_inference_latency(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "PW-04",
        "Inference Latency SLA",
        "Inference <50ms/token on CPU",
    )
    .with_severity(Severity::Major)
    .with_tps("Muda (Waiting) elimination");

    let has_inference = check_for_pattern(project_path, &["inference", "infer", "predict"]);
    let has_latency_tests = check_for_pattern(project_path, &["latency", "p99", "benchmark"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Latency: inference={}, tests={}",
            has_inference, has_latency_tests
        ),
        data: None,
        files: Vec::new(),
    });

    if !has_inference || has_latency_tests {
        item = item.pass();
    } else {
        item = item.partial("Inference without latency benchmarks");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-05: Batch Processing Efficiency
pub fn check_batch_efficiency(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "PW-05",
        "Batch Processing Efficiency",
        "Batching provides near-linear throughput scaling",
    )
    .with_severity(Severity::Major)
    .with_tps("Resource utilization");

    let has_batching = check_for_pattern(project_path, &["batch", "Batch", "batch_size"]);
    let has_scaling_tests = check_for_pattern(
        project_path,
        &["batch_scaling", "throughput", "batch_bench"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Batch: impl={}, tests={}", has_batching, has_scaling_tests),
        data: None,
        files: Vec::new(),
    });

    if !has_batching || has_scaling_tests {
        item = item.pass();
    } else {
        item = item.partial("Batching without scaling verification");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-06: Parallel Scaling Efficiency
pub fn check_parallel_scaling(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "PW-06",
        "Parallel Scaling Efficiency",
        "Multi-threaded operations scale with cores",
    )
    .with_severity(Severity::Major)
    .with_tps("Resource utilization");

    let has_parallel = check_for_pattern(project_path, &["rayon", "parallel", "thread", "spawn"]);
    let has_scaling_tests =
        check_for_pattern(project_path, &["parallel_bench", "scaling", "amdahl"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Parallel: impl={}, tests={}",
            has_parallel, has_scaling_tests
        ),
        data: None,
        files: Vec::new(),
    });

    if has_parallel && has_scaling_tests {
        item = item.pass();
    } else if has_parallel {
        item = item.partial("Parallelization without scaling benchmarks");
    } else {
        item = item.partial("No parallel optimization");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-07: Model Loading Time
pub fn check_model_loading(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new("PW-07", "Model Loading Time", "Model loading <2s per GB")
        .with_severity(Severity::Minor)
        .with_tps("Muda (Waiting)");

    let has_model_loading = check_for_pattern(project_path, &["load_model", "ModelLoader", "mmap"]);
    let has_loading_tests = check_for_pattern(project_path, &["load_bench", "loading_time"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Loading: impl={}, tests={}",
            has_model_loading, has_loading_tests
        ),
        data: None,
        files: Vec::new(),
    });

    if !has_model_loading || has_loading_tests {
        item = item.pass();
    } else {
        item = item.partial("Model loading without benchmarks");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-08: Startup Time
pub fn check_startup_time(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new("PW-08", "Startup Time", "CLI startup <100ms")
        .with_severity(Severity::Minor)
        .with_tps("Developer experience");

    // Check for hyperfine in CI or Makefile
    let makefile = project_path.join("Makefile");
    let has_startup_bench = makefile
        .exists()
        .then(|| std::fs::read_to_string(&makefile).ok())
        .flatten()
        .map(|c| c.contains("hyperfine") || c.contains("startup"))
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Startup: benchmark={}", has_startup_bench),
        data: None,
        files: Vec::new(),
    });

    if has_startup_bench {
        item = item.pass();
    } else {
        item = item.partial("No startup time benchmarking");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-09: Test Suite Performance
pub fn check_test_suite_time(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "PW-09",
        "Test Suite Performance",
        "Full test suite <5 minutes",
    )
    .with_severity(Severity::Major)
    .with_tps("Muda (Waiting) in CI");

    // Check for nextest (faster test runner)
    let has_nextest = project_path.join(".config/nextest.toml").exists();
    let has_parallel_tests = check_for_pattern(project_path, &["nextest", "test-threads"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Test perf: nextest={}, parallel={}",
            has_nextest, has_parallel_tests
        ),
        data: None,
        files: Vec::new(),
    });

    if has_nextest || has_parallel_tests {
        item = item.pass();
    } else {
        item = item.partial("Tests without optimization (consider nextest)");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-10: Overprocessing Detection
pub fn check_overprocessing(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "PW-10",
        "Overprocessing Detection",
        "Model complexity justified by improvement",
    )
    .with_severity(Severity::Major)
    .with_tps("Muda (Overprocessing)");

    let has_baseline = check_for_pattern(project_path, &["baseline", "simple_model", "comparison"]);
    let has_complexity_analysis =
        check_for_pattern(project_path, &["complexity", "flops", "parameters"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Overprocessing: baseline={}, analysis={}",
            has_baseline, has_complexity_analysis
        ),
        data: None,
        files: Vec::new(),
    });

    if has_baseline {
        item = item.pass();
    } else {
        let is_complex = check_for_pattern(project_path, &["transformer", "neural", "deep"]);
        if !is_complex {
            item = item.pass();
        } else {
            item = item.partial("Complex models without baseline comparison");
        }
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-11: Zero-Copy Operation Verification
pub fn check_zero_copy(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "PW-11",
        "Zero-Copy Operation Verification",
        "Hot paths operate without allocation",
    )
    .with_severity(Severity::Minor)
    .with_tps("Muda (Motion) — memory efficiency");

    let has_zero_copy =
        check_for_pattern(project_path, &["zero_copy", "no_alloc", "in_place", "&mut"]);
    let has_alloc_tests = check_for_pattern(
        project_path,
        &["allocation_free", "heaptrack", "alloc_count"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Zero-copy: impl={}, tests={}",
            has_zero_copy, has_alloc_tests
        ),
        data: None,
        files: Vec::new(),
    });

    if has_alloc_tests {
        item = item.pass();
    } else if has_zero_copy {
        item = item.partial("Zero-copy patterns (verify with tests)");
    } else {
        item = item.partial("No explicit zero-copy optimization");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-12: Cache Efficiency
pub fn check_cache_efficiency(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new("PW-12", "Cache Efficiency", "Cache >90% hit ratio")
        .with_severity(Severity::Minor)
        .with_tps("Muda (Waiting) reduction");

    let has_cache = check_for_pattern(project_path, &["cache", "Cache", "lru", "memoize"]);
    let has_cache_metrics =
        check_for_pattern(project_path, &["hit_ratio", "cache_stats", "cache_hit"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Cache: impl={}, metrics={}", has_cache, has_cache_metrics),
        data: None,
        files: Vec::new(),
    });

    if !has_cache || has_cache_metrics {
        item = item.pass();
    } else {
        item = item.partial("Cache without efficiency metrics");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-13: Cost Model Accuracy
pub fn check_cost_model(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "PW-13",
        "Cost Model Accuracy",
        "Backend selection predicts optimal choice >90%",
    )
    .with_severity(Severity::Major)
    .with_tps("Intelligent automation (Jidoka)");

    let has_cost_model = check_for_pattern(
        project_path,
        &["cost_model", "CostModel", "backend_selection"],
    );
    let has_accuracy_tests = check_for_pattern(project_path, &["prediction_accuracy", "cost_test"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Cost model: impl={}, tests={}",
            has_cost_model, has_accuracy_tests
        ),
        data: None,
        files: Vec::new(),
    });

    if !has_cost_model || has_accuracy_tests {
        item = item.pass();
    } else {
        item = item.partial("Cost model without accuracy verification");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-14: Data Transport Minimization
pub fn check_transport_minimization(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "PW-14",
        "Data Transport Minimization",
        "Data movement minimized through co-location",
    )
    .with_severity(Severity::Minor)
    .with_tps("Muda (Transport)");

    let has_colocation =
        check_for_pattern(project_path, &["colocation", "locality", "data_parallel"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Transport: colocation={}", has_colocation),
        data: None,
        files: Vec::new(),
    });

    let has_distributed = check_for_pattern(project_path, &["distributed", "network", "remote"]);
    if !has_distributed || has_colocation {
        item = item.pass();
    } else {
        item = item.partial("Distributed compute without locality optimization");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// PW-15: Inventory (Data) Minimization
pub fn check_inventory_minimization(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new("PW-15", "Inventory Minimization", "No hoarded unused data")
        .with_severity(Severity::Minor)
        .with_tps("Muda (Inventory)");

    let has_lifecycle = check_for_pattern(
        project_path,
        &["lifecycle", "retention", "ttl", "expiration"],
    );
    let has_cleanup = check_for_pattern(project_path, &["cleanup", "prune", "garbage_collect"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Inventory: lifecycle={}, cleanup={}",
            has_lifecycle, has_cleanup
        ),
        data: None,
        files: Vec::new(),
    });

    let stores_data = check_for_pattern(project_path, &["store", "persist", "save"]);
    if !stores_data || has_lifecycle || has_cleanup {
        item = item.pass();
    } else {
        item = item.partial("Data storage without lifecycle management");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

fn check_for_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    super::helpers::source_contains_pattern(project_path, patterns)
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
}
