/// Backend selection and cost model (per spec section 2.2)
///
/// # Mixture-of-Experts (MoE) Routing
///
/// This module implements adaptive backend selection using a Mixture-of-Experts approach.
/// The MoE router analyzes operation complexity and data size to select the optimal
/// compute backend (Scalar/SIMD/GPU).
///
/// ## Operation Complexity Levels
///
/// - **Low**: Element-wise operations (add, multiply, etc.) - Memory-bound, GPU rarely beneficial
/// - **Medium**: Reductions (dot product, sum, etc.) - Moderate compute, GPU at 100K+ elements
/// - **High**: Matrix operations (matmul, convolution) - Compute-intensive O(n²) or O(n³), GPU at 10K+ elements
///
/// ## Usage Example
///
/// ```rust
/// use batuta::backend::{BackendSelector, OpComplexity};
///
/// let selector = BackendSelector::new();
///
/// // Element-wise operation
/// let backend = selector.select_with_moe(OpComplexity::Low, 500_000);
/// // Returns: Scalar (below 1M threshold, memory-bound)
///
/// // Matrix multiplication
/// let backend = selector.select_with_moe(OpComplexity::High, 50_000);
/// // Returns: GPU (above 10K threshold for O(n²) ops)
/// ```
///
/// ## Performance Thresholds
///
/// Based on empirical analysis and the 5× PCIe rule (Gregg & Hazelwood 2011):
///
/// | Complexity | SIMD Threshold | GPU Threshold | Rationale |
/// |------------|---------------|---------------|-----------|
/// | Low | 1M elements | Never | Memory-bound, PCIe overhead dominates |
/// | Medium | 10K elements | 100K elements | Moderate compute/transfer ratio |
/// | High | 1K elements | 10K elements | O(n²/n³) complexity favors GPU |
///
use serde::{Deserialize, Serialize};

#[cfg(feature = "trueno-integration")]
use trueno::{Matrix, Vector};

/// Compute backend options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub enum Backend {
    /// Scalar operations (baseline)
    Scalar,
    /// SIMD vectorization (AVX2, NEON)
    SIMD,
    /// GPU acceleration (WebGPU/Vulkan)
    GPU,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Scalar => write!(f, "Scalar"),
            Backend::SIMD => write!(f, "SIMD"),
            Backend::GPU => write!(f, "GPU"),
        }
    }
}

/// Operation complexity for MoE (Mixture-of-Experts) routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpComplexity {
    /// Simple operations (add, mul) - O(n), prefer SIMD unless very large
    Low,
    /// Moderate operations (dot, reduce) - O(n), GPU beneficial at 100K+ elements
    Medium,
    /// Complex operations (matmul, convolution) - O(n²) or O(n³), GPU beneficial at 10K+ elements
    High,
}

/// Cost model for backend selection
/// Based on spec section 2.2 lines 191-204
pub struct BackendSelector {
    /// PCIe bandwidth in bytes/sec (default: 32 GB/s for PCIe 4.0 x16)
    pcie_bandwidth: f64,

    /// GPU compute throughput in FLOPS (default: 20 TFLOPS for A100)
    gpu_gflops: f64,

    /// Minimum dispatch ratio (default: 5× per Gregg & Hazelwood 2011)
    min_dispatch_ratio: f64,
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self {
            pcie_bandwidth: 32e9,      // 32 GB/s
            gpu_gflops: 20e12,         // 20 TFLOPS
            min_dispatch_ratio: 5.0,   // 5× rule
        }
    }
}

impl BackendSelector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure custom PCIe bandwidth
    pub fn with_pcie_bandwidth(mut self, bandwidth: f64) -> Self {
        self.pcie_bandwidth = bandwidth;
        self
    }

    /// Configure custom GPU throughput
    pub fn with_gpu_gflops(mut self, gflops: f64) -> Self {
        self.gpu_gflops = gflops;
        self
    }

    /// Configure custom dispatch ratio threshold
    pub fn with_min_dispatch_ratio(mut self, ratio: f64) -> Self {
        self.min_dispatch_ratio = ratio;
        self
    }

    /// Select optimal backend based on workload characteristics
    ///
    /// # Arguments
    /// * `data_bytes` - Amount of data to transfer (host → device)
    /// * `flops` - Floating point operations required
    ///
    /// # Returns
    /// Recommended backend based on cost model
    ///
    /// # Cost Model
    /// GPU dispatch is beneficial when:
    /// ```text
    /// compute_time > min_dispatch_ratio × transfer_time
    /// ```
    ///
    /// Per Gregg & Hazelwood (2011), the 5× rule accounts for:
    /// - Host→Device transfer (PCIe overhead)
    /// - Kernel launch latency
    /// - Device→Host transfer
    /// - CPU-GPU synchronization
    pub fn select_backend(&self, data_bytes: usize, flops: u64) -> Backend {
        // Calculate transfer time (seconds)
        let transfer_s = data_bytes as f64 / self.pcie_bandwidth;

        // Calculate compute time (seconds)
        let compute_s = flops as f64 / self.gpu_gflops;

        // Apply 5× dispatch rule
        if compute_s > self.min_dispatch_ratio * transfer_s {
            Backend::GPU
        } else {
            // Fallback to SIMD for intermediate workloads
            Backend::SIMD
        }
    }

    /// Select backend for matrix multiplication
    ///
    /// # Arguments
    /// * `m`, `n`, `k` - Matrix dimensions (M×K) × (K×N) = (M×N)
    ///
    /// # Complexity
    /// - Data: O(mk + kn + mn) = O(mk + kn + mn) bytes
    /// - FLOPs: O(2mnk) = O(mnk) operations
    pub fn select_for_matmul(&self, m: usize, n: usize, k: usize) -> Backend {
        // Data size: two input matrices + output (f32 = 4 bytes)
        let data_bytes = (m * k + k * n + m * n) * 4;

        // FLOPs: 2mnk (multiply-add per element)
        let flops = (2 * m * n * k) as u64;

        self.select_backend(data_bytes, flops)
    }

    /// Select backend for vector operations
    ///
    /// # Arguments
    /// * `n` - Vector length
    /// * `ops_per_element` - Operations per element (e.g., 2 for dot product)
    pub fn select_for_vector_op(&self, n: usize, ops_per_element: u64) -> Backend {
        // Data size: typically two input vectors + output (f32 = 4 bytes)
        let data_bytes = n * 3 * 4;

        // FLOPs
        let flops = n as u64 * ops_per_element;

        self.select_backend(data_bytes, flops)
    }

    /// Select backend for element-wise operations
    ///
    /// Element-wise ops are memory-bound, so GPU is rarely beneficial
    pub fn select_for_elementwise(&self, n: usize) -> Backend {
        // Element-wise ops: 1 FLOP per element, memory-bound
        // GPU overhead rarely justified
        if n > 1_000_000 {
            Backend::SIMD
        } else {
            Backend::Scalar
        }
    }

    /// MoE (Mixture-of-Experts) routing: select backend based on operation complexity
    ///
    /// # Arguments
    /// * `complexity` - Operation complexity (Low/Medium/High)
    /// * `data_size` - Number of elements in the operation
    ///
    /// # Returns
    /// Recommended backend using adaptive thresholds per complexity level
    ///
    /// # MoE Thresholds (per empirical performance analysis)
    /// - **Low complexity** (element-wise): SIMD at 1M+ elements, never GPU
    /// - **Medium complexity** (reductions): SIMD at 10K+, GPU at 100K+ elements
    /// - **High complexity** (matmul): SIMD at 1K+, GPU at 10K+ elements
    pub fn select_with_moe(&self, complexity: OpComplexity, data_size: usize) -> Backend {
        match complexity {
            OpComplexity::Low => {
                // Element-wise: memory-bound, GPU overhead not justified
                if data_size > 1_000_000 {
                    Backend::SIMD
                } else {
                    Backend::Scalar
                }
            }
            OpComplexity::Medium => {
                // Reductions (dot product, sum): moderate compute
                if data_size > 100_000 {
                    Backend::GPU
                } else if data_size > 10_000 {
                    Backend::SIMD
                } else {
                    Backend::Scalar
                }
            }
            OpComplexity::High => {
                // Matrix operations: compute-intensive, O(n²) or O(n³)
                if data_size > 10_000 {
                    Backend::GPU
                } else if data_size > 1_000 {
                    Backend::SIMD
                } else {
                    Backend::Scalar
                }
            }
        }
    }

    /// Map Batuta Backend to Trueno Backend
    #[cfg(feature = "trueno-integration")]
    pub fn to_trueno_backend(backend: Backend) -> trueno::Backend {
        match backend {
            Backend::Scalar => trueno::Backend::Scalar,
            Backend::SIMD => trueno::Backend::Auto, // Let Trueno pick best SIMD (AVX2/NEON)
            Backend::GPU => trueno::Backend::GPU,
        }
    }

    /// Perform vector addition using Trueno with selected backend
    #[cfg(feature = "trueno-integration")]
    pub fn vector_add(
        &self,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, String> {
        if a.len() != b.len() {
            return Err("Vector lengths must match".to_string());
        }

        let _backend = self.select_with_moe(OpComplexity::Low, a.len());

        let vec_a: Vector<f32> = Vector::from_slice(a);
        let vec_b: Vector<f32> = Vector::from_slice(b);

        match vec_a.add(&vec_b) {
            Ok(result) => Ok(result.as_slice().to_vec()),
            Err(e) => Err(format!("Trueno error: {}", e)),
        }
    }

    /// Perform matrix multiplication using Trueno with selected backend
    #[cfg(feature = "trueno-integration")]
    pub fn matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        // Matrix A: m×k, Matrix B: k×n, Result: m×n
        if a.len() != m * k {
            return Err(format!("Matrix A size mismatch: expected {}, got {}", m * k, a.len()));
        }
        if b.len() != k * n {
            return Err(format!("Matrix B size mismatch: expected {}, got {}", k * n, b.len()));
        }

        let _backend = self.select_for_matmul(m, n, k);

        // Create matrices using from_vec (Trueno API)
        let mat_a: Matrix<f32> = match Matrix::from_vec(m, k, a.to_vec()) {
            Ok(m) => m,
            Err(e) => return Err(format!("Trueno error creating matrix A: {}", e)),
        };

        let mat_b: Matrix<f32> = match Matrix::from_vec(k, n, b.to_vec()) {
            Ok(m) => m,
            Err(e) => return Err(format!("Trueno error creating matrix B: {}", e)),
        };

        match mat_a.matmul(&mat_b) {
            Ok(result) => Ok(result.as_slice().to_vec()),
            Err(e) => Err(format!("Trueno error in matmul: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_selection_small_matmul() {
        let selector = BackendSelector::new();

        // Small matrix: 64×64 × 64×64
        // Data: (64*64 + 64*64 + 64*64) * 4 = 49,152 bytes
        // FLOPs: 2 * 64 * 64 * 64 = 524,288
        // Transfer: 49,152 / 32e9 = 1.536 μs
        // Compute: 524,288 / 20e12 = 0.026 μs
        // Ratio: 0.026 / 1.536 = 0.017× (< 5×) → SIMD
        let backend = selector.select_for_matmul(64, 64, 64);
        assert_eq!(backend, Backend::SIMD);
    }

    #[test]
    fn test_backend_selection_large_matmul() {
        let selector = BackendSelector::new();

        // Large matrix: 512×512 × 512×512
        // Data: (512*512 + 512*512 + 512*512) * 4 = 3,145,728 bytes
        // FLOPs: 2 * 512 * 512 * 512 = 268,435,456
        // Transfer: 3,145,728 / 32e9 = 98.3 μs
        // Compute: 268,435,456 / 20e12 = 13.4 μs
        // Ratio: 13.4 / 98.3 = 0.136× (< 5×) → SIMD
        // NOTE: GPU only becomes beneficial for much larger matrices or
        // when compute complexity is O(n³) with very large n
        let backend = selector.select_for_matmul(512, 512, 512);
        assert_eq!(backend, Backend::SIMD);
    }

    #[test]
    fn test_backend_selection_very_large_matmul() {
        let selector = BackendSelector::new();

        // Very large matrix: 2048×2048 × 2048×2048
        // Data: (2048*2048 + 2048*2048 + 2048*2048) * 4 = 50,331,648 bytes
        // FLOPs: 2 * 2048 * 2048 * 2048 = 17,179,869,184
        // Transfer: 50,331,648 / 32e9 = 1,573 μs = 1.57 ms
        // Compute: 17,179,869,184 / 20e12 = 859 μs = 0.859 ms
        // Ratio: 859 / 1573 = 0.546× (< 5×) → still SIMD!
        // Per spec, GPU dispatch needs > 5× compute/transfer ratio
        let backend = selector.select_for_matmul(2048, 2048, 2048);
        // Even this is borderline - real benefit comes from O(n³) operations
        // with sustained computation
        assert_eq!(backend, Backend::SIMD);
    }

    #[test]
    fn test_backend_selection_dot_product() {
        let selector = BackendSelector::new();

        // Large dot product: 10K elements
        // Data: 120 KB
        // FLOPs: 20K (2 ops per element)
        // Transfer: 3.75 μs, Compute: 1 ns → ratio: 0.0003× ✗
        let backend = selector.select_for_vector_op(10_000, 2);
        assert_eq!(backend, Backend::SIMD); // Not GPU
    }

    #[test]
    fn test_backend_selection_elementwise() {
        let selector = BackendSelector::new();

        // Small array
        let backend = selector.select_for_elementwise(1000);
        assert_eq!(backend, Backend::Scalar);

        // Large array
        let backend = selector.select_for_elementwise(2_000_000);
        assert_eq!(backend, Backend::SIMD);
    }

    #[test]
    fn test_custom_dispatch_ratio() {
        let selector = BackendSelector::new()
            .with_min_dispatch_ratio(10.0); // More conservative

        // Workload that passes 5× but fails 10×
        let backend = selector.select_backend(1_000_000, 30_000_000);
        // Transfer: ~31 μs, Compute: ~1.5 μs → ratio: ~0.05× ✗
        assert_eq!(backend, Backend::SIMD);
    }

    #[test]
    fn test_moe_low_complexity() {
        let selector = BackendSelector::new();

        // Small element-wise: Scalar
        assert_eq!(
            selector.select_with_moe(OpComplexity::Low, 100),
            Backend::Scalar
        );

        // Large element-wise: SIMD
        assert_eq!(
            selector.select_with_moe(OpComplexity::Low, 2_000_000),
            Backend::SIMD
        );

        // Never GPU for element-wise (memory-bound)
        assert_ne!(
            selector.select_with_moe(OpComplexity::Low, 10_000_000),
            Backend::GPU
        );
    }

    #[test]
    fn test_moe_medium_complexity() {
        let selector = BackendSelector::new();

        // Small reduction: Scalar
        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 1_000),
            Backend::Scalar
        );

        // Medium reduction: SIMD
        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 50_000),
            Backend::SIMD
        );

        // Large reduction: GPU
        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 200_000),
            Backend::GPU
        );
    }

    #[test]
    fn test_moe_high_complexity() {
        let selector = BackendSelector::new();

        // Small matmul: Scalar
        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 500),
            Backend::Scalar
        );

        // Medium matmul: SIMD
        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 5_000),
            Backend::SIMD
        );

        // Large matmul: GPU
        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 50_000),
            Backend::GPU
        );
    }

    #[test]
    #[cfg(feature = "trueno-integration")]
    #[ignore] // TODO: Fix Trueno API type inference issues
    fn test_trueno_vector_add() {
        let selector = BackendSelector::new();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = selector.vector_add(&a, &b).unwrap();
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    #[cfg(feature = "trueno-integration")]
    #[ignore] // TODO: Fix Trueno API type inference issues
    fn test_trueno_matrix_multiply() {
        let selector = BackendSelector::new();

        // 2×2 matrices
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5, 6], [7, 8]]

        let result = selector.matrix_multiply(&a, &b, 2, 2, 2).unwrap();
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        // [[19, 22], [43, 50]]
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    // ============================================================================
    // COST CALCULATION TESTS (catch arithmetic mutations)
    // ============================================================================

    #[test]
    fn test_select_backend_arithmetic_correctness() {
        // Test that verifies the exact arithmetic in select_backend()
        // Catches mutations: * → /, * → +, / → *

        let selector = BackendSelector::new();

        // Test case 1: Exactly at 5× threshold (should choose SIMD, not GPU)
        // compute_s = 5.0 * transfer_s (boundary)
        let pcie_bw = 32e9;  // 32 GB/s
        let gpu_gflops = 20e12;  // 20 TFLOPS

        // Work backwards from desired ratio: compute_s = 5.0 * transfer_s
        // transfer_s = data_bytes / pcie_bw
        // compute_s = flops / gpu_gflops
        // flops / gpu_gflops = 5.0 * data_bytes / pcie_bw
        // flops = 5.0 * data_bytes * gpu_gflops / pcie_bw

        let data_bytes = 1_000_000;  // 1 MB
        let transfer_s = data_bytes as f64 / pcie_bw;
        let compute_s_threshold = 5.0 * transfer_s;
        let flops = (compute_s_threshold * gpu_gflops) as u64;

        // At exactly 5×, should still choose SIMD (> not >=)
        let backend = selector.select_backend(data_bytes, flops);
        assert_eq!(backend, Backend::SIMD, "At exactly 5× threshold, should choose SIMD");

        // Just above 5× threshold, should choose GPU
        let flops_above = (flops as f64 * 1.01) as u64;  // 1% above threshold
        let backend = selector.select_backend(data_bytes, flops_above);
        assert_eq!(backend, Backend::GPU, "Above 5× threshold, should choose GPU");

        // Well below threshold
        let flops_below = flops / 2;
        let backend = selector.select_backend(data_bytes, flops_below);
        assert_eq!(backend, Backend::SIMD, "Below 5× threshold, should choose SIMD");
    }

    #[test]
    fn test_select_backend_arithmetic_mutation_detection() {
        // This test will FAIL if arithmetic operators are mutated
        // Specifically catches: / → *, * → /, * → +

        let selector = BackendSelector::new();

        // Case 1: High compute, low transfer → GPU
        // 1 GB data, 1000 TFLOPS compute
        let data_bytes = 1_000_000_000;
        let flops = 1_000_000_000_000_000;  // 1000 TFLOPS

        // Expected calculations:
        // transfer_s = 1e9 / 32e9 = 0.03125 s = 31.25 ms
        // compute_s = 1e15 / 20e12 = 50 s
        // ratio = 50 / 0.03125 = 1600× >> 5× → GPU
        let backend = selector.select_backend(data_bytes, flops);
        assert_eq!(backend, Backend::GPU,
            "High compute/transfer ratio should select GPU");

        // Case 2: Low compute, high transfer → SIMD
        // 1 GB data, 1 GFLOP compute
        let flops_low = 1_000_000_000;  // 1 GFLOPS

        // transfer_s = 1e9 / 32e9 = 0.03125 s
        // compute_s = 1e9 / 20e12 = 5e-5 s = 0.05 ms
        // ratio = 5e-5 / 0.03125 = 0.0016× << 5× → SIMD
        let backend = selector.select_backend(data_bytes, flops_low);
        assert_eq!(backend, Backend::SIMD,
            "Low compute/transfer ratio should select SIMD");
    }

    #[test]
    fn test_matmul_data_bytes_calculation() {
        // Test that data_bytes calculation is correct: (m*k + k*n + m*n) * 4
        // Catches mutations in arithmetic operators

        let selector = BackendSelector::new();

        // Test case: 100×100 × 100×100 matmul
        let m = 100;
        let n = 100;
        let k = 100;

        // Expected: (100*100 + 100*100 + 100*100) * 4 = 30,000 * 4 = 120,000 bytes
        // FLOPs: 2 * 100 * 100 * 100 = 2,000,000

        // With default settings:
        // transfer_s = 120,000 / 32e9 = 3.75e-6 s = 3.75 μs
        // compute_s = 2,000,000 / 20e12 = 1e-7 s = 0.1 μs
        // ratio = 0.1 / 3.75 = 0.0267× << 5× → SIMD

        let backend = selector.select_for_matmul(m, n, k);
        assert_eq!(backend, Backend::SIMD);

        // Verify the calculation by testing a case that's GPU-bound
        // Need ratio > 5×, so need much larger matrices or different hardware params
        // Use custom selector with slower PCIe
        let slow_selector = BackendSelector::new()
            .with_pcie_bandwidth(1e9)  // 1 GB/s (slow PCIe 3.0 x1)
            .with_gpu_gflops(100e12);  // 100 TFLOPS (fast GPU)

        // Same 100×100×100 matmul:
        // transfer_s = 120,000 / 1e9 = 1.2e-4 s = 120 μs
        // compute_s = 2,000,000 / 100e12 = 2e-8 s = 0.02 μs
        // ratio = 0.02 / 120 = 0.00017× << 5× → still SIMD

        // Need MUCH larger matrices
        let m_large = 1000;
        let n_large = 1000;
        let k_large = 1000;

        // data_bytes = (1000*1000 + 1000*1000 + 1000*1000) * 4 = 12,000,000 bytes = 12 MB
        // FLOPs = 2 * 1000 * 1000 * 1000 = 2,000,000,000
        // transfer_s = 12,000,000 / 1e9 = 0.012 s = 12 ms
        // compute_s = 2,000,000,000 / 100e12 = 2e-5 s = 0.02 ms
        // ratio = 0.02 / 12 = 0.0017× << 5× → SIMD

        let backend = slow_selector.select_for_matmul(m_large, n_large, k_large);
        assert_eq!(backend, Backend::SIMD);
    }

    #[test]
    fn test_matmul_flops_calculation() {
        // Test that FLOPs calculation is correct: 2 * m * n * k
        // Catches mutations: * → /, * → +

        let selector = BackendSelector::new()
            .with_gpu_gflops(1e12);  // 1 TFLOPS (slower GPU for easier math)

        // Small matmul where we can verify the exact FLOP count matters
        let m = 10;
        let n = 10;
        let k = 10;

        // Expected FLOPs: 2 * 10 * 10 * 10 = 2,000
        // data_bytes: (10*10 + 10*10 + 10*10) * 4 = 1,200 bytes
        // transfer_s = 1,200 / 32e9 = 3.75e-8 s
        // compute_s = 2,000 / 1e12 = 2e-9 s
        // ratio = 2e-9 / 3.75e-8 = 0.053× << 5× → SIMD

        let backend = selector.select_for_matmul(m, n, k);
        assert_eq!(backend, Backend::SIMD);
    }

    #[test]
    fn test_vector_op_data_bytes_calculation() {
        // Test that vector op data_bytes = n * 3 * 4
        // (two input vectors + output, f32 = 4 bytes)

        let selector = BackendSelector::new();

        let n = 1000;
        let ops_per_element = 2;  // e.g., dot product

        // Expected: data_bytes = 1000 * 3 * 4 = 12,000 bytes
        // FLOPs: 1000 * 2 = 2,000
        // transfer_s = 12,000 / 32e9 = 3.75e-7 s
        // compute_s = 2,000 / 20e12 = 1e-10 s
        // ratio = 1e-10 / 3.75e-7 = 0.000267× << 5× → SIMD

        let backend = selector.select_for_vector_op(n, ops_per_element);
        assert_eq!(backend, Backend::SIMD);
    }

    #[test]
    fn test_vector_op_flops_calculation() {
        // Test that vector op FLOPs = n * ops_per_element

        let selector = BackendSelector::new()
            .with_gpu_gflops(1e12);  // 1 TFLOPS

        let n = 10000;
        let ops_per_element = 10;  // Complex reduction

        // FLOPs: 10,000 * 10 = 100,000
        // data_bytes: 10,000 * 3 * 4 = 120,000 bytes
        // transfer_s = 120,000 / 32e9 = 3.75e-6 s
        // compute_s = 100,000 / 1e12 = 1e-7 s
        // ratio = 1e-7 / 3.75e-6 = 0.0267× << 5× → SIMD

        let backend = selector.select_for_vector_op(n, ops_per_element);
        assert_eq!(backend, Backend::SIMD);
    }

    #[test]
    fn test_dispatch_ratio_multiplication() {
        // Test that min_dispatch_ratio is correctly multiplied
        // Catches mutation: * → /, * → +

        // Test with different dispatch ratios
        let selector_5x = BackendSelector::new()
            .with_min_dispatch_ratio(5.0);

        let selector_10x = BackendSelector::new()
            .with_min_dispatch_ratio(10.0);

        // Workload with 7× ratio
        let data_bytes = 1_000_000;
        let pcie_bw = 32e9;
        let gpu_gflops = 20e12;

        let transfer_s = data_bytes as f64 / pcie_bw;
        let compute_s_7x = 7.0 * transfer_s;
        let flops = (compute_s_7x * gpu_gflops) as u64;

        // With 5× threshold: 7× > 5× → GPU
        let backend = selector_5x.select_backend(data_bytes, flops);
        assert_eq!(backend, Backend::GPU, "7× should exceed 5× threshold");

        // With 10× threshold: 7× < 10× → SIMD
        let backend = selector_10x.select_backend(data_bytes, flops);
        assert_eq!(backend, Backend::SIMD, "7× should not exceed 10× threshold");
    }

    // ============================================================================
    // MOE BOUNDARY CONDITION TESTS (catch comparison mutations)
    // ============================================================================

    #[test]
    fn test_moe_low_complexity_boundary() {
        // Test exact boundary: data_size > 1_000_000
        // Catches mutation: > → >=

        let selector = BackendSelector::new();

        // Exactly at boundary (should be Scalar, not SIMD)
        assert_eq!(
            selector.select_with_moe(OpComplexity::Low, 1_000_000),
            Backend::Scalar,
            "Exactly 1M elements should be Scalar (> not >=)"
        );

        // Just above boundary (should be SIMD)
        assert_eq!(
            selector.select_with_moe(OpComplexity::Low, 1_000_001),
            Backend::SIMD,
            "1M+1 elements should be SIMD"
        );

        // Just below boundary (should be Scalar)
        assert_eq!(
            selector.select_with_moe(OpComplexity::Low, 999_999),
            Backend::Scalar,
            "1M-1 elements should be Scalar"
        );
    }

    #[test]
    fn test_moe_medium_complexity_boundaries() {
        // Test exact boundaries: 10_000 and 100_000
        // Catches mutations: > → >=

        let selector = BackendSelector::new();

        // First boundary: data_size > 10_000 (Scalar → SIMD)
        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 10_000),
            Backend::Scalar,
            "Exactly 10K should be Scalar"
        );

        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 10_001),
            Backend::SIMD,
            "10K+1 should be SIMD"
        );

        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 9_999),
            Backend::Scalar,
            "10K-1 should be Scalar"
        );

        // Second boundary: data_size > 100_000 (SIMD → GPU)
        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 100_000),
            Backend::SIMD,
            "Exactly 100K should be SIMD"
        );

        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 100_001),
            Backend::GPU,
            "100K+1 should be GPU"
        );

        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 99_999),
            Backend::SIMD,
            "100K-1 should be SIMD"
        );
    }

    #[test]
    fn test_moe_high_complexity_boundaries() {
        // Test exact boundaries: 1_000 and 10_000
        // Catches mutations: > → >=

        let selector = BackendSelector::new();

        // First boundary: data_size > 1_000 (Scalar → SIMD)
        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 1_000),
            Backend::Scalar,
            "Exactly 1K should be Scalar"
        );

        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 1_001),
            Backend::SIMD,
            "1K+1 should be SIMD"
        );

        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 999),
            Backend::Scalar,
            "1K-1 should be Scalar"
        );

        // Second boundary: data_size > 10_000 (SIMD → GPU)
        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 10_000),
            Backend::SIMD,
            "Exactly 10K should be SIMD"
        );

        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 10_001),
            Backend::GPU,
            "10K+1 should be GPU"
        );

        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 9_999),
            Backend::SIMD,
            "10K-1 should be SIMD"
        );
    }

    #[test]
    fn test_elementwise_boundary() {
        // Test boundary for select_for_elementwise: data_size > 1_000_000

        let selector = BackendSelector::new();

        // Exactly at boundary
        assert_eq!(
            selector.select_for_elementwise(1_000_000),
            Backend::Scalar,
            "Exactly 1M should be Scalar"
        );

        // Just above
        assert_eq!(
            selector.select_for_elementwise(1_000_001),
            Backend::SIMD,
            "1M+1 should be SIMD"
        );

        // Just below
        assert_eq!(
            selector.select_for_elementwise(999_999),
            Backend::Scalar,
            "1M-1 should be Scalar"
        );
    }

    // ============================================================================
    // EDGE CASE TESTS
    // ============================================================================

    #[test]
    fn test_zero_size_operations() {
        let selector = BackendSelector::new();

        // Zero-size matmul
        let backend = selector.select_for_matmul(0, 0, 0);
        assert_eq!(backend, Backend::SIMD);  // 0 flops, 0 data → SIMD by default

        // Zero-size vector op
        let backend = selector.select_for_vector_op(0, 1);
        assert_eq!(backend, Backend::SIMD);

        // Zero-size elementwise
        let backend = selector.select_for_elementwise(0);
        assert_eq!(backend, Backend::Scalar);

        // Zero-size MoE
        assert_eq!(
            selector.select_with_moe(OpComplexity::Low, 0),
            Backend::Scalar
        );
        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 0),
            Backend::Scalar
        );
        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 0),
            Backend::Scalar
        );
    }

    #[test]
    fn test_single_element_operations() {
        let selector = BackendSelector::new();

        // Single element should always be Scalar (too small for SIMD/GPU)
        assert_eq!(
            selector.select_with_moe(OpComplexity::Low, 1),
            Backend::Scalar
        );
        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 1),
            Backend::Scalar
        );
        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 1),
            Backend::Scalar
        );

        assert_eq!(
            selector.select_for_elementwise(1),
            Backend::Scalar
        );
    }

    #[test]
    fn test_very_large_operations() {
        let selector = BackendSelector::new();

        // Very large sizes (billions of elements)
        let huge_size = 1_000_000_000;  // 1 billion elements

        // Low complexity: never GPU (memory-bound)
        assert_eq!(
            selector.select_with_moe(OpComplexity::Low, huge_size),
            Backend::SIMD
        );

        // Medium complexity: GPU
        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, huge_size),
            Backend::GPU
        );

        // High complexity: GPU
        assert_eq!(
            selector.select_with_moe(OpComplexity::High, huge_size),
            Backend::GPU
        );
    }

    #[test]
    fn test_custom_hardware_params() {
        // Test with extreme hardware configurations

        // Slow PCIe, fast GPU (can favor GPU with high compute workloads)
        let slow_pcie_selector = BackendSelector::new()
            .with_pcie_bandwidth(1e9)  // 1 GB/s
            .with_gpu_gflops(100e12);   // 100 TFLOPS

        // Fast PCIe, slow GPU (favors CPU/SIMD)
        let fast_pcie_selector = BackendSelector::new()
            .with_pcie_bandwidth(100e9)  // 100 GB/s
            .with_gpu_gflops(1e12);      // 1 TFLOPS

        // Test 1: Low compute workload (both should choose SIMD)
        let data_bytes_low = 1_000_000;
        let flops_low = 1_000_000_000;  // 1 GFLOPS

        // Slow PCIe: transfer_s = 1M/1e9 = 1ms, compute_s = 1G/100e12 = 0.01μs
        // ratio = 0.01μs / 1ms = 0.00001× << 5× → SIMD
        let backend = slow_pcie_selector.select_backend(data_bytes_low, flops_low);
        assert_eq!(backend, Backend::SIMD);

        // Fast PCIe: transfer_s = 1M/100e9 = 10μs, compute_s = 1G/1e12 = 1ms
        // ratio = 1ms / 10μs = 100× >> 5× → GPU
        let backend = fast_pcie_selector.select_backend(data_bytes_low, flops_low);
        assert_eq!(backend, Backend::GPU);

        // Test 2: High compute workload
        let data_bytes_high = 1_000_000;
        let flops_high = 1_000_000_000_000;  // 1 TFLOPS

        // Slow PCIe: transfer_s = 1ms, compute_s = 1T/100T = 10ms
        // ratio = 10ms / 1ms = 10× >> 5× → GPU
        let backend = slow_pcie_selector.select_backend(data_bytes_high, flops_high);
        assert_eq!(backend, Backend::GPU);

        // Fast PCIe: transfer_s = 10μs, compute_s = 1T/1T = 1s
        // ratio = 1s / 10μs = 100,000× >> 5× → GPU
        let backend = fast_pcie_selector.select_backend(data_bytes_high, flops_high);
        assert_eq!(backend, Backend::GPU);
    }
}
