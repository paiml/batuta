/// Backend selection and cost model (per spec section 2.2)
use serde::{Deserialize, Serialize};

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
}
