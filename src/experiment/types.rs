//! Core type definitions for experiment tracking.
//!
//! This module contains the fundamental types for compute devices, architectures,
//! and model paradigms used throughout the experiment tracking system.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during experiment tracking operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ExperimentError {
    #[error("Invalid compute device configuration: {0}")]
    InvalidComputeDevice(String),

    #[error("Metrics collection failed: {0}")]
    MetricsCollectionFailed(String),

    #[error("Pareto frontier calculation failed: {0}")]
    ParetoFrontierFailed(String),

    #[error("Invalid ORCID format: {0}")]
    InvalidOrcid(String),

    #[error("Citation generation failed: {0}")]
    CitationGenerationFailed(String),

    #[error("Sovereign distribution validation failed: {0}")]
    SovereignValidationFailed(String),

    #[error("Experiment storage error: {0}")]
    StorageError(String),
}

/// Compute device abstraction for heterogeneous hardware
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComputeDevice {
    /// Standard CPU execution
    Cpu {
        cores: u32,
        threads_per_core: u32,
        architecture: CpuArchitecture,
    },
    /// NVIDIA/AMD GPU acceleration
    Gpu {
        name: String,
        memory_gb: f32,
        compute_capability: Option<String>,
        vendor: GpuVendor,
    },
    /// Google TPU accelerator
    Tpu { version: TpuVersion, cores: u32 },
    /// Apple Silicon unified memory
    AppleSilicon {
        chip: AppleChip,
        neural_engine_cores: u32,
        gpu_cores: u32,
        memory_gb: u32,
    },
    /// Edge/embedded devices
    Edge {
        name: String,
        power_budget_watts: f32,
    },
}

/// CPU architecture variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuArchitecture {
    X86_64,
    Aarch64,
    Riscv64,
    Wasm32,
}

/// GPU vendor identification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
}

/// TPU version variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpuVersion {
    V2,
    V3,
    V4,
    V5e,
    V5p,
}

/// Apple Silicon chip variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AppleChip {
    M1,
    M1Pro,
    M1Max,
    M1Ultra,
    M2,
    M2Pro,
    M2Max,
    M2Ultra,
    M3,
    M3Pro,
    M3Max,
    M4,
    M4Pro,
    M4Max,
}

impl ComputeDevice {
    /// Calculate theoretical FLOPS for the device
    pub fn theoretical_flops(&self) -> f64 {
        match self {
            ComputeDevice::Cpu {
                cores,
                threads_per_core,
                architecture,
            } => {
                let base_flops = match architecture {
                    CpuArchitecture::X86_64 => 32.0,  // AVX2: 8 FP32 * 4 ops
                    CpuArchitecture::Aarch64 => 16.0, // NEON: 4 FP32 * 4 ops
                    CpuArchitecture::Riscv64 => 8.0,
                    CpuArchitecture::Wasm32 => 4.0,
                };
                (*cores as f64) * (*threads_per_core as f64) * base_flops * 1e9
            }
            ComputeDevice::Gpu {
                memory_gb, vendor, ..
            } => {
                // Rough estimate based on memory bandwidth
                let bandwidth_factor = match vendor {
                    GpuVendor::Nvidia => 15.0,
                    GpuVendor::Amd => 12.0,
                    GpuVendor::Intel => 8.0,
                    GpuVendor::Apple => 10.0,
                };
                (*memory_gb as f64) * bandwidth_factor * 1e12
            }
            ComputeDevice::Tpu { version, cores } => {
                let flops_per_core = match version {
                    TpuVersion::V2 => 45e12,
                    TpuVersion::V3 => 90e12,
                    TpuVersion::V4 => 275e12,
                    TpuVersion::V5e => 197e12,
                    TpuVersion::V5p => 459e12,
                };
                (*cores as f64) * flops_per_core
            }
            ComputeDevice::AppleSilicon {
                chip, gpu_cores, ..
            } => {
                let flops_per_gpu_core = match chip {
                    AppleChip::M1 | AppleChip::M1Pro | AppleChip::M1Max | AppleChip::M1Ultra => {
                        128e9
                    }
                    AppleChip::M2 | AppleChip::M2Pro | AppleChip::M2Max | AppleChip::M2Ultra => {
                        150e9
                    }
                    AppleChip::M3 | AppleChip::M3Pro | AppleChip::M3Max => 180e9,
                    AppleChip::M4 | AppleChip::M4Pro | AppleChip::M4Max => 200e9,
                };
                (*gpu_cores as f64) * flops_per_gpu_core
            }
            ComputeDevice::Edge {
                power_budget_watts, ..
            } => {
                // Assume ~10 GFLOPS per watt for edge devices
                (*power_budget_watts as f64) * 10e9
            }
        }
    }

    /// Estimate power consumption in watts
    pub fn estimated_power_watts(&self) -> f32 {
        match self {
            ComputeDevice::Cpu { cores, .. } => (*cores as f32) * 15.0,
            ComputeDevice::Gpu {
                memory_gb, vendor, ..
            } => {
                let base = match vendor {
                    GpuVendor::Nvidia => 30.0,
                    GpuVendor::Amd => 35.0,
                    GpuVendor::Intel => 25.0,
                    GpuVendor::Apple => 20.0,
                };
                *memory_gb * base
            }
            ComputeDevice::Tpu { version, cores } => {
                let per_core = match version {
                    TpuVersion::V2 => 40.0,
                    TpuVersion::V3 => 50.0,
                    TpuVersion::V4 => 60.0,
                    TpuVersion::V5e => 45.0,
                    TpuVersion::V5p => 70.0,
                };
                (*cores as f32) * per_core
            }
            ComputeDevice::AppleSilicon { chip, .. } => match chip {
                AppleChip::M1 => 20.0,
                AppleChip::M1Pro => 30.0,
                AppleChip::M1Max => 40.0,
                AppleChip::M1Ultra => 60.0,
                AppleChip::M2 => 22.0,
                AppleChip::M2Pro => 32.0,
                AppleChip::M2Max => 45.0,
                AppleChip::M2Ultra => 65.0,
                AppleChip::M3 => 24.0,
                AppleChip::M3Pro => 35.0,
                AppleChip::M3Max => 50.0,
                AppleChip::M4 => 25.0,
                AppleChip::M4Pro => 38.0,
                AppleChip::M4Max => 55.0,
            },
            ComputeDevice::Edge {
                power_budget_watts, ..
            } => *power_budget_watts,
        }
    }
}

/// Model training paradigm classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelParadigm {
    /// Traditional ML (sklearn-style)
    TraditionalML,
    /// Deep Learning from scratch
    DeepLearning,
    /// Fine-tuning pretrained models
    FineTuning,
    /// Knowledge distillation
    Distillation,
    /// Mixture of Experts
    MoE,
    /// Reinforcement Learning
    ReinforcementLearning,
    /// Federated Learning
    FederatedLearning,
    /// Neural Architecture Search
    Nas,
    /// Continual/Lifelong Learning
    ContinualLearning,
    /// Meta-Learning
    MetaLearning,
}

impl ModelParadigm {
    /// Get typical compute intensity for this paradigm
    pub fn compute_intensity(&self) -> ComputeIntensity {
        match self {
            ModelParadigm::TraditionalML => ComputeIntensity::Low,
            ModelParadigm::DeepLearning => ComputeIntensity::High,
            ModelParadigm::FineTuning => ComputeIntensity::Medium,
            ModelParadigm::Distillation => ComputeIntensity::Medium,
            ModelParadigm::MoE => ComputeIntensity::VeryHigh,
            ModelParadigm::ReinforcementLearning => ComputeIntensity::High,
            ModelParadigm::FederatedLearning => ComputeIntensity::Medium,
            ModelParadigm::Nas => ComputeIntensity::VeryHigh,
            ModelParadigm::ContinualLearning => ComputeIntensity::Medium,
            ModelParadigm::MetaLearning => ComputeIntensity::High,
        }
    }

    /// Check if paradigm typically benefits from GPU
    pub fn benefits_from_gpu(&self) -> bool {
        matches!(
            self,
            ModelParadigm::DeepLearning
                | ModelParadigm::FineTuning
                | ModelParadigm::MoE
                | ModelParadigm::ReinforcementLearning
                | ModelParadigm::Nas
        )
    }
}

/// Compute intensity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeIntensity {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Platform efficiency classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlatformEfficiency {
    /// Server-grade hardware (datacenter)
    Server,
    /// Desktop workstation
    Workstation,
    /// Laptop
    Laptop,
    /// Edge device
    Edge,
    /// Mobile device
    Mobile,
    /// Embedded system
    Embedded,
}

impl PlatformEfficiency {
    /// Get typical power budget for platform
    pub fn typical_power_budget_watts(&self) -> f32 {
        match self {
            PlatformEfficiency::Server => 500.0,
            PlatformEfficiency::Workstation => 350.0,
            PlatformEfficiency::Laptop => 65.0,
            PlatformEfficiency::Edge => 15.0,
            PlatformEfficiency::Mobile => 5.0,
            PlatformEfficiency::Embedded => 1.0,
        }
    }
}
