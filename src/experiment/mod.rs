#![allow(dead_code)]
//! Experiment Tracking Integration Module
//!
//! Integrates with Entrenar Experiment Tracking Spec v1.8.0 for orchestrating
//! ML experiment workflows with full traceability, cost optimization, and
//! academic research support.
//!
//! # Entrenar CLI (v0.2.4)
//!
//! The entrenar crate provides a comprehensive CLI:
//!
//! ```bash
//! # Training (YAML Mode v1.0)
//! entrenar train config.yaml           # Train from declarative YAML
//! entrenar validate config.yaml        # Validate configuration
//! entrenar init --template lora        # Generate config template
//!
//! # Model Operations
//! entrenar quantize model.safetensors --bits 4
//! entrenar merge model1.st model2.st --method ties
//!
//! # Research Workflows
//! entrenar research init --id my-dataset
//! entrenar research cite artifact.yaml --format bibtex
//!
//! # Inspection & Auditing
//! entrenar inspect model.safetensors   # Model/data inspection
//! entrenar audit data.parquet --type bias
//! entrenar monitor data.parquet        # Drift detection
//!
//! # Benchmarking (entrenar-bench)
//! entrenar-bench temperature --start 1.0 --end 8.0
//! entrenar-bench cost-performance --gpu a100-80gb
//! ```
//!
//! # MCP Tooling (pmcp v1.8.6 + pforge v0.1.2)
//!
//! The stack includes Model Context Protocol (MCP) infrastructure:
//!
//! ```bash
//! # pmcp - Rust SDK for MCP servers/clients
//! # Build MCP servers with full TypeScript SDK compatibility
//!
//! # pforge - Declarative MCP framework
//! pforge new my-server              # Create new MCP server project
//! pforge serve                       # Run MCP server
//!
//! # Define tools in YAML (pforge.yaml):
//! # tools:
//! #   - type: native
//! #     name: train_model
//! #     handler: { path: handlers::train }
//! #     params:
//! #       config: { type: string, required: true }
//! ```
//!
//! **Handler Types:**
//! - `native` - Rust functions with full type safety
//! - `cli` - Execute shell commands
//! - `http` - Proxy HTTP endpoints
//! - `pipeline` - Chain multiple tools together
//!
//! # Features
//! - ComputeDevice abstraction (CPU/GPU/TPU/AppleSilicon)
//! - EnergyMetrics and CostMetrics for efficiency tracking
//! - ModelParadigm classification
//! - CostPerformanceBenchmark with Pareto frontier analysis
//! - SovereignDistribution for air-gapped deployments
//! - ResearchArtifact with ORCID/CRediT academic support
//! - CitationMetadata for BibTeX/CFF generation
//! - Experiment tree visualization for run comparison (MLflow replacement)
//! - YAML Mode Training v1.0 declarative configuration

pub mod tree;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

/// Energy consumption metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyMetrics {
    /// Total energy consumed in joules
    pub total_joules: f64,
    /// Average power draw in watts
    pub average_power_watts: f64,
    /// Peak power draw in watts
    pub peak_power_watts: f64,
    /// Duration of measurement in seconds
    pub duration_seconds: f64,
    /// CO2 equivalent emissions in grams (based on grid carbon intensity)
    pub co2_grams: Option<f64>,
    /// Power Usage Effectiveness (datacenter overhead)
    pub pue: f64,
}

impl EnergyMetrics {
    /// Create new energy metrics
    pub fn new(
        total_joules: f64,
        average_power_watts: f64,
        peak_power_watts: f64,
        duration_seconds: f64,
    ) -> Self {
        Self {
            total_joules,
            average_power_watts,
            peak_power_watts,
            duration_seconds,
            co2_grams: None,
            pue: 1.0,
        }
    }

    /// Calculate CO2 emissions based on carbon intensity (g CO2/kWh)
    pub fn with_carbon_intensity(mut self, carbon_intensity_g_per_kwh: f64) -> Self {
        let kwh = self.total_joules / 3_600_000.0;
        self.co2_grams = Some(kwh * carbon_intensity_g_per_kwh * self.pue);
        self
    }

    /// Set the Power Usage Effectiveness factor
    pub fn with_pue(mut self, pue: f64) -> Self {
        let old_pue = self.pue;
        self.pue = pue;
        // Recalculate CO2 if already set (scale by new PUE / old PUE)
        if let Some(co2) = self.co2_grams {
            self.co2_grams = Some(co2 / old_pue * pue);
        }
        self
    }

    /// Calculate energy efficiency in FLOPS per watt
    pub fn flops_per_watt(&self, total_flops: f64) -> f64 {
        if self.average_power_watts > 0.0 {
            total_flops / self.average_power_watts
        } else {
            0.0
        }
    }
}

/// Cost metrics for experiments
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostMetrics {
    /// Compute cost in USD
    pub compute_cost_usd: f64,
    /// Storage cost in USD
    pub storage_cost_usd: f64,
    /// Network transfer cost in USD
    pub network_cost_usd: f64,
    /// Total cost in USD
    pub total_cost_usd: f64,
    /// Cost per FLOP in USD
    pub cost_per_flop: Option<f64>,
    /// Cost per sample processed
    pub cost_per_sample: Option<f64>,
    /// Currency (default USD)
    pub currency: String,
}

impl CostMetrics {
    /// Create new cost metrics
    pub fn new(compute_cost: f64, storage_cost: f64, network_cost: f64) -> Self {
        Self {
            compute_cost_usd: compute_cost,
            storage_cost_usd: storage_cost,
            network_cost_usd: network_cost,
            total_cost_usd: compute_cost + storage_cost + network_cost,
            cost_per_flop: None,
            cost_per_sample: None,
            currency: "USD".to_string(),
        }
    }

    /// Add FLOP-based cost calculation
    pub fn with_flops(mut self, total_flops: f64) -> Self {
        if total_flops > 0.0 {
            self.cost_per_flop = Some(self.total_cost_usd / total_flops);
        }
        self
    }

    /// Add sample-based cost calculation
    pub fn with_samples(mut self, total_samples: u64) -> Self {
        if total_samples > 0 {
            self.cost_per_sample = Some(self.total_cost_usd / total_samples as f64);
        }
        self
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

/// A point in the cost-performance space
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostPerformancePoint {
    /// Unique identifier for this configuration
    pub id: String,
    /// Performance metric (e.g., accuracy, F1, throughput)
    pub performance: f64,
    /// Cost in USD
    pub cost: f64,
    /// Energy consumption in joules
    pub energy_joules: f64,
    /// Latency in milliseconds (for inference)
    pub latency_ms: Option<f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Cost-performance benchmark with Pareto frontier analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostPerformanceBenchmark {
    /// Name of the benchmark
    pub name: String,
    /// All data points
    pub points: Vec<CostPerformancePoint>,
    /// Pareto-optimal points (computed lazily)
    pareto_frontier: Option<Vec<usize>>,
}

impl CostPerformanceBenchmark {
    /// Create a new benchmark
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            points: Vec::new(),
            pareto_frontier: None,
        }
    }

    /// Add a data point
    pub fn add_point(&mut self, point: CostPerformancePoint) {
        self.points.push(point);
        self.pareto_frontier = None; // Invalidate cache
    }

    /// Compute the Pareto frontier (maximize performance, minimize cost)
    pub fn compute_pareto_frontier(&mut self) -> &[usize] {
        if self.pareto_frontier.is_some() {
            return self.pareto_frontier.as_ref().expect("checked is_some above");
        }

        let mut frontier = Vec::new();

        for (i, point) in self.points.iter().enumerate() {
            let mut is_dominated = false;

            for (j, other) in self.points.iter().enumerate() {
                if i == j {
                    continue;
                }

                // Other dominates point if: better or equal on all, strictly better on at least one
                let other_better_perf = other.performance >= point.performance;
                let other_better_cost = other.cost <= point.cost;
                let other_strictly_better =
                    other.performance > point.performance || other.cost < point.cost;

                if other_better_perf && other_better_cost && other_strictly_better {
                    is_dominated = true;
                    break;
                }
            }

            if !is_dominated {
                frontier.push(i);
            }
        }

        // Sort by performance descending
        frontier.sort_by(|&a, &b| {
            self.points[b]
                .performance
                .partial_cmp(&self.points[a].performance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.pareto_frontier = Some(frontier);
        self.pareto_frontier.as_ref().expect("just assigned Some above")
    }

    /// Get Pareto-optimal points
    pub fn pareto_optimal_points(&mut self) -> Vec<&CostPerformancePoint> {
        let frontier = self.compute_pareto_frontier().to_vec();
        frontier.iter().map(|&i| &self.points[i]).collect()
    }

    /// Find the best point within a cost budget
    pub fn best_within_budget(&mut self, max_cost: f64) -> Option<&CostPerformancePoint> {
        self.compute_pareto_frontier();

        self.pareto_frontier
            .as_ref()
            .expect("compute_pareto_frontier ensures Some")
            .iter()
            .map(|&i| &self.points[i])
            .filter(|p| p.cost <= max_cost)
            .max_by(|a, b| {
                a.performance
                    .partial_cmp(&b.performance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Calculate cost-performance efficiency (performance per dollar)
    pub fn efficiency_scores(&self) -> Vec<(usize, f64)> {
        self.points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let efficiency = if p.cost > 0.0 {
                    p.performance / p.cost
                } else {
                    f64::INFINITY
                };
                (i, efficiency)
            })
            .collect()
    }
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

/// Sovereign distribution manifest for air-gapped deployments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignDistribution {
    /// Distribution name
    pub name: String,
    /// Version
    pub version: String,
    /// Target platforms
    pub platforms: Vec<String>,
    /// Required artifacts
    pub artifacts: Vec<SovereignArtifact>,
    /// Cryptographic signatures
    pub signatures: Vec<ArtifactSignature>,
    /// Offline registry configuration
    pub offline_registry: Option<OfflineRegistryConfig>,
    /// Nix flake hash for reproducibility
    pub nix_flake_hash: Option<String>,
}

/// Artifact in a sovereign distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignArtifact {
    /// Artifact name
    pub name: String,
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// SHA-256 hash
    pub sha256: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Download URL (for pre-staging)
    pub source_url: Option<String>,
}

/// Types of distributable artifacts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactType {
    Binary,
    Model,
    Dataset,
    Config,
    Documentation,
    Container,
    NixDerivation,
}

/// Cryptographic signature for artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactSignature {
    /// Artifact name this signature is for
    pub artifact_name: String,
    /// Signature algorithm
    pub algorithm: SignatureAlgorithm,
    /// Base64-encoded signature
    pub signature: String,
    /// Public key identifier
    pub key_id: String,
}

/// Supported signature algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignatureAlgorithm {
    Ed25519,
    RSA4096,
    EcdsaP256,
}

/// Offline model registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineRegistryConfig {
    /// Registry path
    pub path: String,
    /// Index file location
    pub index_path: String,
    /// Supported platforms
    pub platforms: Vec<String>,
    /// Last sync timestamp
    pub last_sync: Option<String>,
}

impl SovereignDistribution {
    /// Create a new sovereign distribution
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            platforms: Vec::new(),
            artifacts: Vec::new(),
            signatures: Vec::new(),
            offline_registry: None,
            nix_flake_hash: None,
        }
    }

    /// Add a platform target
    pub fn add_platform(&mut self, platform: impl Into<String>) {
        self.platforms.push(platform.into());
    }

    /// Add an artifact
    pub fn add_artifact(&mut self, artifact: SovereignArtifact) {
        self.artifacts.push(artifact);
    }

    /// Validate all artifacts have signatures
    pub fn validate_signatures(&self) -> Result<(), ExperimentError> {
        for artifact in &self.artifacts {
            let has_sig = self
                .signatures
                .iter()
                .any(|s| s.artifact_name == artifact.name);
            if !has_sig {
                return Err(ExperimentError::SovereignValidationFailed(format!(
                    "Missing signature for artifact: {}",
                    artifact.name
                )));
            }
        }
        Ok(())
    }

    /// Calculate total distribution size
    pub fn total_size_bytes(&self) -> u64 {
        self.artifacts.iter().map(|a| a.size_bytes).sum()
    }
}

/// ORCID identifier (Open Researcher and Contributor ID)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Orcid(String);

impl Orcid {
    /// Create and validate an ORCID
    pub fn new(orcid: impl Into<String>) -> Result<Self, ExperimentError> {
        let orcid = orcid.into();
        // ORCID format: 0000-0000-0000-000X where X can be 0-9 or X
        let re = regex_lite::Regex::new(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
            .expect("static regex pattern is valid");
        if re.is_match(&orcid) {
            Ok(Self(orcid))
        } else {
            Err(ExperimentError::InvalidOrcid(orcid))
        }
    }

    /// Get the ORCID string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// CRediT (Contributor Roles Taxonomy) roles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CreditRole {
    Conceptualization,
    DataCuration,
    FormalAnalysis,
    FundingAcquisition,
    Investigation,
    Methodology,
    ProjectAdministration,
    Resources,
    Software,
    Supervision,
    Validation,
    Visualization,
    WritingOriginalDraft,
    WritingReviewEditing,
}

/// Research contributor with ORCID and CRediT roles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchContributor {
    /// Full name
    pub name: String,
    /// ORCID identifier
    pub orcid: Option<Orcid>,
    /// Affiliation
    pub affiliation: String,
    /// CRediT roles
    pub roles: Vec<CreditRole>,
    /// Email (optional)
    pub email: Option<String>,
}

/// Research artifact with full academic metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchArtifact {
    /// Title
    pub title: String,
    /// Abstract
    pub abstract_text: String,
    /// Contributors with roles
    pub contributors: Vec<ResearchContributor>,
    /// Keywords
    pub keywords: Vec<String>,
    /// DOI if published
    pub doi: Option<String>,
    /// ArXiv ID if applicable
    pub arxiv_id: Option<String>,
    /// License
    pub license: String,
    /// Creation date
    pub created_at: String,
    /// Associated datasets
    pub datasets: Vec<String>,
    /// Associated code repositories
    pub code_repositories: Vec<String>,
    /// Pre-registration info
    pub pre_registration: Option<PreRegistration>,
}

/// Pre-registration for reproducible research
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreRegistration {
    /// Registration timestamp
    pub timestamp: String,
    /// Ed25519 signature of the registration
    pub signature: String,
    /// Public key used for signing
    pub public_key: String,
    /// Hash of the pre-registered hypotheses
    pub hypotheses_hash: String,
    /// Registry where registered (e.g., OSF, AsPredicted)
    pub registry: String,
    /// Registration ID
    pub registration_id: String,
}

/// Citation metadata for BibTeX/CFF generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationMetadata {
    /// Citation type
    pub citation_type: CitationType,
    /// Title
    pub title: String,
    /// Authors
    pub authors: Vec<String>,
    /// Year
    pub year: u16,
    /// Month (optional)
    pub month: Option<u8>,
    /// DOI
    pub doi: Option<String>,
    /// URL
    pub url: Option<String>,
    /// Journal/Conference name
    pub venue: Option<String>,
    /// Volume
    pub volume: Option<String>,
    /// Pages
    pub pages: Option<String>,
    /// Publisher
    pub publisher: Option<String>,
    /// Version (for software)
    pub version: Option<String>,
}

/// Citation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CitationType {
    Article,
    InProceedings,
    Book,
    Software,
    Dataset,
    Misc,
}

impl CitationMetadata {
    /// Generate BibTeX entry
    pub fn to_bibtex(&self, key: &str) -> String {
        let type_str = match self.citation_type {
            CitationType::Article => "article",
            CitationType::InProceedings => "inproceedings",
            CitationType::Book => "book",
            CitationType::Software => "software",
            CitationType::Dataset => "dataset",
            CitationType::Misc => "misc",
        };

        let mut bibtex = format!("@{}{{{},\n", type_str, key);
        bibtex.push_str(&format!("  title = {{{}}},\n", self.title));
        bibtex.push_str(&format!("  author = {{{}}},\n", self.authors.join(" and ")));
        bibtex.push_str(&format!("  year = {{{}}},\n", self.year));

        if let Some(month) = self.month {
            bibtex.push_str(&format!("  month = {{{}}},\n", month));
        }
        if let Some(ref doi) = self.doi {
            bibtex.push_str(&format!("  doi = {{{}}},\n", doi));
        }
        if let Some(ref url) = self.url {
            bibtex.push_str(&format!("  url = {{{}}},\n", url));
        }
        if let Some(ref venue) = self.venue {
            let field = match self.citation_type {
                CitationType::Article => "journal",
                CitationType::InProceedings => "booktitle",
                _ => "howpublished",
            };
            bibtex.push_str(&format!("  {} = {{{}}},\n", field, venue));
        }
        if let Some(ref volume) = self.volume {
            bibtex.push_str(&format!("  volume = {{{}}},\n", volume));
        }
        if let Some(ref pages) = self.pages {
            bibtex.push_str(&format!("  pages = {{{}}},\n", pages));
        }
        if let Some(ref publisher) = self.publisher {
            bibtex.push_str(&format!("  publisher = {{{}}},\n", publisher));
        }
        if let Some(ref version) = self.version {
            bibtex.push_str(&format!("  version = {{{}}},\n", version));
        }

        bibtex.push('}');
        bibtex
    }

    /// Generate CITATION.cff content
    pub fn to_cff(&self) -> String {
        let mut cff = String::from("cff-version: 1.2.0\n");
        cff.push_str(&format!("title: \"{}\"\n", self.title));
        cff.push_str("authors:\n");
        for author in &self.authors {
            cff.push_str(&format!("  - name: \"{}\"\n", author));
        }
        cff.push_str(&format!(
            "date-released: \"{}-{:02}-01\"\n",
            self.year,
            self.month.unwrap_or(1)
        ));

        if let Some(ref version) = self.version {
            cff.push_str(&format!("version: \"{}\"\n", version));
        }
        if let Some(ref doi) = self.doi {
            cff.push_str(&format!("doi: \"{}\"\n", doi));
        }
        if let Some(ref url) = self.url {
            cff.push_str(&format!("url: \"{}\"\n", url));
        }

        cff
    }
}

/// Experiment run with full tracking metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRun {
    /// Unique run ID
    pub run_id: String,
    /// Experiment name
    pub experiment_name: String,
    /// Model paradigm
    pub paradigm: ModelParadigm,
    /// Compute device used
    pub device: ComputeDevice,
    /// Platform efficiency class
    pub platform: PlatformEfficiency,
    /// Energy metrics
    pub energy: Option<EnergyMetrics>,
    /// Cost metrics
    pub cost: Option<CostMetrics>,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, serde_json::Value>,
    /// Metrics collected
    pub metrics: HashMap<String, f64>,
    /// Tags for organization
    pub tags: Vec<String>,
    /// Start time
    pub started_at: String,
    /// End time
    pub ended_at: Option<String>,
    /// Status
    pub status: RunStatus,
}

/// Run status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl ExperimentRun {
    /// Create a new experiment run
    pub fn new(
        run_id: impl Into<String>,
        experiment_name: impl Into<String>,
        paradigm: ModelParadigm,
        device: ComputeDevice,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            experiment_name: experiment_name.into(),
            paradigm,
            device,
            platform: PlatformEfficiency::Server,
            energy: None,
            cost: None,
            hyperparameters: HashMap::new(),
            metrics: HashMap::new(),
            tags: Vec::new(),
            started_at: chrono::Utc::now().to_rfc3339(),
            ended_at: None,
            status: RunStatus::Running,
        }
    }

    /// Log a metric
    pub fn log_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.insert(name.into(), value);
    }

    /// Log a hyperparameter
    pub fn log_param(&mut self, name: impl Into<String>, value: serde_json::Value) {
        self.hyperparameters.insert(name.into(), value);
    }

    /// Complete the run
    pub fn complete(&mut self) {
        self.ended_at = Some(chrono::Utc::now().to_rfc3339());
        self.status = RunStatus::Completed;
    }

    /// Mark the run as failed
    pub fn fail(&mut self) {
        self.ended_at = Some(chrono::Utc::now().to_rfc3339());
        self.status = RunStatus::Failed;
    }
}

/// Experiment storage backend trait
pub trait ExperimentStorage: Send + Sync {
    /// Store an experiment run
    fn store_run(&self, run: &ExperimentRun) -> Result<(), ExperimentError>;

    /// Retrieve a run by ID
    fn get_run(&self, run_id: &str) -> Result<Option<ExperimentRun>, ExperimentError>;

    /// List runs for an experiment
    fn list_runs(&self, experiment_name: &str) -> Result<Vec<ExperimentRun>, ExperimentError>;

    /// Delete a run
    fn delete_run(&self, run_id: &str) -> Result<(), ExperimentError>;
}

/// In-memory experiment storage for testing
#[derive(Debug, Default)]
pub struct InMemoryExperimentStorage {
    runs: std::sync::RwLock<HashMap<String, ExperimentRun>>,
}

impl InMemoryExperimentStorage {
    /// Create new in-memory storage
    pub fn new() -> Self {
        Self::default()
    }
}

impl ExperimentStorage for InMemoryExperimentStorage {
    fn store_run(&self, run: &ExperimentRun) -> Result<(), ExperimentError> {
        let mut runs = self
            .runs
            .write()
            .map_err(|e| ExperimentError::StorageError(format!("Lock error: {}", e)))?;
        runs.insert(run.run_id.clone(), run.clone());
        Ok(())
    }

    fn get_run(&self, run_id: &str) -> Result<Option<ExperimentRun>, ExperimentError> {
        let runs = self
            .runs
            .read()
            .map_err(|e| ExperimentError::StorageError(format!("Lock error: {}", e)))?;
        Ok(runs.get(run_id).cloned())
    }

    fn list_runs(&self, experiment_name: &str) -> Result<Vec<ExperimentRun>, ExperimentError> {
        let runs = self
            .runs
            .read()
            .map_err(|e| ExperimentError::StorageError(format!("Lock error: {}", e)))?;
        Ok(runs
            .values()
            .filter(|r| r.experiment_name == experiment_name)
            .cloned()
            .collect())
    }

    fn delete_run(&self, run_id: &str) -> Result<(), ExperimentError> {
        let mut runs = self
            .runs
            .write()
            .map_err(|e| ExperimentError::StorageError(format!("Lock error: {}", e)))?;
        runs.remove(run_id);
        Ok(())
    }
}

// ============================================================================
// TESTS - EXTREME TDD
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ComputeDevice Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_device_cpu_creation() {
        let cpu = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };

        match cpu {
            ComputeDevice::Cpu {
                cores,
                threads_per_core,
                architecture,
            } => {
                assert_eq!(cores, 8);
                assert_eq!(threads_per_core, 2);
                assert_eq!(architecture, CpuArchitecture::X86_64);
            }
            _ => panic!("Expected CPU device"),
        }
    }

    #[test]
    fn test_compute_device_gpu_creation() {
        let gpu = ComputeDevice::Gpu {
            name: "RTX 4090".to_string(),
            memory_gb: 24.0,
            compute_capability: Some("8.9".to_string()),
            vendor: GpuVendor::Nvidia,
        };

        match gpu {
            ComputeDevice::Gpu {
                name,
                memory_gb,
                vendor,
                ..
            } => {
                assert_eq!(name, "RTX 4090");
                assert_eq!(memory_gb, 24.0);
                assert_eq!(vendor, GpuVendor::Nvidia);
            }
            _ => panic!("Expected GPU device"),
        }
    }

    #[test]
    fn test_compute_device_tpu_creation() {
        let tpu = ComputeDevice::Tpu {
            version: TpuVersion::V4,
            cores: 8,
        };

        match tpu {
            ComputeDevice::Tpu { version, cores } => {
                assert_eq!(version, TpuVersion::V4);
                assert_eq!(cores, 8);
            }
            _ => panic!("Expected TPU device"),
        }
    }

    #[test]
    fn test_compute_device_apple_silicon_creation() {
        let m3 = ComputeDevice::AppleSilicon {
            chip: AppleChip::M3Max,
            neural_engine_cores: 16,
            gpu_cores: 40,
            memory_gb: 64,
        };

        match m3 {
            ComputeDevice::AppleSilicon {
                chip,
                gpu_cores,
                memory_gb,
                ..
            } => {
                assert_eq!(chip, AppleChip::M3Max);
                assert_eq!(gpu_cores, 40);
                assert_eq!(memory_gb, 64);
            }
            _ => panic!("Expected Apple Silicon device"),
        }
    }

    #[test]
    fn test_compute_device_edge_creation() {
        let edge = ComputeDevice::Edge {
            name: "Raspberry Pi 5".to_string(),
            power_budget_watts: 15.0,
        };

        match edge {
            ComputeDevice::Edge {
                name,
                power_budget_watts,
            } => {
                assert_eq!(name, "Raspberry Pi 5");
                assert_eq!(power_budget_watts, 15.0);
            }
            _ => panic!("Expected Edge device"),
        }
    }

    #[test]
    fn test_compute_device_theoretical_flops_cpu() {
        let cpu = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };

        let flops = cpu.theoretical_flops();
        assert!(flops > 0.0);
        // 8 cores * 2 threads * 32 FLOPS * 1e9 = 512e9
        assert_eq!(flops, 512e9);
    }

    #[test]
    fn test_compute_device_theoretical_flops_gpu() {
        let gpu = ComputeDevice::Gpu {
            name: "RTX 4090".to_string(),
            memory_gb: 24.0,
            compute_capability: Some("8.9".to_string()),
            vendor: GpuVendor::Nvidia,
        };

        let flops = gpu.theoretical_flops();
        assert!(flops > 0.0);
        // 24 GB * 15 * 1e12 = 360e12
        assert_eq!(flops, 360e12);
    }

    #[test]
    fn test_compute_device_theoretical_flops_tpu() {
        let tpu = ComputeDevice::Tpu {
            version: TpuVersion::V4,
            cores: 8,
        };

        let flops = tpu.theoretical_flops();
        assert!(flops > 0.0);
        // 8 * 275e12 = 2200e12
        assert_eq!(flops, 2200e12);
    }

    #[test]
    fn test_compute_device_estimated_power() {
        let cpu = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };
        assert_eq!(cpu.estimated_power_watts(), 120.0); // 8 * 15

        let gpu = ComputeDevice::Gpu {
            name: "RTX 4090".to_string(),
            memory_gb: 24.0,
            compute_capability: None,
            vendor: GpuVendor::Nvidia,
        };
        assert_eq!(gpu.estimated_power_watts(), 720.0); // 24 * 30
    }

    // -------------------------------------------------------------------------
    // EnergyMetrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_energy_metrics_creation() {
        let metrics = EnergyMetrics::new(3600.0, 100.0, 150.0, 36.0);

        assert_eq!(metrics.total_joules, 3600.0);
        assert_eq!(metrics.average_power_watts, 100.0);
        assert_eq!(metrics.peak_power_watts, 150.0);
        assert_eq!(metrics.duration_seconds, 36.0);
        assert_eq!(metrics.pue, 1.0);
        assert!(metrics.co2_grams.is_none());
    }

    #[test]
    fn test_energy_metrics_with_carbon_intensity() {
        let metrics =
            EnergyMetrics::new(3_600_000.0, 100.0, 150.0, 36000.0).with_carbon_intensity(400.0); // 400g CO2/kWh

        assert!(metrics.co2_grams.is_some());
        // 3600000 J = 1 kWh, * 400 g/kWh = 400g
        assert_eq!(metrics.co2_grams.unwrap(), 400.0);
    }

    #[test]
    fn test_energy_metrics_with_pue() {
        let metrics = EnergyMetrics::new(3_600_000.0, 100.0, 150.0, 36000.0)
            .with_carbon_intensity(400.0)
            .with_pue(1.5);

        assert_eq!(metrics.pue, 1.5);
        // CO2 should be 400 * 1.5 = 600
        assert_eq!(metrics.co2_grams.unwrap(), 600.0);
    }

    #[test]
    fn test_energy_metrics_flops_per_watt() {
        let metrics = EnergyMetrics::new(3600.0, 100.0, 150.0, 36.0);
        let flops_per_watt = metrics.flops_per_watt(1e15);

        assert_eq!(flops_per_watt, 1e13); // 1e15 / 100
    }

    #[test]
    fn test_energy_metrics_flops_per_watt_zero_power() {
        let metrics = EnergyMetrics::new(0.0, 0.0, 0.0, 0.0);
        let flops_per_watt = metrics.flops_per_watt(1e15);

        assert_eq!(flops_per_watt, 0.0);
    }

    // -------------------------------------------------------------------------
    // CostMetrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cost_metrics_creation() {
        let metrics = CostMetrics::new(10.0, 2.0, 0.5);

        assert_eq!(metrics.compute_cost_usd, 10.0);
        assert_eq!(metrics.storage_cost_usd, 2.0);
        assert_eq!(metrics.network_cost_usd, 0.5);
        assert_eq!(metrics.total_cost_usd, 12.5);
        assert_eq!(metrics.currency, "USD");
    }

    #[test]
    fn test_cost_metrics_with_flops() {
        let metrics = CostMetrics::new(10.0, 0.0, 0.0).with_flops(1e18);

        assert!(metrics.cost_per_flop.is_some());
        assert_eq!(metrics.cost_per_flop.unwrap(), 1e-17);
    }

    #[test]
    fn test_cost_metrics_with_samples() {
        let metrics = CostMetrics::new(10.0, 0.0, 0.0).with_samples(1000);

        assert!(metrics.cost_per_sample.is_some());
        assert_eq!(metrics.cost_per_sample.unwrap(), 0.01);
    }

    // -------------------------------------------------------------------------
    // ModelParadigm Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_model_paradigm_compute_intensity() {
        assert_eq!(
            ModelParadigm::TraditionalML.compute_intensity(),
            ComputeIntensity::Low
        );
        assert_eq!(
            ModelParadigm::DeepLearning.compute_intensity(),
            ComputeIntensity::High
        );
        assert_eq!(
            ModelParadigm::MoE.compute_intensity(),
            ComputeIntensity::VeryHigh
        );
        assert_eq!(
            ModelParadigm::FineTuning.compute_intensity(),
            ComputeIntensity::Medium
        );
    }

    #[test]
    fn test_model_paradigm_benefits_from_gpu() {
        assert!(ModelParadigm::DeepLearning.benefits_from_gpu());
        assert!(ModelParadigm::FineTuning.benefits_from_gpu());
        assert!(ModelParadigm::MoE.benefits_from_gpu());
        assert!(!ModelParadigm::TraditionalML.benefits_from_gpu());
        assert!(!ModelParadigm::FederatedLearning.benefits_from_gpu());
    }

    // -------------------------------------------------------------------------
    // CostPerformanceBenchmark Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_benchmark_creation() {
        let benchmark = CostPerformanceBenchmark::new("test-benchmark");
        assert_eq!(benchmark.name, "test-benchmark");
        assert!(benchmark.points.is_empty());
    }

    #[test]
    fn test_benchmark_add_point() {
        let mut benchmark = CostPerformanceBenchmark::new("test");
        benchmark.add_point(CostPerformancePoint {
            id: "p1".to_string(),
            performance: 0.95,
            cost: 100.0,
            energy_joules: 1000.0,
            latency_ms: Some(10.0),
            metadata: HashMap::new(),
        });

        assert_eq!(benchmark.points.len(), 1);
    }

    #[test]
    fn test_pareto_frontier_single_point() {
        let mut benchmark = CostPerformanceBenchmark::new("test");
        benchmark.add_point(CostPerformancePoint {
            id: "p1".to_string(),
            performance: 0.95,
            cost: 100.0,
            energy_joules: 1000.0,
            latency_ms: None,
            metadata: HashMap::new(),
        });

        let frontier = benchmark.compute_pareto_frontier();
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0], 0);
    }

    #[test]
    fn test_pareto_frontier_dominated_point() {
        let mut benchmark = CostPerformanceBenchmark::new("test");

        // Point that dominates
        benchmark.add_point(CostPerformancePoint {
            id: "dominant".to_string(),
            performance: 0.95,
            cost: 50.0,
            energy_joules: 500.0,
            latency_ms: None,
            metadata: HashMap::new(),
        });

        // Dominated point (worse on both dimensions)
        benchmark.add_point(CostPerformancePoint {
            id: "dominated".to_string(),
            performance: 0.90,
            cost: 100.0,
            energy_joules: 1000.0,
            latency_ms: None,
            metadata: HashMap::new(),
        });

        let frontier = benchmark.compute_pareto_frontier().to_vec();
        assert_eq!(frontier.len(), 1);
        assert_eq!(benchmark.points[frontier[0]].id, "dominant");
    }

    #[test]
    fn test_pareto_frontier_multiple_optimal() {
        let mut benchmark = CostPerformanceBenchmark::new("test");

        // High performance, high cost
        benchmark.add_point(CostPerformancePoint {
            id: "high-perf".to_string(),
            performance: 0.99,
            cost: 200.0,
            energy_joules: 2000.0,
            latency_ms: None,
            metadata: HashMap::new(),
        });

        // Low performance, low cost
        benchmark.add_point(CostPerformancePoint {
            id: "low-cost".to_string(),
            performance: 0.85,
            cost: 20.0,
            energy_joules: 200.0,
            latency_ms: None,
            metadata: HashMap::new(),
        });

        let frontier = benchmark.compute_pareto_frontier();
        assert_eq!(frontier.len(), 2); // Both are Pareto-optimal
    }

    #[test]
    fn test_best_within_budget() {
        let mut benchmark = CostPerformanceBenchmark::new("test");

        benchmark.add_point(CostPerformancePoint {
            id: "expensive".to_string(),
            performance: 0.99,
            cost: 200.0,
            energy_joules: 2000.0,
            latency_ms: None,
            metadata: HashMap::new(),
        });

        benchmark.add_point(CostPerformancePoint {
            id: "cheap".to_string(),
            performance: 0.85,
            cost: 20.0,
            energy_joules: 200.0,
            latency_ms: None,
            metadata: HashMap::new(),
        });

        let best = benchmark.best_within_budget(50.0);
        assert!(best.is_some());
        assert_eq!(best.unwrap().id, "cheap");
    }

    #[test]
    fn test_efficiency_scores() {
        let mut benchmark = CostPerformanceBenchmark::new("test");

        benchmark.add_point(CostPerformancePoint {
            id: "p1".to_string(),
            performance: 0.90,
            cost: 100.0,
            energy_joules: 1000.0,
            latency_ms: None,
            metadata: HashMap::new(),
        });

        let scores = benchmark.efficiency_scores();
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].0, 0);
        // Use approximate comparison for floating point
        assert!((scores[0].1 - 0.009).abs() < 1e-10); // 0.90 / 100
    }

    // -------------------------------------------------------------------------
    // SovereignDistribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sovereign_distribution_creation() {
        let dist = SovereignDistribution::new("my-model", "1.0.0");
        assert_eq!(dist.name, "my-model");
        assert_eq!(dist.version, "1.0.0");
        assert!(dist.platforms.is_empty());
        assert!(dist.artifacts.is_empty());
    }

    #[test]
    fn test_sovereign_distribution_add_platform() {
        let mut dist = SovereignDistribution::new("my-model", "1.0.0");
        dist.add_platform("linux-x86_64");
        dist.add_platform("darwin-aarch64");

        assert_eq!(dist.platforms.len(), 2);
        assert!(dist.platforms.contains(&"linux-x86_64".to_string()));
    }

    #[test]
    fn test_sovereign_distribution_add_artifact() {
        let mut dist = SovereignDistribution::new("my-model", "1.0.0");
        dist.add_artifact(SovereignArtifact {
            name: "model.onnx".to_string(),
            artifact_type: ArtifactType::Model,
            sha256: "abc123".to_string(),
            size_bytes: 1_000_000,
            source_url: None,
        });

        assert_eq!(dist.artifacts.len(), 1);
        assert_eq!(dist.total_size_bytes(), 1_000_000);
    }

    #[test]
    fn test_sovereign_distribution_validate_signatures_missing() {
        let mut dist = SovereignDistribution::new("my-model", "1.0.0");
        dist.add_artifact(SovereignArtifact {
            name: "model.onnx".to_string(),
            artifact_type: ArtifactType::Model,
            sha256: "abc123".to_string(),
            size_bytes: 1_000_000,
            source_url: None,
        });

        let result = dist.validate_signatures();
        assert!(result.is_err());
        match result {
            Err(ExperimentError::SovereignValidationFailed(msg)) => {
                assert!(msg.contains("model.onnx"));
            }
            _ => panic!("Expected SovereignValidationFailed error"),
        }
    }

    #[test]
    fn test_sovereign_distribution_validate_signatures_present() {
        let mut dist = SovereignDistribution::new("my-model", "1.0.0");
        dist.add_artifact(SovereignArtifact {
            name: "model.onnx".to_string(),
            artifact_type: ArtifactType::Model,
            sha256: "abc123".to_string(),
            size_bytes: 1_000_000,
            source_url: None,
        });
        dist.signatures.push(ArtifactSignature {
            artifact_name: "model.onnx".to_string(),
            algorithm: SignatureAlgorithm::Ed25519,
            signature: "sig123".to_string(),
            key_id: "key1".to_string(),
        });

        let result = dist.validate_signatures();
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // ORCID Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_orcid_valid() {
        let orcid = Orcid::new("0000-0002-1825-0097");
        assert!(orcid.is_ok());
        assert_eq!(orcid.unwrap().as_str(), "0000-0002-1825-0097");
    }

    #[test]
    fn test_orcid_valid_with_x() {
        let orcid = Orcid::new("0000-0002-1825-009X");
        assert!(orcid.is_ok());
    }

    #[test]
    fn test_orcid_invalid_format() {
        let orcid = Orcid::new("invalid-orcid");
        assert!(orcid.is_err());
        match orcid {
            Err(ExperimentError::InvalidOrcid(s)) => {
                assert_eq!(s, "invalid-orcid");
            }
            _ => panic!("Expected InvalidOrcid error"),
        }
    }

    #[test]
    fn test_orcid_invalid_too_short() {
        let orcid = Orcid::new("0000-0002-1825");
        assert!(orcid.is_err());
    }

    // -------------------------------------------------------------------------
    // CitationMetadata Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_citation_metadata_to_bibtex() {
        let citation = CitationMetadata {
            citation_type: CitationType::Article,
            title: "Test Paper".to_string(),
            authors: vec!["Alice Smith".to_string(), "Bob Jones".to_string()],
            year: 2024,
            month: Some(6),
            doi: Some("10.1234/test".to_string()),
            url: None,
            venue: Some("Journal of Testing".to_string()),
            volume: Some("42".to_string()),
            pages: Some("1-10".to_string()),
            publisher: None,
            version: None,
        };

        let bibtex = citation.to_bibtex("smith2024test");
        assert!(bibtex.contains("@article{smith2024test,"));
        assert!(bibtex.contains("title = {Test Paper}"));
        assert!(bibtex.contains("author = {Alice Smith and Bob Jones}"));
        assert!(bibtex.contains("year = {2024}"));
        assert!(bibtex.contains("doi = {10.1234/test}"));
        assert!(bibtex.contains("journal = {Journal of Testing}"));
    }

    #[test]
    fn test_citation_metadata_to_cff() {
        let citation = CitationMetadata {
            citation_type: CitationType::Software,
            title: "My Tool".to_string(),
            authors: vec!["Developer".to_string()],
            year: 2024,
            month: Some(11),
            doi: Some("10.5281/zenodo.123".to_string()),
            url: Some("https://github.com/example/tool".to_string()),
            venue: None,
            volume: None,
            pages: None,
            publisher: None,
            version: Some("1.0.0".to_string()),
        };

        let cff = citation.to_cff();
        assert!(cff.contains("cff-version: 1.2.0"));
        assert!(cff.contains("title: \"My Tool\""));
        assert!(cff.contains("name: \"Developer\""));
        assert!(cff.contains("version: \"1.0.0\""));
        assert!(cff.contains("doi: \"10.5281/zenodo.123\""));
    }

    // -------------------------------------------------------------------------
    // ExperimentRun Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_experiment_run_creation() {
        let device = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };

        let run = ExperimentRun::new(
            "run-001",
            "my-experiment",
            ModelParadigm::DeepLearning,
            device,
        );

        assert_eq!(run.run_id, "run-001");
        assert_eq!(run.experiment_name, "my-experiment");
        assert_eq!(run.paradigm, ModelParadigm::DeepLearning);
        assert_eq!(run.status, RunStatus::Running);
        assert!(run.ended_at.is_none());
    }

    #[test]
    fn test_experiment_run_log_metric() {
        let device = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };

        let mut run = ExperimentRun::new(
            "run-001",
            "my-experiment",
            ModelParadigm::DeepLearning,
            device,
        );
        run.log_metric("accuracy", 0.95);
        run.log_metric("loss", 0.05);

        assert_eq!(run.metrics.get("accuracy"), Some(&0.95));
        assert_eq!(run.metrics.get("loss"), Some(&0.05));
    }

    #[test]
    fn test_experiment_run_log_param() {
        let device = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };

        let mut run = ExperimentRun::new(
            "run-001",
            "my-experiment",
            ModelParadigm::DeepLearning,
            device,
        );
        run.log_param("learning_rate", serde_json::json!(0.001));
        run.log_param("batch_size", serde_json::json!(32));

        assert_eq!(
            run.hyperparameters.get("learning_rate"),
            Some(&serde_json::json!(0.001))
        );
    }

    #[test]
    fn test_experiment_run_complete() {
        let device = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };

        let mut run = ExperimentRun::new(
            "run-001",
            "my-experiment",
            ModelParadigm::DeepLearning,
            device,
        );
        run.complete();

        assert_eq!(run.status, RunStatus::Completed);
        assert!(run.ended_at.is_some());
    }

    #[test]
    fn test_experiment_run_fail() {
        let device = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };

        let mut run = ExperimentRun::new(
            "run-001",
            "my-experiment",
            ModelParadigm::DeepLearning,
            device,
        );
        run.fail();

        assert_eq!(run.status, RunStatus::Failed);
        assert!(run.ended_at.is_some());
    }

    // -------------------------------------------------------------------------
    // ExperimentStorage Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_in_memory_storage_store_and_get() {
        let storage = InMemoryExperimentStorage::new();
        let device = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };

        let run = ExperimentRun::new(
            "run-001",
            "my-experiment",
            ModelParadigm::DeepLearning,
            device,
        );
        storage.store_run(&run).unwrap();

        let retrieved = storage.get_run("run-001").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().run_id, "run-001");
    }

    #[test]
    fn test_in_memory_storage_get_nonexistent() {
        let storage = InMemoryExperimentStorage::new();
        let retrieved = storage.get_run("nonexistent").unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_in_memory_storage_list_runs() {
        let storage = InMemoryExperimentStorage::new();
        let device = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };

        let run1 = ExperimentRun::new(
            "run-001",
            "exp-a",
            ModelParadigm::DeepLearning,
            device.clone(),
        );
        let run2 = ExperimentRun::new(
            "run-002",
            "exp-a",
            ModelParadigm::FineTuning,
            device.clone(),
        );
        let run3 = ExperimentRun::new("run-003", "exp-b", ModelParadigm::TraditionalML, device);

        storage.store_run(&run1).unwrap();
        storage.store_run(&run2).unwrap();
        storage.store_run(&run3).unwrap();

        let runs = storage.list_runs("exp-a").unwrap();
        assert_eq!(runs.len(), 2);
    }

    #[test]
    fn test_in_memory_storage_delete_run() {
        let storage = InMemoryExperimentStorage::new();
        let device = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };

        let run = ExperimentRun::new(
            "run-001",
            "my-experiment",
            ModelParadigm::DeepLearning,
            device,
        );
        storage.store_run(&run).unwrap();

        storage.delete_run("run-001").unwrap();
        let retrieved = storage.get_run("run-001").unwrap();
        assert!(retrieved.is_none());
    }

    // -------------------------------------------------------------------------
    // PlatformEfficiency Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_platform_efficiency_power_budget() {
        assert_eq!(
            PlatformEfficiency::Server.typical_power_budget_watts(),
            500.0
        );
        assert_eq!(
            PlatformEfficiency::Laptop.typical_power_budget_watts(),
            65.0
        );
        assert_eq!(PlatformEfficiency::Edge.typical_power_budget_watts(), 15.0);
        assert_eq!(PlatformEfficiency::Mobile.typical_power_budget_watts(), 5.0);
        assert_eq!(
            PlatformEfficiency::Embedded.typical_power_budget_watts(),
            1.0
        );
    }

    // -------------------------------------------------------------------------
    // Serialization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_device_serialization() {
        let device = ComputeDevice::Gpu {
            name: "RTX 4090".to_string(),
            memory_gb: 24.0,
            compute_capability: Some("8.9".to_string()),
            vendor: GpuVendor::Nvidia,
        };

        let json = serde_json::to_string(&device).unwrap();
        let deserialized: ComputeDevice = serde_json::from_str(&json).unwrap();
        assert_eq!(device, deserialized);
    }

    #[test]
    fn test_experiment_run_serialization() {
        let device = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };

        let mut run = ExperimentRun::new(
            "run-001",
            "my-experiment",
            ModelParadigm::DeepLearning,
            device,
        );
        run.log_metric("accuracy", 0.95);

        let json = serde_json::to_string(&run).unwrap();
        let deserialized: ExperimentRun = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.run_id, "run-001");
        assert_eq!(deserialized.metrics.get("accuracy"), Some(&0.95));
    }

    // -------------------------------------------------------------------------
    // CRediT Role Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_credit_role_variants() {
        let roles = vec![
            CreditRole::Conceptualization,
            CreditRole::DataCuration,
            CreditRole::FormalAnalysis,
            CreditRole::FundingAcquisition,
            CreditRole::Investigation,
            CreditRole::Methodology,
            CreditRole::ProjectAdministration,
            CreditRole::Resources,
            CreditRole::Software,
            CreditRole::Supervision,
            CreditRole::Validation,
            CreditRole::Visualization,
            CreditRole::WritingOriginalDraft,
            CreditRole::WritingReviewEditing,
        ];
        assert_eq!(roles.len(), 14);
    }

    #[test]
    fn test_research_contributor_creation() {
        let contributor = ResearchContributor {
            name: "Alice Researcher".to_string(),
            orcid: Orcid::new("0000-0002-1825-0097").ok(),
            affiliation: "MIT".to_string(),
            roles: vec![CreditRole::Conceptualization, CreditRole::Software],
            email: Some("alice@mit.edu".to_string()),
        };

        assert_eq!(contributor.name, "Alice Researcher");
        assert!(contributor.orcid.is_some());
        assert_eq!(contributor.roles.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Error Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_experiment_error_display() {
        let err = ExperimentError::InvalidComputeDevice("bad config".to_string());
        assert_eq!(
            format!("{}", err),
            "Invalid compute device configuration: bad config"
        );

        let err = ExperimentError::InvalidOrcid("bad-orcid".to_string());
        assert_eq!(format!("{}", err), "Invalid ORCID format: bad-orcid");
    }
}
