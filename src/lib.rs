#![allow(dead_code)]
#![allow(unused_imports)]

// ANSI colors module (DEP-REDUCE: replaces colored crate)
pub mod ansi_colors;
pub mod timing;

// Library exports for Batuta orchestration framework
pub mod analyzer;
pub mod audit;
pub mod backend;
pub mod config;
pub mod experiment;
pub mod numpy_converter;
pub mod oracle;
pub mod parf;
pub mod pipeline;
pub mod pipeline_analysis;
pub mod plugin;
pub mod pytorch_converter;
pub mod recipes;
pub mod report;
pub mod sklearn_converter;
pub mod tools;
pub mod types;

// WASM-specific API (only compiled for wasm32 target)
#[cfg(feature = "wasm")]
pub mod wasm;

// Model Serving Ecosystem (native-only)
#[cfg(feature = "native")]
pub mod serve;

// Stack Dependency Orchestration (native-only)
#[cfg(feature = "native")]
pub mod stack;

// Data Platforms Integration
pub mod data;

// Visualization Frameworks Integration
pub mod viz;

// Content Creation Tooling
pub mod content;

// TUI Graph Visualization (native-only, requires ratatui)
#[cfg(feature = "native")]
pub mod tui;

// Popperian Falsification Checklist (native-only)
#[cfg(feature = "native")]
pub mod falsification;

// Proactive Bug Hunting (native-only)
#[cfg(feature = "native")]
pub mod bug_hunter;

// Stack Compliance Engine (native-only)
#[cfg(feature = "native")]
pub mod comply;

// HuggingFace Hub Integration (native-only)
#[cfg(feature = "native")]
pub mod hf;

// MCP (Model Context Protocol) Server
#[cfg(feature = "native")]
pub mod mcp;

// Re-export key types for convenience
pub use backend::{Backend, BackendSelector, OpComplexity};
pub use numpy_converter::{NumPyConverter, NumPyOp};
pub use parf::{
    CodePattern, DeadCode, DependencyKind, FileDependency, ParfAnalyzer, SymbolKind,
    SymbolReference,
};
pub use pipeline::{
    AnalysisStage, BuildStage, OptimizationStage, PipelineStage, TranspilationPipeline,
    TranspilationStage, ValidationStage, ValidationStrategy,
};
pub use plugin::{PluginMetadata, PluginRegistry, PluginStage, TranspilerPlugin};
pub use pytorch_converter::{PyTorchConverter, PyTorchOperation, RealizarOperation};
pub use report::{MigrationReport, ReportFormat};
pub use sklearn_converter::{AprenderAlgorithm, SklearnAlgorithm, SklearnConverter};
pub use types::{Language, PhaseStatus, ProjectAnalysis, WorkflowPhase, WorkflowState};

// Audit Trail exports (entrenar-style provenance)
pub use audit::{
    new_audit_collector, record_failure, record_success, ChainVerification, ContextSnapshot,
    HashChainEntry, PipelineAuditCollector, PipelinePath, PipelineTrace, StageTimer,
};

// Experiment Tracking exports (Entrenar v1.8.0 integration)
pub use experiment::{
    AppleChip, ArtifactSignature, ArtifactType, CitationMetadata, CitationType, ComputeDevice,
    ComputeIntensity, CostMetrics, CostPerformanceBenchmark, CostPerformancePoint, CpuArchitecture,
    CreditRole, EnergyMetrics, ExperimentError, ExperimentRun, ExperimentStorage, GpuVendor,
    InMemoryExperimentStorage, ModelParadigm, OfflineRegistryConfig, Orcid, PlatformEfficiency,
    PreRegistration, ResearchArtifact, ResearchContributor, RunStatus, SignatureAlgorithm,
    SovereignArtifact, SovereignDistribution, TpuVersion,
};

// Orchestration Recipes exports
pub use recipes::{
    CiCdBenchmarkRecipe, CostPerformanceBenchmarkRecipe, ExperimentTrackingConfig,
    ExperimentTrackingRecipe, RecipeResult, ResearchArtifactRecipe, SovereignDeploymentConfig,
    SovereignDeploymentRecipe,
};

// Oracle Mode exports
pub use oracle::{
    Backend as OracleBackend, Capability, CapabilityCategory, ComponentRecommendation,
    ComputeRecommendation, DataSize, DistributionRecommendation, HardwareSpec, IntegrationPattern,
    KnowledgeGraph, OpComplexity as OracleOpComplexity, OptimizationTarget, OracleQuery,
    OracleResponse, ProblemDomain, QueryConstraints, QueryEngine, QueryPreferences, Recommender,
    StackComponent, StackLayer,
};

// Model Serving Ecosystem exports (native-only)
#[cfg(feature = "native")]
pub use serve::{
    BackendSelector as ServeBackendSelector, ChatMessage, ChatTemplateEngine, CircuitBreakerConfig,
    ContextManager, ContextWindow, CostCircuitBreaker, FailoverConfig, FailoverManager,
    PrivacyTier, RejectReason, Role, RouterConfig, RoutingDecision, ServingBackend,
    SpilloverRouter, StreamingContext, TemplateFormat, TokenEstimator, TruncationStrategy,
};

// Stack Dependency Orchestration exports (native-only)
#[cfg(feature = "native")]
pub use stack::{
    // Diagnostics types
    render_dashboard,
    AndonStatus,
    Anomaly,
    AnomalyCategory,
    ComponentMetrics,
    ComponentNode,
    // Quality types
    ComponentQuality,
    ErrorForecaster,
    ForecastMetrics,
    GraphMetrics,
    HealthStatus,
    HealthSummary,
    HeroImageResult,
    ImageFormat,
    IsolationForest,
    QualityChecker,
    QualityGrade,
    QualityIssue,
    QualitySummary,
    Score,
    StackDiagnostics,
    StackLayer as QualityStackLayer,
    StackQualityReport,
};

// Popperian Falsification Checklist exports (native-only)
#[cfg(feature = "native")]
pub use falsification::{
    evaluate_critical_only, evaluate_project, CheckItem, CheckStatus, ChecklistResult, Evidence,
    EvidenceType, Severity, TpsGrade,
};


// Stack Compliance Engine exports (native-only)
#[cfg(feature = "native")]
pub use comply::{ComplyConfig, ComplyReport, ComplyReportFormat, StackComplyEngine};
