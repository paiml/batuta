#![allow(dead_code)]
#![allow(unused_imports)]

// Library exports for Batuta orchestration framework
pub mod analyzer;
pub mod backend;
pub mod config;
pub mod numpy_converter;
pub mod oracle;
pub mod parf;
pub mod pipeline;
pub mod plugin;
pub mod pytorch_converter;
pub mod report;
pub mod sklearn_converter;
pub mod tools;
pub mod types;

// WASM-specific API (only compiled for wasm32 target)
#[cfg(feature = "wasm")]
pub mod wasm;

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
pub use types::{
    Language, PhaseStatus, ProjectAnalysis, WorkflowPhase, WorkflowState,
};

// Oracle Mode exports
pub use oracle::{
    Backend as OracleBackend, Capability, CapabilityCategory, ComponentRecommendation,
    ComputeRecommendation, DataSize, DistributionRecommendation, HardwareSpec,
    IntegrationPattern, KnowledgeGraph, OpComplexity as OracleOpComplexity,
    OptimizationTarget, OracleQuery, OracleResponse, ProblemDomain, QueryConstraints,
    QueryEngine, QueryPreferences, Recommender, StackComponent, StackLayer,
};
