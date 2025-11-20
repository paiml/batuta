#![allow(dead_code)]
#![allow(unused_imports)]

// Library exports for Batuta orchestration framework
pub mod analyzer;
pub mod backend;
pub mod config;
pub mod numpy_converter;
pub mod parf;
pub mod pipeline;
pub mod pytorch_converter;
pub mod report;
pub mod sklearn_converter;
pub mod tools;
pub mod types;

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
pub use pytorch_converter::{PyTorchConverter, PyTorchOperation, RealizarOperation};
pub use report::{MigrationReport, ReportFormat};
pub use sklearn_converter::{AprenderAlgorithm, SklearnAlgorithm, SklearnConverter};
pub use types::{
    Language, PhaseStatus, ProjectAnalysis, WorkflowPhase, WorkflowState,
};
