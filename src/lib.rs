// Library exports for Batuta orchestration framework
pub mod analyzer;
pub mod backend;
pub mod config;
pub mod pipeline;
pub mod report;
pub mod tools;
pub mod types;

// Re-export key types for convenience
pub use backend::{Backend, BackendSelector};
pub use pipeline::{
    AnalysisStage, BuildStage, OptimizationStage, PipelineStage, TranspilationPipeline,
    TranspilationStage, ValidationStage, ValidationStrategy,
};
pub use report::{MigrationReport, ReportFormat};
pub use types::{
    Language, PhaseStatus, ProjectAnalysis, WorkflowPhase, WorkflowState,
};
