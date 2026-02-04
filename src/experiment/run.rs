//! Experiment run tracking and storage.
//!
//! This module provides types for tracking individual experiment runs,
//! including metrics, hyperparameters, and storage backends.

use super::{
    ComputeDevice, CostMetrics, EnergyMetrics, ExperimentError, ModelParadigm, PlatformEfficiency,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
