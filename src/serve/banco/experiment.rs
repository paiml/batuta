//! Experiment tracking — group training runs, compare metrics.
//!
//! An experiment is a named collection of training runs that can be compared.
//! This enables the iterate loop: train → eval → compare → retrain.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// An experiment (group of training runs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub description: String,
    pub run_ids: Vec<String>,
    pub created_at: u64,
}

/// Comparison of runs within an experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunComparison {
    pub experiment_id: String,
    pub runs: Vec<RunSummary>,
    pub best_run: Option<String>,
}

/// Summary of a single run for comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub id: String,
    pub method: String,
    pub status: String,
    pub final_loss: Option<f32>,
    pub total_steps: u64,
}

/// Experiment store.
pub struct ExperimentStore {
    experiments: RwLock<HashMap<String, Experiment>>,
    counter: std::sync::atomic::AtomicU64,
}

impl ExperimentStore {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            experiments: RwLock::new(HashMap::new()),
            counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Create a new experiment.
    pub fn create(&self, name: &str, description: &str) -> Experiment {
        let seq = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let exp = Experiment {
            id: format!("exp-{}-{seq}", epoch_secs()),
            name: name.to_string(),
            description: description.to_string(),
            run_ids: Vec::new(),
            created_at: epoch_secs(),
        };
        if let Ok(mut store) = self.experiments.write() {
            store.insert(exp.id.clone(), exp.clone());
        }
        exp
    }

    /// Add a run to an experiment.
    pub fn add_run(&self, experiment_id: &str, run_id: &str) -> Result<(), ExperimentError> {
        let mut store = self.experiments.write().map_err(|_| ExperimentError::LockPoisoned)?;
        let exp = store
            .get_mut(experiment_id)
            .ok_or(ExperimentError::NotFound(experiment_id.to_string()))?;
        if !exp.run_ids.contains(&run_id.to_string()) {
            exp.run_ids.push(run_id.to_string());
        }
        Ok(())
    }

    /// List all experiments.
    #[must_use]
    pub fn list(&self) -> Vec<Experiment> {
        let store = self.experiments.read().unwrap_or_else(|e| e.into_inner());
        let mut exps: Vec<Experiment> = store.values().cloned().collect();
        exps.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        exps
    }

    /// Get experiment by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<Experiment> {
        self.experiments.read().unwrap_or_else(|e| e.into_inner()).get(id).cloned()
    }

    /// Compare runs in an experiment using the training store.
    pub fn compare(
        &self,
        experiment_id: &str,
        training: &super::training::TrainingStore,
    ) -> Result<RunComparison, ExperimentError> {
        let exp =
            self.get(experiment_id).ok_or(ExperimentError::NotFound(experiment_id.to_string()))?;

        let mut summaries = Vec::new();
        let mut best_loss = f32::MAX;
        let mut best_id = None;

        for run_id in &exp.run_ids {
            if let Some(run) = training.get(run_id) {
                let final_loss = run.metrics.last().map(|m| m.loss);
                let total_steps = run.metrics.last().map(|m| m.step).unwrap_or(0);

                if let Some(loss) = final_loss {
                    if loss < best_loss {
                        best_loss = loss;
                        best_id = Some(run_id.clone());
                    }
                }

                summaries.push(RunSummary {
                    id: run_id.clone(),
                    method: format!("{:?}", run.method),
                    status: format!("{:?}", run.status),
                    final_loss,
                    total_steps,
                });
            }
        }

        Ok(RunComparison {
            experiment_id: experiment_id.to_string(),
            runs: summaries,
            best_run: best_id,
        })
    }
}

/// Experiment errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentError {
    NotFound(String),
    LockPoisoned,
}

impl std::fmt::Display for ExperimentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Experiment not found: {id}"),
            Self::LockPoisoned => write!(f, "Internal lock error"),
        }
    }
}

impl std::error::Error for ExperimentError {}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}
