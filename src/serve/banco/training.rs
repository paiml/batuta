//! Training run management — start, stop, list, metrics.
//!
//! Phase 3 skeleton: tracks training runs with status and config.
//! Actual training via entrenar is wired when `ml` feature is enabled.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Training run metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRun {
    pub id: String,
    pub dataset_id: String,
    pub method: TrainingMethod,
    pub config: TrainingConfig,
    pub status: TrainingStatus,
    pub created_at: u64,
    pub metrics: Vec<TrainingMetric>,
}

/// Training method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMethod {
    Lora,
    Qlora,
    FullFinetune,
}

/// Training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    #[serde(default = "default_lora_r")]
    pub lora_r: u32,
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: u32,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_max_seq_length")]
    pub max_seq_length: u32,
}

fn default_lora_r() -> u32 {
    16
}
fn default_lora_alpha() -> u32 {
    32
}
fn default_learning_rate() -> f64 {
    2e-4
}
fn default_epochs() -> u32 {
    3
}
fn default_batch_size() -> u32 {
    4
}
fn default_max_seq_length() -> u32 {
    2048
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            lora_r: default_lora_r(),
            lora_alpha: default_lora_alpha(),
            learning_rate: default_learning_rate(),
            epochs: default_epochs(),
            batch_size: default_batch_size(),
            max_seq_length: default_max_seq_length(),
        }
    }
}

/// Training run status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingStatus {
    Queued,
    Running,
    Complete,
    Failed,
    Stopped,
}

/// A single training metric snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetric {
    pub step: u64,
    pub loss: f32,
    pub learning_rate: f64,
}

/// Training run store.
pub struct TrainingStore {
    runs: RwLock<HashMap<String, TrainingRun>>,
    counter: std::sync::atomic::AtomicU64,
}

impl TrainingStore {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            runs: RwLock::new(HashMap::new()),
            counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Start a training run (dry-run without ml feature).
    pub fn start(
        &self,
        dataset_id: &str,
        method: TrainingMethod,
        config: TrainingConfig,
    ) -> TrainingRun {
        let seq = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let run = TrainingRun {
            id: format!("run-{}-{seq}", epoch_secs()),
            dataset_id: dataset_id.to_string(),
            method,
            config,
            status: TrainingStatus::Queued,
            created_at: epoch_secs(),
            metrics: Vec::new(),
        };
        if let Ok(mut store) = self.runs.write() {
            store.insert(run.id.clone(), run.clone());
        }
        run
    }

    /// List all runs.
    #[must_use]
    pub fn list(&self) -> Vec<TrainingRun> {
        let store = self.runs.read().unwrap_or_else(|e| e.into_inner());
        let mut runs: Vec<TrainingRun> = store.values().cloned().collect();
        runs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        runs
    }

    /// Get a run by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<TrainingRun> {
        self.runs.read().unwrap_or_else(|e| e.into_inner()).get(id).cloned()
    }

    /// Stop a run.
    pub fn stop(&self, id: &str) -> Result<(), TrainingError> {
        let mut store = self.runs.write().map_err(|_| TrainingError::LockPoisoned)?;
        let run = store.get_mut(id).ok_or(TrainingError::NotFound(id.to_string()))?;
        run.status = TrainingStatus::Stopped;
        Ok(())
    }

    /// Delete a run.
    pub fn delete(&self, id: &str) -> Result<(), TrainingError> {
        let mut store = self.runs.write().map_err(|_| TrainingError::LockPoisoned)?;
        store.remove(id).ok_or(TrainingError::NotFound(id.to_string()))?;
        Ok(())
    }
}

/// Training errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingError {
    NotFound(String),
    LockPoisoned,
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Training run not found: {id}"),
            Self::LockPoisoned => write!(f, "Internal lock error"),
        }
    }
}

impl std::error::Error for TrainingError {}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}
