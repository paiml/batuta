//! Data recipe engine — declarative pipelines that transform files into datasets.
//!
//! A recipe is a sequence of steps: extract_text → chunk → filter → format.
//! Each step transforms records, producing a dataset suitable for training.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// A data recipe definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    pub id: String,
    pub name: String,
    pub source_files: Vec<String>,
    pub steps: Vec<RecipeStep>,
    pub output_format: String,
    pub created_at: u64,
    pub status: RecipeStatus,
}

/// Recipe execution status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecipeStatus {
    Created,
    Running,
    Complete,
    Failed,
}

/// A single step in a recipe pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeStep {
    #[serde(rename = "type")]
    pub step_type: StepType,
    #[serde(default)]
    pub config: serde_json::Value,
}

/// Built-in recipe step types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StepType {
    ExtractText,
    Chunk,
    Filter,
    Format,
    Deduplicate,
}

/// A single record flowing through the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub text: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Result of running a recipe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetResult {
    pub dataset_id: String,
    pub recipe_id: String,
    pub record_count: usize,
    pub records: Vec<Record>,
}

/// Recipe store — manages recipe definitions and execution.
pub struct RecipeStore {
    recipes: RwLock<HashMap<String, Recipe>>,
    datasets: RwLock<HashMap<String, DatasetResult>>,
    counter: std::sync::atomic::AtomicU64,
}

impl RecipeStore {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            recipes: RwLock::new(HashMap::new()),
            datasets: RwLock::new(HashMap::new()),
            counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Create a recipe definition.
    pub fn create(
        &self,
        name: &str,
        source_files: Vec<String>,
        steps: Vec<RecipeStep>,
        output_format: &str,
    ) -> Recipe {
        let seq = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let recipe = Recipe {
            id: format!("recipe-{}-{seq}", epoch_secs()),
            name: name.to_string(),
            source_files,
            steps,
            output_format: output_format.to_string(),
            created_at: epoch_secs(),
            status: RecipeStatus::Created,
        };
        if let Ok(mut store) = self.recipes.write() {
            store.insert(recipe.id.clone(), recipe.clone());
        }
        recipe
    }

    /// List all recipes.
    #[must_use]
    pub fn list(&self) -> Vec<Recipe> {
        let store = self.recipes.read().unwrap_or_else(|e| e.into_inner());
        let mut recipes: Vec<Recipe> = store.values().cloned().collect();
        recipes.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        recipes
    }

    /// Get a recipe by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<Recipe> {
        self.recipes.read().unwrap_or_else(|e| e.into_inner()).get(id).cloned()
    }

    /// Run a recipe against source texts, producing a dataset.
    pub fn run(
        &self,
        recipe_id: &str,
        source_texts: &[(&str, &str)],
    ) -> Result<DatasetResult, RecipeError> {
        let recipe = self.get(recipe_id).ok_or(RecipeError::NotFound(recipe_id.to_string()))?;

        // Update status
        if let Ok(mut store) = self.recipes.write() {
            if let Some(r) = store.get_mut(recipe_id) {
                r.status = RecipeStatus::Running;
            }
        }

        // Initialize records from source texts
        let mut records: Vec<Record> = source_texts
            .iter()
            .map(|(name, text)| Record {
                text: text.to_string(),
                metadata: [("source".to_string(), name.to_string())].into(),
            })
            .collect();

        // Execute pipeline steps
        for step in &recipe.steps {
            records = execute_step(step, records)?;
        }

        let seq = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let result = DatasetResult {
            dataset_id: format!("ds-{}-{seq}", epoch_secs()),
            recipe_id: recipe_id.to_string(),
            record_count: records.len(),
            records,
        };

        // Store dataset
        if let Ok(mut store) = self.datasets.write() {
            store.insert(result.dataset_id.clone(), result.clone());
        }

        // Update status
        if let Ok(mut store) = self.recipes.write() {
            if let Some(r) = store.get_mut(recipe_id) {
                r.status = RecipeStatus::Complete;
            }
        }

        Ok(result)
    }

    /// Get a dataset by ID.
    #[must_use]
    pub fn get_dataset(&self, id: &str) -> Option<DatasetResult> {
        self.datasets.read().unwrap_or_else(|e| e.into_inner()).get(id).cloned()
    }

    /// List all datasets.
    #[must_use]
    pub fn list_datasets(&self) -> Vec<DatasetResult> {
        let store = self.datasets.read().unwrap_or_else(|e| e.into_inner());
        store.values().cloned().collect()
    }
}

/// Execute a single pipeline step on a set of records.
fn execute_step(step: &RecipeStep, records: Vec<Record>) -> Result<Vec<Record>, RecipeError> {
    match step.step_type {
        StepType::ExtractText => Ok(records), // Already text — passthrough
        StepType::Chunk => execute_chunk(step, records),
        StepType::Filter => execute_filter(step, records),
        StepType::Format => execute_format(step, records),
        StepType::Deduplicate => execute_dedup(records),
    }
}

/// Chunk: split text into token-aware pieces.
fn execute_chunk(step: &RecipeStep, records: Vec<Record>) -> Result<Vec<Record>, RecipeError> {
    let max_chars =
        step.config.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(512) as usize * 4;
    let overlap = step.config.get("overlap").and_then(|v| v.as_u64()).unwrap_or(64) as usize * 4;

    let mut chunks = Vec::new();
    for record in &records {
        let text = &record.text;
        if text.len() <= max_chars {
            chunks.push(record.clone());
            continue;
        }

        let mut start = 0;
        let mut chunk_idx = 0;
        while start < text.len() {
            let end = (start + max_chars).min(text.len());
            let chunk_text = &text[start..end];
            let mut meta = record.metadata.clone();
            meta.insert("chunk_index".to_string(), chunk_idx.to_string());
            chunks.push(Record { text: chunk_text.to_string(), metadata: meta });
            start = if end == text.len() { end } else { end - overlap };
            chunk_idx += 1;
        }
    }
    Ok(chunks)
}

/// Filter: remove records by min/max length.
fn execute_filter(step: &RecipeStep, records: Vec<Record>) -> Result<Vec<Record>, RecipeError> {
    let min_len = step.config.get("min_length").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let max_len =
        step.config.get("max_length").and_then(|v| v.as_u64()).unwrap_or(u64::MAX) as usize;

    Ok(records.into_iter().filter(|r| r.text.len() >= min_len && r.text.len() <= max_len).collect())
}

/// Format: apply a chat template to records.
fn execute_format(step: &RecipeStep, records: Vec<Record>) -> Result<Vec<Record>, RecipeError> {
    let template = step.config.get("template").and_then(|v| v.as_str()).unwrap_or("chatml");

    Ok(records
        .into_iter()
        .map(|r| {
            let formatted = match template {
                "chatml" => {
                    format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", r.text)
                }
                "alpaca" => format!("### Instruction:\n{}\n\n### Response:\n", r.text),
                "llama2" => format!("[INST] {} [/INST]", r.text),
                _ => r.text.clone(),
            };
            let mut meta = r.metadata;
            meta.insert("template".to_string(), template.to_string());
            Record { text: formatted, metadata: meta }
        })
        .collect())
}

/// Deduplicate: remove exact duplicate texts.
fn execute_dedup(records: Vec<Record>) -> Result<Vec<Record>, RecipeError> {
    let mut seen = std::collections::HashSet::new();
    Ok(records.into_iter().filter(|r| seen.insert(r.text.clone())).collect())
}

/// Recipe errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecipeError {
    NotFound(String),
}

impl std::fmt::Display for RecipeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Recipe not found: {id}"),
        }
    }
}

impl std::error::Error for RecipeError {}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}
