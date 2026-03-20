//! Model slot — tracks which model is loaded in Banco.
//!
//! Phase 2a: metadata-only (path, format, size, loaded_at).
//! Phase 2b: holds `Arc<OwnedQuantizedModel>` for actual inference.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::Instant;

/// Detected model format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Gguf,
    Apr,
    SafeTensors,
    Unknown,
}

impl ModelFormat {
    /// Detect format from file extension.
    #[must_use]
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("gguf") => Self::Gguf,
            Some("apr") => Self::Apr,
            Some("safetensors") => Self::SafeTensors,
            _ => Self::Unknown,
        }
    }
}

/// Metadata about a loaded model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSlotInfo {
    pub model_id: String,
    pub path: String,
    pub format: ModelFormat,
    pub size_bytes: u64,
    pub loaded_at_secs: u64,
}

/// Model slot — holds the currently loaded model (or None).
pub struct ModelSlot {
    info: RwLock<Option<ModelSlotInfo>>,
    loaded_at: RwLock<Option<Instant>>,
}

impl ModelSlot {
    /// Create an empty slot.
    #[must_use]
    pub fn empty() -> Self {
        Self { info: RwLock::new(None), loaded_at: RwLock::new(None) }
    }

    /// Load a model from a path. Phase 2a: just records metadata.
    pub fn load(&self, path: &str) -> Result<ModelSlotInfo, ModelSlotError> {
        let pb = PathBuf::from(path);

        // Resolve model ID from filename
        let model_id = pb.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();

        let format = ModelFormat::from_path(&pb);

        // Get file size (0 if file doesn't exist — allows pacha:// URIs)
        let size_bytes = std::fs::metadata(&pb).map(|m| m.len()).unwrap_or(0);

        let info = ModelSlotInfo {
            model_id,
            path: path.to_string(),
            format,
            size_bytes,
            loaded_at_secs: epoch_secs(),
        };

        if let Ok(mut slot) = self.info.write() {
            *slot = Some(info.clone());
        }
        if let Ok(mut t) = self.loaded_at.write() {
            *t = Some(Instant::now());
        }

        Ok(info)
    }

    /// Unload the current model.
    pub fn unload(&self) -> Result<(), ModelSlotError> {
        let had_model = self.info.write().map(|mut s| s.take().is_some()).unwrap_or(false);
        if let Ok(mut t) = self.loaded_at.write() {
            *t = None;
        }
        if had_model {
            Ok(())
        } else {
            Err(ModelSlotError::NoModelLoaded)
        }
    }

    /// Get current model info (None if empty).
    #[must_use]
    pub fn info(&self) -> Option<ModelSlotInfo> {
        self.info.read().ok()?.clone()
    }

    /// Check if a model is loaded.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.info.read().map(|s| s.is_some()).unwrap_or(false)
    }

    /// How long the model has been loaded.
    #[must_use]
    pub fn uptime_secs(&self) -> u64 {
        self.loaded_at.read().ok().and_then(|t| t.map(|i| i.elapsed().as_secs())).unwrap_or(0)
    }
}

/// Model slot errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSlotError {
    NoModelLoaded,
}

impl std::fmt::Display for ModelSlotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoModelLoaded => write!(f, "No model loaded"),
        }
    }
}

impl std::error::Error for ModelSlotError {}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}
