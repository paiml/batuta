//! Model slot — tracks which model is loaded in Banco.
//!
//! Phase 2a: metadata-only (path, format, size, loaded_at).
//! Phase 2b: GGUF metadata extraction via realizar (behind `inference` feature).

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
    /// Architecture name (e.g., "llama", "phi2", "qwen2"). Available when inference feature enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// Vocabulary size. Available when inference feature enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<usize>,
    /// Hidden dimension. Available when inference feature enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_dim: Option<usize>,
    /// Number of transformer layers. Available when inference feature enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_layers: Option<usize>,
    /// Context length. Available when inference feature enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<usize>,
    /// Number of tensors in the model file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_count: Option<usize>,
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

    /// Load a model from a path.
    ///
    /// With `inference` feature: parses GGUF metadata (architecture, vocab, layers).
    /// Without: records file metadata only.
    pub fn load(&self, path: &str) -> Result<ModelSlotInfo, ModelSlotError> {
        let pb = PathBuf::from(path);

        let model_id = pb.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
        let format = ModelFormat::from_path(&pb);
        let size_bytes = std::fs::metadata(&pb).map(|m| m.len()).unwrap_or(0);

        // Extract GGUF metadata when inference feature is available
        let gguf_meta = extract_gguf_metadata(&pb, format);

        let info = ModelSlotInfo {
            model_id,
            path: path.to_string(),
            format,
            size_bytes,
            loaded_at_secs: epoch_secs(),
            architecture: gguf_meta.as_ref().map(|m| m.architecture.clone()),
            vocab_size: gguf_meta.as_ref().map(|m| m.vocab_size),
            hidden_dim: gguf_meta.as_ref().map(|m| m.hidden_dim),
            num_layers: gguf_meta.as_ref().map(|m| m.num_layers),
            context_length: gguf_meta.as_ref().map(|m| m.context_length),
            tensor_count: gguf_meta.as_ref().map(|m| m.tensor_count),
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

/// Extracted GGUF metadata (architecture, vocab, etc.).
struct GgufMeta {
    architecture: String,
    vocab_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    context_length: usize,
    tensor_count: usize,
}

/// Extract GGUF metadata from a file. Returns None if not GGUF or parsing fails.
#[cfg(feature = "inference")]
fn extract_gguf_metadata(path: &Path, format: ModelFormat) -> Option<GgufMeta> {
    if format != ModelFormat::Gguf {
        return None;
    }
    let data = std::fs::read(path).ok()?;
    let model = realizar::gguf::GGUFModel::from_bytes(&data).ok()?;
    let config = realizar::gguf::GGUFConfig::from_gguf(&model).ok()?;
    Some(GgufMeta {
        architecture: config.architecture.clone(),
        vocab_size: config.vocab_size,
        hidden_dim: config.hidden_dim,
        num_layers: config.num_layers,
        context_length: config.context_length,
        tensor_count: model.tensors.len(),
    })
}

/// Stub when inference feature is not enabled.
#[cfg(not(feature = "inference"))]
fn extract_gguf_metadata(_path: &Path, _format: ModelFormat) -> Option<GgufMeta> {
    None
}
