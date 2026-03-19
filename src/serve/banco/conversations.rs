//! Conversation persistence for Banco.
//!
//! Stores conversations as JSONL files in `~/.banco/conversations/`.
//! Each conversation has an ID, title, and append-only message log.

use crate::serve::templates::ChatMessage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Metadata for a single conversation (stored in index).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMeta {
    pub id: String,
    pub title: String,
    pub model: String,
    pub created: u64,
    pub updated: u64,
    pub message_count: usize,
}

/// A full conversation with messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub meta: ConversationMeta,
    pub messages: Vec<ChatMessage>,
}

/// In-memory conversation store with optional disk persistence.
pub struct ConversationStore {
    conversations: RwLock<HashMap<String, Conversation>>,
    data_dir: Option<PathBuf>,
    counter: std::sync::atomic::AtomicU64,
}

impl ConversationStore {
    /// Create an in-memory-only store (for testing).
    #[must_use]
    pub fn in_memory() -> Arc<Self> {
        Arc::new(Self {
            conversations: RwLock::new(HashMap::new()),
            data_dir: None,
            counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Create a store backed by `~/.banco/conversations/`.
    #[must_use]
    pub fn with_data_dir(dir: PathBuf) -> Arc<Self> {
        let _ = std::fs::create_dir_all(&dir);
        Arc::new(Self {
            conversations: RwLock::new(HashMap::new()),
            data_dir: Some(dir),
            counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Create a new conversation, returning its ID.
    pub fn create(&self, model: &str) -> String {
        let seq = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let id = format!("conv-{}-{seq}", epoch_secs());
        let meta = ConversationMeta {
            id: id.clone(),
            title: "New conversation".to_string(),
            model: model.to_string(),
            created: epoch_secs(),
            updated: epoch_secs(),
            message_count: 0,
        };
        let conv = Conversation { meta, messages: Vec::new() };
        if let Ok(mut store) = self.conversations.write() {
            store.insert(id.clone(), conv);
        }
        id
    }

    /// Append a message to a conversation. Auto-titles on first user message.
    pub fn append(&self, id: &str, message: ChatMessage) -> Result<(), ConversationError> {
        let mut store = self.conversations.write().map_err(|_| ConversationError::LockPoisoned)?;
        let conv = store.get_mut(id).ok_or(ConversationError::NotFound(id.to_string()))?;

        // Auto-title from first user message
        if conv.messages.is_empty()
            && conv.meta.title == "New conversation"
            && matches!(message.role, crate::serve::templates::Role::User)
        {
            conv.meta.title = auto_title(&message.content);
        }

        conv.messages.push(message);
        conv.meta.message_count = conv.messages.len();
        conv.meta.updated = epoch_secs();

        // Persist to disk if configured
        if let Some(ref dir) = self.data_dir {
            let path = dir.join(format!("{id}.jsonl"));
            let json = serde_json::to_string(&conv.messages.last().expect("just pushed"))
                .unwrap_or_default();
            let _ = std::fs::OpenOptions::new().create(true).append(true).open(path).and_then(
                |mut f| {
                    use std::io::Write;
                    writeln!(f, "{json}")
                },
            );
        }

        Ok(())
    }

    /// List all conversations (most recent first).
    #[must_use]
    pub fn list(&self) -> Vec<ConversationMeta> {
        let store = self.conversations.read().unwrap_or_else(|e| e.into_inner());
        let mut metas: Vec<ConversationMeta> = store.values().map(|c| c.meta.clone()).collect();
        metas.sort_by(|a, b| b.updated.cmp(&a.updated));
        metas
    }

    /// Get a conversation by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<Conversation> {
        let store = self.conversations.read().unwrap_or_else(|e| e.into_inner());
        store.get(id).cloned()
    }

    /// Delete a conversation by ID.
    pub fn delete(&self, id: &str) -> Result<(), ConversationError> {
        let mut store = self.conversations.write().map_err(|_| ConversationError::LockPoisoned)?;
        store.remove(id).ok_or(ConversationError::NotFound(id.to_string()))?;

        // Remove from disk
        if let Some(ref dir) = self.data_dir {
            let _ = std::fs::remove_file(dir.join(format!("{id}.jsonl")));
        }

        Ok(())
    }

    /// Number of conversations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.conversations.read().map(|s| s.len()).unwrap_or(0)
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Auto-generate a title from the first user message (first 5 words).
fn auto_title(content: &str) -> String {
    let words: Vec<&str> = content.split_whitespace().take(5).collect();
    if words.is_empty() {
        "New conversation".to_string()
    } else {
        let mut title = words.join(" ");
        if content.split_whitespace().count() > 5 {
            title.push_str("...");
        }
        title
    }
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// Conversation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationError {
    NotFound(String),
    LockPoisoned,
}

impl std::fmt::Display for ConversationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Conversation not found: {id}"),
            Self::LockPoisoned => write!(f, "Internal lock error"),
        }
    }
}

impl std::error::Error for ConversationError {}
