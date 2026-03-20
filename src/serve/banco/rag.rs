//! Banco RAG (Retrieval-Augmented Generation) pipeline.
//!
//! Indexes uploaded documents into a BM25 inverted index for keyword search.
//! Chat requests with `rag: true` retrieve relevant chunks before generation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

/// A chunk in the RAG index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagChunk {
    pub file_id: String,
    pub file_name: String,
    pub chunk_index: usize,
    pub text: String,
}

/// A search result with relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagResult {
    pub file: String,
    pub chunk: usize,
    pub score: f64,
    pub text: String,
}

/// RAG index status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagStatus {
    pub doc_count: usize,
    pub chunk_count: usize,
    pub indexed: bool,
}

/// BM25-based RAG index for document search.
pub struct RagIndex {
    /// All indexed chunks.
    chunks: RwLock<Vec<RagChunk>>,
    /// Inverted index: term → list of (chunk_idx, term_freq).
    postings: RwLock<HashMap<String, Vec<(usize, u32)>>>,
    /// Document lengths (in terms) per chunk.
    doc_lengths: RwLock<Vec<usize>>,
    /// Set of indexed file IDs.
    indexed_files: RwLock<std::collections::HashSet<String>>,
}

impl Default for RagIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl RagIndex {
    #[must_use]
    pub fn new() -> Self {
        Self {
            chunks: RwLock::new(Vec::new()),
            postings: RwLock::new(HashMap::new()),
            doc_lengths: RwLock::new(Vec::new()),
            indexed_files: RwLock::new(std::collections::HashSet::new()),
        }
    }

    /// Index a document's text, splitting into chunks.
    pub fn index_document(&self, file_id: &str, file_name: &str, text: &str) {
        let chunk_texts = chunk_text(text, 512, 64);

        let mut chunks = self.chunks.write().unwrap_or_else(|e| e.into_inner());
        let mut postings = self.postings.write().unwrap_or_else(|e| e.into_inner());
        let mut doc_lens = self.doc_lengths.write().unwrap_or_else(|e| e.into_inner());

        for (i, chunk_text) in chunk_texts.iter().enumerate() {
            let chunk_idx = chunks.len();
            chunks.push(RagChunk {
                file_id: file_id.to_string(),
                file_name: file_name.to_string(),
                chunk_index: i,
                text: chunk_text.clone(),
            });

            // Tokenize and build postings
            let terms = tokenize(chunk_text);
            let mut term_freqs: HashMap<&str, u32> = HashMap::new();
            for term in &terms {
                *term_freqs.entry(term.as_str()).or_insert(0) += 1;
            }

            for (term, freq) in term_freqs {
                postings.entry(term.to_string()).or_default().push((chunk_idx, freq));
            }

            doc_lens.push(terms.len());
        }

        if let Ok(mut files) = self.indexed_files.write() {
            files.insert(file_id.to_string());
        }
    }

    /// Search the index using BM25 scoring.
    pub fn search(&self, query: &str, top_k: usize, min_score: f64) -> Vec<RagResult> {
        let chunks = self.chunks.read().unwrap_or_else(|e| e.into_inner());
        let postings = self.postings.read().unwrap_or_else(|e| e.into_inner());
        let doc_lens = self.doc_lengths.read().unwrap_or_else(|e| e.into_inner());

        if chunks.is_empty() {
            return Vec::new();
        }

        let n = chunks.len() as f64;
        let avg_dl: f64 = if doc_lens.is_empty() {
            1.0
        } else {
            doc_lens.iter().sum::<usize>() as f64 / doc_lens.len() as f64
        };

        let query_terms = tokenize(query);
        let mut scores: HashMap<usize, f64> = HashMap::new();

        let k1 = 1.2;
        let b = 0.75;

        for term in &query_terms {
            if let Some(posting_list) = postings.get(term.as_str()) {
                let df = posting_list.len() as f64;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

                for &(chunk_idx, tf) in posting_list {
                    let dl = doc_lens.get(chunk_idx).copied().unwrap_or(1) as f64;
                    let tf_norm =
                        (tf as f64 * (k1 + 1.0)) / (tf as f64 + k1 * (1.0 - b + b * dl / avg_dl));
                    *scores.entry(chunk_idx).or_insert(0.0) += idf * tf_norm;
                }
            }
        }

        let mut results: Vec<(usize, f64)> =
            scores.into_iter().filter(|&(_, s)| s >= min_score).collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        results
            .into_iter()
            .filter_map(|(idx, score)| {
                chunks.get(idx).map(|c| RagResult {
                    file: c.file_name.clone(),
                    chunk: c.chunk_index,
                    score,
                    text: c.text.clone(),
                })
            })
            .collect()
    }

    /// Get index status.
    #[must_use]
    pub fn status(&self) -> RagStatus {
        let chunk_count = self.chunks.read().map(|c| c.len()).unwrap_or(0);
        let doc_count = self.indexed_files.read().map(|f| f.len()).unwrap_or(0);
        RagStatus { doc_count, chunk_count, indexed: chunk_count > 0 }
    }

    /// Clear the entire index.
    pub fn clear(&self) {
        if let Ok(mut c) = self.chunks.write() {
            c.clear();
        }
        if let Ok(mut p) = self.postings.write() {
            p.clear();
        }
        if let Ok(mut d) = self.doc_lengths.write() {
            d.clear();
        }
        if let Ok(mut f) = self.indexed_files.write() {
            f.clear();
        }
    }

    /// Check if a file has been indexed.
    #[must_use]
    pub fn is_indexed(&self, file_id: &str) -> bool {
        self.indexed_files.read().map(|f| f.contains(file_id)).unwrap_or(false)
    }
}

/// Split text into overlapping chunks (~token_count * 4 chars each).
fn chunk_text(text: &str, max_tokens: usize, overlap_tokens: usize) -> Vec<String> {
    let max_chars = max_tokens * 4;
    let overlap_chars = overlap_tokens * 4;

    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let end = (start + max_chars).min(text.len());
        chunks.push(text[start..end].to_string());
        if end == text.len() {
            break;
        }
        start = end - overlap_chars;
    }
    chunks
}

/// Simple whitespace + lowercase tokenizer.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| w.len() > 1)
        .collect()
}
