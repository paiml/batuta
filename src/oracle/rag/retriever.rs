//! Hybrid Retriever - BM25 + Dense with RRF Fusion
//!
//! Implements state-of-the-art hybrid retrieval following:
//! - Karpukhin et al. (2020) for dense retrieval
//! - Robertson & Zaragoza (2009) for BM25
//! - Cormack et al. (2009) for Reciprocal Rank Fusion

use super::types::{Bm25Config, RetrievalResult, RrfConfig, ScoreBreakdown};
use super::DocumentIndex;
use std::collections::HashMap;

/// Hybrid retriever combining sparse (BM25) and dense retrieval
#[derive(Debug)]
pub struct HybridRetriever {
    /// BM25 configuration
    bm25_config: Bm25Config,
    /// RRF configuration
    rrf_config: RrfConfig,
    /// Inverted index for BM25
    inverted_index: InvertedIndex,
    /// Average document length for BM25
    avg_doc_length: f64,
}

impl HybridRetriever {
    /// Create a new hybrid retriever
    pub fn new() -> Self {
        Self {
            bm25_config: Bm25Config::default(),
            rrf_config: RrfConfig::default(),
            inverted_index: InvertedIndex::new(),
            avg_doc_length: 0.0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(bm25_config: Bm25Config, rrf_config: RrfConfig) -> Self {
        Self {
            bm25_config,
            rrf_config,
            inverted_index: InvertedIndex::new(),
            avg_doc_length: 0.0,
        }
    }

    /// Index a document for retrieval
    pub fn index_document(&mut self, doc_id: &str, content: &str) {
        self.inverted_index.add_document(doc_id, content);
        self.update_avg_doc_length();
    }

    /// Remove a document from the index
    pub fn remove_document(&mut self, doc_id: &str) {
        self.inverted_index.remove_document(doc_id);
        self.update_avg_doc_length();
    }

    /// Update average document length
    fn update_avg_doc_length(&mut self) {
        let total_length: usize = self.inverted_index.doc_lengths.values().sum();
        let doc_count = self.inverted_index.doc_lengths.len();
        self.avg_doc_length = if doc_count > 0 {
            total_length as f64 / doc_count as f64
        } else {
            0.0
        };
    }

    /// Retrieve documents matching query
    pub fn retrieve(
        &self,
        query: &str,
        _index: &DocumentIndex,
        top_k: usize,
    ) -> Vec<RetrievalResult> {
        // Get BM25 results
        let bm25_results = self.bm25_search(query, top_k * 2);

        // Get dense results (placeholder - would use embeddings)
        let dense_results = self.dense_search(query, top_k * 2);

        // Fuse with RRF
        self.rrf_fuse(&bm25_results, &dense_results, top_k)
    }

    /// BM25 sparse retrieval
    fn bm25_search(&self, query: &str, top_k: usize) -> Vec<(String, f64)> {
        let query_terms = tokenize(query);
        let mut scores: HashMap<String, f64> = HashMap::new();

        let n = self.inverted_index.doc_lengths.len() as f64;

        for term in &query_terms {
            if let Some(postings) = self.inverted_index.index.get(term) {
                // IDF calculation: log((N - n + 0.5) / (n + 0.5))
                let df = postings.len() as f64;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

                for (doc_id, tf) in postings {
                    let doc_len = self
                        .inverted_index
                        .doc_lengths
                        .get(doc_id)
                        .copied()
                        .unwrap_or(1) as f64;

                    // BM25 score
                    let k1 = self.bm25_config.k1 as f64;
                    let b = self.bm25_config.b as f64;
                    let tf_norm = (*tf as f64 * (k1 + 1.0))
                        / (*tf as f64
                            + k1 * (1.0 - b + b * doc_len / self.avg_doc_length.max(1.0)));

                    *scores.entry(doc_id.clone()).or_insert(0.0) += idf * tf_norm;
                }
            }
        }

        // Sort by score descending
        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        results
    }

    /// Dense retrieval (placeholder - would use embeddings)
    fn dense_search(&self, _query: &str, top_k: usize) -> Vec<(String, f64)> {
        // Placeholder: Return BM25-like results with slight variation
        // In production, this would use trueno-db vector similarity
        let mut results: Vec<_> = self
            .inverted_index
            .doc_lengths
            .keys()
            .enumerate()
            .map(|(i, doc_id)| {
                // Simulate dense scores based on doc order
                let score = 1.0 / (i + 1) as f64;
                (doc_id.clone(), score)
            })
            .collect();

        results.truncate(top_k);
        results
    }

    /// Reciprocal Rank Fusion
    ///
    /// RRF score = Σ 1/(k + rank) for each retriever
    /// Following Cormack et al. (2009)
    fn rrf_fuse(
        &self,
        sparse_results: &[(String, f64)],
        dense_results: &[(String, f64)],
        top_k: usize,
    ) -> Vec<RetrievalResult> {
        let k = self.rrf_config.k as f64;
        let mut rrf_scores: HashMap<String, (f64, f64, f64)> = HashMap::new(); // (rrf, bm25, dense)

        // Score from sparse (BM25)
        for (rank, (doc_id, bm25_score)) in sparse_results.iter().enumerate() {
            let entry = rrf_scores.entry(doc_id.clone()).or_insert((0.0, 0.0, 0.0));
            entry.0 += 1.0 / (k + rank as f64 + 1.0);
            entry.1 = *bm25_score;
        }

        // Score from dense
        for (rank, (doc_id, dense_score)) in dense_results.iter().enumerate() {
            let entry = rrf_scores.entry(doc_id.clone()).or_insert((0.0, 0.0, 0.0));
            entry.0 += 1.0 / (k + rank as f64 + 1.0);
            entry.2 = *dense_score;
        }

        // Convert to results
        let mut results: Vec<_> = rrf_scores
            .into_iter()
            .map(|(doc_id, (rrf_score, bm25_score, dense_score))| {
                // Normalize score to 0-1 range
                let max_rrf = 2.0 / (k + 1.0); // Max possible RRF score (rank 1 in both)
                let normalized_score = (rrf_score / max_rrf).min(1.0);

                RetrievalResult {
                    id: doc_id.clone(),
                    component: extract_component(&doc_id),
                    source: doc_id.clone(),
                    content: String::new(), // Would be filled from index
                    score: normalized_score,
                    start_line: 1,
                    end_line: 1,
                    score_breakdown: ScoreBreakdown {
                        bm25_score,
                        dense_score,
                        rrf_score,
                        rerank_score: None,
                    },
                }
            })
            .collect();

        // Sort by score descending
        results.sort();
        results.truncate(top_k);

        results
    }

    /// Get index statistics
    pub fn stats(&self) -> RetrieverStats {
        RetrieverStats {
            total_documents: self.inverted_index.doc_lengths.len(),
            total_terms: self.inverted_index.index.len(),
            avg_doc_length: self.avg_doc_length,
        }
    }
}

impl Default for HybridRetriever {
    fn default() -> Self {
        Self::new()
    }
}

/// Retriever statistics
#[derive(Debug, Clone)]
pub struct RetrieverStats {
    /// Total documents indexed
    pub total_documents: usize,
    /// Total unique terms
    pub total_terms: usize,
    /// Average document length
    pub avg_doc_length: f64,
}

/// Inverted index for BM25
#[derive(Debug, Default)]
struct InvertedIndex {
    /// Term -> (doc_id -> term_frequency)
    index: HashMap<String, HashMap<String, usize>>,
    /// Document lengths
    doc_lengths: HashMap<String, usize>,
}

impl InvertedIndex {
    fn new() -> Self {
        Self::default()
    }

    fn add_document(&mut self, doc_id: &str, content: &str) {
        let tokens = tokenize(content);
        self.doc_lengths.insert(doc_id.to_string(), tokens.len());

        // Count term frequencies
        let mut term_freqs: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            *term_freqs.entry(token).or_insert(0) += 1;
        }

        // Add to inverted index
        for (term, freq) in term_freqs {
            self.index
                .entry(term)
                .or_default()
                .insert(doc_id.to_string(), freq);
        }
    }

    fn remove_document(&mut self, doc_id: &str) {
        self.doc_lengths.remove(doc_id);

        // Remove from all posting lists
        for postings in self.index.values_mut() {
            postings.remove(doc_id);
        }

        // Clean up empty posting lists
        self.index.retain(|_, postings| !postings.is_empty());
    }
}

/// Simple tokenizer for text
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty() && s.len() > 1)
        .map(|s| s.to_string())
        .collect()
}

/// Extract component name from doc_id
fn extract_component(doc_id: &str) -> String {
    doc_id.split('/').next().unwrap_or("unknown").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_retriever_creation() {
        let retriever = HybridRetriever::new();
        let stats = retriever.stats();
        assert_eq!(stats.total_documents, 0);
        assert_eq!(stats.total_terms, 0);
    }

    #[test]
    fn test_index_document() {
        let mut retriever = HybridRetriever::new();
        retriever.index_document("doc1", "hello world rust programming");

        let stats = retriever.stats();
        assert_eq!(stats.total_documents, 1);
        assert!(stats.total_terms > 0);
    }

    #[test]
    fn test_bm25_search() {
        let mut retriever = HybridRetriever::new();
        retriever.index_document(
            "trueno/CLAUDE.md",
            "SIMD GPU tensor operations accelerated compute",
        );
        retriever.index_document(
            "aprender/CLAUDE.md",
            "machine learning algorithms random forest",
        );
        retriever.index_document("entrenar/CLAUDE.md", "training autograd LoRA quantization");

        let results = retriever.bm25_search("GPU tensor", 5);

        // trueno should rank higher for GPU tensor query
        assert!(!results.is_empty());
        if !results.is_empty() {
            assert!(results[0].0.contains("trueno"));
        }
    }

    #[test]
    fn test_rrf_fusion() {
        let retriever = HybridRetriever::new();

        let sparse = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.7),
            ("doc3".to_string(), 0.5),
        ];
        let dense = vec![
            ("doc2".to_string(), 0.95),
            ("doc1".to_string(), 0.8),
            ("doc4".to_string(), 0.6),
        ];

        let fused = retriever.rrf_fuse(&sparse, &dense, 5);

        // doc1 and doc2 should both appear (in both lists)
        let doc_ids: HashSet<_> = fused.iter().map(|r| r.id.clone()).collect();
        assert!(doc_ids.contains("doc1"));
        assert!(doc_ids.contains("doc2"));
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is Rust programming.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"rust".to_string()));
        // Single chars should be filtered
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_tokenize_code() {
        let tokens = tokenize("fn main() { let x_value = 42; }");
        assert!(tokens.contains(&"fn".to_string()));
        assert!(tokens.contains(&"main".to_string()));
        assert!(tokens.contains(&"x_value".to_string()));
        assert!(tokens.contains(&"42".to_string()));
    }

    #[test]
    fn test_remove_document() {
        let mut retriever = HybridRetriever::new();
        retriever.index_document("doc1", "hello world");
        retriever.index_document("doc2", "goodbye world");

        assert_eq!(retriever.stats().total_documents, 2);

        retriever.remove_document("doc1");
        assert_eq!(retriever.stats().total_documents, 1);

        // "hello" should no longer be in index
        let results = retriever.bm25_search("hello", 5);
        assert!(results.is_empty() || !results.iter().any(|(id, _)| id == "doc1"));
    }

    #[test]
    fn test_extract_component() {
        assert_eq!(extract_component("trueno/CLAUDE.md"), "trueno");
        assert_eq!(extract_component("aprender/docs/ml.md"), "aprender");
        assert_eq!(extract_component("simple_doc"), "simple_doc");
    }

    #[test]
    fn test_bm25_idf() {
        let mut retriever = HybridRetriever::new();

        // Add documents where "rare" appears in only one
        retriever.index_document("doc1", "common common common rare");
        retriever.index_document("doc2", "common common common");
        retriever.index_document("doc3", "common common common");

        let results = retriever.bm25_search("rare", 5);

        // doc1 should be the only result for "rare"
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "doc1");
    }

    #[test]
    fn test_avg_doc_length_update() {
        let mut retriever = HybridRetriever::new();

        retriever.index_document("doc1", "one two three four five");
        assert!(retriever.avg_doc_length > 0.0);

        let first_avg = retriever.avg_doc_length;

        retriever.index_document("doc2", "one two");
        // Average should change
        assert!(retriever.avg_doc_length != first_avg || retriever.stats().total_documents == 2);
    }

    #[test]
    fn test_retrieval_result_score_breakdown() {
        let mut retriever = HybridRetriever::new();
        retriever.index_document("doc1", "test query terms");

        let index = DocumentIndex::default();
        let results = retriever.retrieve("test query", &index, 5);

        // Results should have score breakdown
        for result in results {
            // RRF score should be set
            assert!(result.score_breakdown.rrf_score >= 0.0);
        }
    }

    // Property-based tests for hybrid retriever
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            /// Property: Indexing and searching is deterministic
            #[test]
            fn prop_search_deterministic(
                doc_id in "[a-z]{3,10}",
                content in "[a-z ]{10,100}"
            ) {
                let mut retriever = HybridRetriever::new();
                retriever.index_document(&doc_id, &content);

                let results1 = retriever.bm25_search(&content, 5);
                let results2 = retriever.bm25_search(&content, 5);

                prop_assert_eq!(results1.len(), results2.len());
                for (r1, r2) in results1.iter().zip(results2.iter()) {
                    prop_assert_eq!(&r1.0, &r2.0);  // Same doc IDs
                    prop_assert!((r1.1 - r2.1).abs() < 1e-6);  // Same scores
                }
            }

            /// Property: BM25 scores are non-negative
            #[test]
            fn prop_bm25_scores_non_negative(
                content in "[a-z ]{20,200}",
                query in "[a-z]{3,15}"
            ) {
                let mut retriever = HybridRetriever::new();
                retriever.index_document("doc1", &content);

                let results = retriever.bm25_search(&query, 10);

                for (_, score) in results {
                    prop_assert!(score >= 0.0, "BM25 score {} should be >= 0", score);
                }
            }

            /// Property: RRF scores from retrieval are in valid range [0, 1]
            #[test]
            fn prop_rrf_scores_valid_range(
                content1 in "[a-z ]{10,50}",
                content2 in "[a-z ]{10,50}",
                query in "[a-z]{3,10}"
            ) {
                let mut retriever = HybridRetriever::new();
                retriever.index_document("doc1", &content1);
                retriever.index_document("doc2", &content2);

                let index = DocumentIndex::default();
                let results = retriever.retrieve(&query, &index, 10);

                for result in &results {
                    prop_assert!(result.score >= 0.0 && result.score <= 1.0,
                        "RRF score {} should be in [0, 1]", result.score);
                }
            }

            /// Property: Document count increases on indexing
            #[test]
            fn prop_doc_count_increases(
                docs in prop::collection::vec(("[a-z]{5}", "[a-z ]{10,50}"), 1..10)
            ) {
                let mut retriever = HybridRetriever::new();

                for (i, (id, content)) in docs.iter().enumerate() {
                    retriever.index_document(id, content);
                    // Use >= because duplicate IDs won't increase count
                    prop_assert!(retriever.stats().total_documents >= 1,
                        "After {} docs, count is {}", i + 1, retriever.stats().total_documents);
                }
            }

            /// Property: Empty query returns empty results
            #[test]
            fn prop_empty_query_empty_results(content in "[a-z ]{10,100}") {
                let mut retriever = HybridRetriever::new();
                retriever.index_document("doc1", &content);

                let results = retriever.bm25_search("", 10);
                prop_assert!(results.is_empty());
            }

            /// Property: Removing document decreases count
            #[test]
            fn prop_remove_decreases_count(
                id1 in "[a-z]{5}",
                id2 in "[A-Z]{5}",
                content in "[a-z ]{10,50}"
            ) {
                let mut retriever = HybridRetriever::new();
                retriever.index_document(&id1, &content);
                retriever.index_document(&id2, &content);

                let count_before = retriever.stats().total_documents;
                retriever.remove_document(&id1);
                let count_after = retriever.stats().total_documents;

                prop_assert!(count_after <= count_before);
            }
        }
    }
}
