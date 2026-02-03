//! Binary Index Format for RAG
//!
//! Memory-mapped binary format for faster cold start and reduced memory.
//!
//! # Format
//!
//! ```text
//! Header (64 bytes):
//!   magic:      [u8; 4]  "BRAG"
//!   version:    u32      2
//!   doc_count:  u64
//!   term_count: u64
//!   checksum:   [u8; 32] BLAKE3
//!   reserved:   [u8; 8]
//!
//! Term Index (sorted by term):
//!   term_len:     u16
//!   term:         [u8; term_len]
//!   posting_count: u32
//!   postings:     [(doc_id: u32, tf: u16); posting_count]
//!
//! Document Metadata:
//!   path_offset:    u64
//!   content_offset: u64
//!   length:         u32
//!   fingerprint:    [u8; 32]
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

/// Magic number for binary index
pub const MAGIC: [u8; 4] = *b"BRAG";

/// Current format version
pub const VERSION: u32 = 2;

/// Binary index header
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct IndexHeader {
    /// Magic number ("BRAG")
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Number of documents
    pub doc_count: u64,
    /// Number of unique terms
    pub term_count: u64,
    /// BLAKE3 checksum of content
    pub checksum: [u8; 32],
    /// Reserved for future use
    pub reserved: [u8; 8],
}

impl IndexHeader {
    /// Create a new header
    pub fn new(doc_count: u64, term_count: u64, checksum: [u8; 32]) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            doc_count,
            term_count,
            checksum,
            reserved: [0; 8],
        }
    }

    /// Validate header
    pub fn validate(&self) -> Result<(), BinaryIndexError> {
        if self.magic != MAGIC {
            return Err(BinaryIndexError::InvalidMagic);
        }
        if self.version != VERSION {
            return Err(BinaryIndexError::VersionMismatch {
                expected: VERSION,
                found: self.version,
            });
        }
        Ok(())
    }

    /// Serialize to bytes
    #[allow(clippy::wrong_self_convention)]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut bytes = [0u8; 64];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.doc_count.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.term_count.to_le_bytes());
        bytes[24..56].copy_from_slice(&self.checksum);
        bytes[56..64].copy_from_slice(&self.reserved);
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);

        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let doc_count = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let term_count = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);

        let mut checksum = [0u8; 32];
        checksum.copy_from_slice(&bytes[24..56]);

        let mut reserved = [0u8; 8];
        reserved.copy_from_slice(&bytes[56..64]);

        Self {
            magic,
            version,
            doc_count,
            term_count,
            checksum,
            reserved,
        }
    }
}

/// Document metadata entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentEntry {
    /// Path to document
    pub path: String,
    /// Content hash
    pub fingerprint: [u8; 32],
    /// Document length in chars
    pub length: u32,
}

/// Term posting
#[derive(Debug, Clone, Copy)]
pub struct Posting {
    /// Document ID
    pub doc_id: u32,
    /// Term frequency
    pub tf: u16,
}

/// Binary index writer
pub struct BinaryIndexWriter {
    /// Document entries
    documents: Vec<DocumentEntry>,
    /// Term to postings mapping
    terms: HashMap<String, Vec<Posting>>,
}

impl BinaryIndexWriter {
    /// Create a new writer
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
            terms: HashMap::new(),
        }
    }

    /// Add a document
    pub fn add_document(&mut self, path: String, fingerprint: [u8; 32], length: u32) -> u32 {
        let doc_id = self.documents.len() as u32;
        self.documents.push(DocumentEntry {
            path,
            fingerprint,
            length,
        });
        doc_id
    }

    /// Add a term posting
    pub fn add_posting(&mut self, term: &str, doc_id: u32, tf: u16) {
        self.terms
            .entry(term.to_string())
            .or_default()
            .push(Posting { doc_id, tf });
    }

    /// Write to file
    pub fn write_to_file(&self, path: &Path) -> Result<(), BinaryIndexError> {
        let mut file = std::fs::File::create(path)?;

        // Compute checksum
        let checksum = self.compute_checksum();

        // Write header
        let header = IndexHeader::new(
            self.documents.len() as u64,
            self.terms.len() as u64,
            checksum,
        );
        file.write_all(&header.to_bytes())?;

        // Write documents (JSON for simplicity, could be more compact)
        let docs_json = serde_json::to_vec(&self.documents)?;
        file.write_all(&(docs_json.len() as u64).to_le_bytes())?;
        file.write_all(&docs_json)?;

        // Write terms (sorted for binary search)
        let mut sorted_terms: Vec<_> = self.terms.iter().collect();
        sorted_terms.sort_by_key(|(k, _)| *k);

        file.write_all(&(sorted_terms.len() as u64).to_le_bytes())?;
        for (term, postings) in sorted_terms {
            let term_bytes = term.as_bytes();
            file.write_all(&(term_bytes.len() as u16).to_le_bytes())?;
            file.write_all(term_bytes)?;
            file.write_all(&(postings.len() as u32).to_le_bytes())?;
            for posting in postings {
                file.write_all(&posting.doc_id.to_le_bytes())?;
                file.write_all(&posting.tf.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Compute checksum of content
    fn compute_checksum(&self) -> [u8; 32] {
        use super::fingerprint::blake3_hash;

        let mut data = Vec::new();
        for doc in &self.documents {
            data.extend_from_slice(doc.path.as_bytes());
            data.extend_from_slice(&doc.fingerprint);
        }
        for (term, postings) in &self.terms {
            data.extend_from_slice(term.as_bytes());
            for posting in postings {
                data.extend_from_slice(&posting.doc_id.to_le_bytes());
                data.extend_from_slice(&posting.tf.to_le_bytes());
            }
        }
        blake3_hash(&data)
    }
}

impl Default for BinaryIndexWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Binary index reader
pub struct BinaryIndexReader {
    /// Index header
    header: IndexHeader,
    /// Document entries
    documents: Vec<DocumentEntry>,
    /// Term index (sorted)
    terms: Vec<(String, Vec<Posting>)>,
}

impl BinaryIndexReader {
    /// Load from file
    pub fn load(path: &Path) -> Result<Self, BinaryIndexError> {
        let mut file = std::fs::File::open(path)?;

        // Read header
        let mut header_bytes = [0u8; 64];
        file.read_exact(&mut header_bytes)?;
        let header = IndexHeader::from_bytes(&header_bytes);
        header.validate()?;

        // Read documents
        let mut doc_len_bytes = [0u8; 8];
        file.read_exact(&mut doc_len_bytes)?;
        let doc_len = u64::from_le_bytes(doc_len_bytes) as usize;

        let mut docs_json = vec![0u8; doc_len];
        file.read_exact(&mut docs_json)?;
        let documents: Vec<DocumentEntry> = serde_json::from_slice(&docs_json)?;

        // Read terms
        let mut term_count_bytes = [0u8; 8];
        file.read_exact(&mut term_count_bytes)?;
        let term_count = u64::from_le_bytes(term_count_bytes) as usize;

        let mut terms = Vec::with_capacity(term_count);
        for _ in 0..term_count {
            let mut term_len_bytes = [0u8; 2];
            file.read_exact(&mut term_len_bytes)?;
            let term_len = u16::from_le_bytes(term_len_bytes) as usize;

            let mut term_bytes = vec![0u8; term_len];
            file.read_exact(&mut term_bytes)?;
            let term = String::from_utf8(term_bytes).map_err(|_| BinaryIndexError::InvalidUtf8)?;

            let mut posting_count_bytes = [0u8; 4];
            file.read_exact(&mut posting_count_bytes)?;
            let posting_count = u32::from_le_bytes(posting_count_bytes) as usize;

            let mut postings = Vec::with_capacity(posting_count);
            for _ in 0..posting_count {
                let mut doc_id_bytes = [0u8; 4];
                let mut tf_bytes = [0u8; 2];
                file.read_exact(&mut doc_id_bytes)?;
                file.read_exact(&mut tf_bytes)?;
                postings.push(Posting {
                    doc_id: u32::from_le_bytes(doc_id_bytes),
                    tf: u16::from_le_bytes(tf_bytes),
                });
            }

            terms.push((term, postings));
        }

        Ok(Self {
            header,
            documents,
            terms,
        })
    }

    /// Get document by ID
    pub fn get_document(&self, doc_id: u32) -> Option<&DocumentEntry> {
        self.documents.get(doc_id as usize)
    }

    /// Binary search for term postings
    pub fn get_postings(&self, term: &str) -> Option<&[Posting]> {
        match self.terms.binary_search_by_key(&term, |(t, _)| t.as_str()) {
            Ok(idx) => Some(&self.terms[idx].1),
            Err(_) => None,
        }
    }

    /// Document count
    pub fn doc_count(&self) -> usize {
        self.documents.len()
    }

    /// Term count
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }
}

/// Binary index error
#[derive(Debug, thiserror::Error)]
pub enum BinaryIndexError {
    #[error("Invalid magic number")]
    InvalidMagic,

    #[error("Version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: u32, found: u32 },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid UTF-8")]
    InvalidUtf8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_header_roundtrip() {
        let header = IndexHeader::new(100, 5000, [42u8; 32]);
        let bytes = header.to_bytes();
        let parsed = IndexHeader::from_bytes(&bytes);

        assert_eq!(parsed.magic, MAGIC);
        assert_eq!(parsed.version, VERSION);
        assert_eq!(parsed.doc_count, 100);
        assert_eq!(parsed.term_count, 5000);
    }

    #[test]
    fn test_write_and_read() {
        let temp = TempDir::new().unwrap();
        let index_path = temp.path().join("test.brag");

        // Write index
        let mut writer = BinaryIndexWriter::new();
        let doc_id = writer.add_document("test.txt".to_string(), [1u8; 32], 100);
        writer.add_posting("hello", doc_id, 5);
        writer.add_posting("world", doc_id, 3);
        writer.write_to_file(&index_path).unwrap();

        // Read index
        let reader = BinaryIndexReader::load(&index_path).unwrap();
        assert_eq!(reader.doc_count(), 1);
        assert_eq!(reader.term_count(), 2);

        let postings = reader.get_postings("hello").unwrap();
        assert_eq!(postings.len(), 1);
        assert_eq!(postings[0].doc_id, 0);
        assert_eq!(postings[0].tf, 5);
    }

    #[test]
    fn test_header_validation() {
        let mut header = IndexHeader::new(100, 5000, [42u8; 32]);
        assert!(header.validate().is_ok());

        // Invalid magic
        header.magic = *b"XXXX";
        assert!(matches!(
            header.validate(),
            Err(BinaryIndexError::InvalidMagic)
        ));
    }

    #[test]
    fn test_header_version_mismatch() {
        let mut header = IndexHeader::new(100, 5000, [42u8; 32]);
        header.version = 999;
        assert!(matches!(
            header.validate(),
            Err(BinaryIndexError::VersionMismatch { .. })
        ));
    }

    #[test]
    fn test_document_lookup() {
        let temp = TempDir::new().unwrap();
        let index_path = temp.path().join("test.brag");

        let mut writer = BinaryIndexWriter::new();
        writer.add_document("doc1.txt".to_string(), [1u8; 32], 100);
        writer.add_document("doc2.txt".to_string(), [2u8; 32], 200);
        writer.write_to_file(&index_path).unwrap();

        let reader = BinaryIndexReader::load(&index_path).unwrap();

        let doc = reader.get_document(0).unwrap();
        assert_eq!(doc.path, "doc1.txt");
        assert_eq!(doc.length, 100);

        let doc = reader.get_document(1).unwrap();
        assert_eq!(doc.path, "doc2.txt");
        assert_eq!(doc.length, 200);

        assert!(reader.get_document(999).is_none());
    }

    #[test]
    fn test_missing_term_returns_none() {
        let temp = TempDir::new().unwrap();
        let index_path = temp.path().join("test.brag");

        let mut writer = BinaryIndexWriter::new();
        let doc_id = writer.add_document("test.txt".to_string(), [1u8; 32], 100);
        writer.add_posting("exists", doc_id, 1);
        writer.write_to_file(&index_path).unwrap();

        let reader = BinaryIndexReader::load(&index_path).unwrap();
        assert!(reader.get_postings("exists").is_some());
        assert!(reader.get_postings("nonexistent").is_none());
    }

    #[test]
    fn test_multiple_documents_same_term() {
        let temp = TempDir::new().unwrap();
        let index_path = temp.path().join("test.brag");

        let mut writer = BinaryIndexWriter::new();
        let doc1 = writer.add_document("doc1.txt".to_string(), [1u8; 32], 100);
        let doc2 = writer.add_document("doc2.txt".to_string(), [2u8; 32], 200);
        writer.add_posting("common", doc1, 3);
        writer.add_posting("common", doc2, 7);
        writer.write_to_file(&index_path).unwrap();

        let reader = BinaryIndexReader::load(&index_path).unwrap();
        let postings = reader.get_postings("common").unwrap();

        assert_eq!(postings.len(), 2);
        assert_eq!(postings[0].tf, 3);
        assert_eq!(postings[1].tf, 7);
    }
}
