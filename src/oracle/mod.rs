//! Oracle Mode - Intelligent query interface for the Sovereign AI Stack
//!
//! Provides:
//! - Knowledge graph of stack components
//! - Natural language query interface
//! - Component recommendations
//! - Integration pattern discovery
//! - RAG-based documentation retrieval (APR-Powered)

mod knowledge_graph;
mod query_engine;
pub mod rag;
mod recommender;
mod types;

// Re-export public API - these are used by library consumers
#[allow(unused_imports)]
pub use knowledge_graph::*;
#[allow(unused_imports)]
pub use query_engine::QueryEngine;
pub use recommender::*;
#[allow(unused_imports)]
pub use types::*;

// Re-export RAG oracle - will be used by CLI integration
#[allow(unused_imports)]
pub use rag::{RagOracle, RagOracleConfig};
