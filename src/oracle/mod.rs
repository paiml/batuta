//! Oracle Mode - Intelligent query interface for the Sovereign AI Stack
//!
//! Provides:
//! - Knowledge graph of stack components
//! - Natural language query interface
//! - Component recommendations
//! - Integration pattern discovery

mod types;
mod knowledge_graph;
mod query_engine;
mod recommender;

// Re-export public API - these are used by library consumers
#[allow(unused_imports)]
pub use types::*;
#[allow(unused_imports)]
pub use knowledge_graph::*;
#[allow(unused_imports)]
pub use query_engine::QueryEngine;
pub use recommender::*;
