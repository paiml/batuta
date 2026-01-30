//! Oracle command implementations
//!
//! This module contains all Oracle-related CLI command types.

#![cfg(feature = "native")]

/// Oracle output format
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum OracleOutputFormat {
    /// Human-readable text output
    #[default]
    Text,
    /// JSON output
    Json,
    /// Markdown output
    Markdown,
}
