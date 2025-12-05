//! HuggingFace Hub Integration
//!
//! Provides query, pull, and publish capabilities for the HuggingFace ecosystem.
//!
//! ## Commands
//!
//! - `batuta hf search` - Search models, datasets, spaces
//! - `batuta hf pull` - Download from Hub
//! - `batuta hf push` - Publish to Hub
//! - `batuta hf tree` - Visualize HF ecosystem
//! - `batuta hf tree --integration` - Show PAIML-HF integration map
//!
//! ## Security Features
//!
//! - SafeTensors enforcement (--safe-only default)
//! - Secret scanning before push
//! - Rate limit handling with exponential backoff

#![cfg(feature = "native")]

pub mod client;
pub mod tree;
