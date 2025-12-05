//! PAIML Stack Dependency Orchestration
//!
//! This module provides dependency management and coordinated release capabilities
//! for the PAIML (Pragmatic AI Labs) Rust ecosystem.
//!
//! ## Commands
//!
//! - `batuta stack check` - Dependency health analysis
//! - `batuta stack release` - Coordinated multi-crate release
//! - `batuta stack status` - Stack dashboard
//! - `batuta stack sync` - Dependency synchronization
//!
//! ## Toyota Way Principles
//!
//! - **Jidoka**: Pre-flight checks stop broken releases
//! - **Just-in-Time**: Pull-based release ordering
//! - **Heijunka**: Version alignment across stack
//! - **Genchi Genbutsu**: Real-time crates.io verification

#![cfg(feature = "native")]
#![allow(unused_imports)] // Public API re-exports

pub mod types;
pub mod graph;
pub mod crates_io;
pub mod checker;
pub mod releaser;
pub mod tree;

pub use types::*;
pub use graph::DependencyGraph;
pub use crates_io::CratesIoClient;
pub use checker::StackChecker;

/// PAIML stack crate names for identification
pub const PAIML_CRATES: &[&str] = &[
    "trueno",
    "trueno-viz",
    "trueno-db",
    "trueno-graph",
    "trueno-rag",
    "aprender",
    "aprender-shell",
    "aprender-tsp",
    "realizar",
    "renacer",
    "alimentar",
    "entrenar",
    "certeza",
    "batuta",
    "presentar",
    "pacha",
    "repartir",
    "ruchy",
    "decy",
    "depyler",
    "sovereign-ai-stack-book",
];

/// Check if a crate name is part of the PAIML stack
pub fn is_paiml_crate(name: &str) -> bool {
    PAIML_CRATES.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paiml_crate_detection() {
        assert!(is_paiml_crate("trueno"));
        assert!(is_paiml_crate("aprender"));
        assert!(is_paiml_crate("batuta"));
        assert!(!is_paiml_crate("serde"));
        assert!(!is_paiml_crate("tokio"));
        assert!(!is_paiml_crate("arrow"));
    }

    #[test]
    fn test_paiml_crates_count() {
        // Spec says 13+ crates
        assert!(PAIML_CRATES.len() >= 13);
    }
}
