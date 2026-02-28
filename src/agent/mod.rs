//! Autonomous Agent Runtime (perceive-reason-act loop).
//!
//! Implements a sovereign agent that uses local LLM inference
//! (realizar), RAG retrieval (trueno-rag), and persistent memory
//! (trueno-db) — all running locally with zero API dependencies.
//!
//! # Architecture
//!
//! ```text
//! AgentManifest (TOML)
//!   → PERCEIVE: recall memories
//!   → REASON:   LlmDriver.complete()
//!   → ACT:      Tool.execute()
//!   → repeat until Done or guard triggers
//! ```
//!
//! # Toyota Production System Principles
//!
//! - **Jidoka**: LoopGuard stops on ping-pong, budget, max iterations
//! - **Poka-Yoke**: Capability system prevents unauthorized tool access
//! - **Muda**: CostCircuitBreaker prevents runaway spend
//! - **Genchi Genbutsu**: Default sovereign — local hardware, no proxies
//!
//! # References
//!
//! - arXiv:2512.10350 — Geometric dynamics of agentic loops
//! - arXiv:2501.09136 — Agentic RAG survey
//! - arXiv:2406.09187 — GuardAgent safety

pub mod capability;
pub mod driver;
pub mod guard;
pub mod manifest;
pub mod memory;
pub mod phase;
pub mod result;
pub mod runtime;
pub mod tool;

// Re-export key types for convenience.
pub use capability::{capability_matches, Capability};
pub use guard::{LoopGuard, LoopVerdict};
pub use manifest::{AgentManifest, ModelConfig, ResourceQuota};
pub use phase::LoopPhase;
pub use result::{
    AgentError, AgentLoopResult, DriverError, StopReason, TokenUsage,
};
