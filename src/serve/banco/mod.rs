//! Banco: Local-first AI Workbench HTTP API
//!
//! Phase 1 delivers the API skeleton: health, model listing,
//! chat completions (with SSE streaming), and system info.
//!
//! All endpoints reuse the existing serve module orchestration:
//! `BackendSelector`, `SpilloverRouter`, `CostCircuitBreaker`,
//! `ContextManager`, and `ChatTemplateEngine`.

pub mod config;
mod handlers;
mod middleware;
mod router;
mod server;
pub mod state;
pub mod types;

pub use server::start_server;
pub use state::BancoState;

#[cfg(test)]
#[path = "types_tests.rs"]
mod types_tests;

#[cfg(test)]
#[path = "state_tests.rs"]
mod state_tests;

#[cfg(test)]
#[path = "middleware_tests.rs"]
mod middleware_tests;

#[cfg(test)]
#[path = "handlers_tests.rs"]
mod handlers_tests;

#[cfg(test)]
#[path = "contract_tests.rs"]
mod contract_tests;

#[cfg(test)]
#[path = "p0_tests.rs"]
mod p0_tests;
