//! Banco application state shared across all handlers via `Arc`.

use crate::serve::backends::{BackendSelector, PrivacyTier, ServingBackend};
use crate::serve::circuit_breaker::{CircuitState, CostCircuitBreaker};
use crate::serve::context::ContextManager;
use crate::serve::router::SpilloverRouter;
use crate::serve::templates::ChatTemplateEngine;
use std::sync::Arc;
use std::time::Instant;

use super::conversations::ConversationStore;
use super::prompts::PromptStore;
use super::types::{HealthResponse, ModelInfo, ModelsResponse, SystemResponse};

// ============================================================================
// BANCO-STA-001: State
// ============================================================================

/// Inner state — not `Clone` because of atomics in router/circuit breaker.
pub struct BancoStateInner {
    pub backend_selector: BackendSelector,
    pub router: SpilloverRouter,
    pub circuit_breaker: CostCircuitBreaker,
    pub context_manager: ContextManager,
    pub template_engine: ChatTemplateEngine,
    pub privacy_tier: PrivacyTier,
    pub start_time: Instant,
    pub conversations: Arc<ConversationStore>,
    pub prompts: PromptStore,
}

/// Shared handle passed to axum handlers.
pub type BancoState = Arc<BancoStateInner>;

impl BancoStateInner {
    /// Create default state (Standard privacy, default backends).
    #[must_use]
    pub fn with_defaults() -> BancoState {
        Self::with_privacy(PrivacyTier::Standard)
    }

    /// Create state with a specific privacy tier.
    #[must_use]
    pub fn with_privacy(tier: PrivacyTier) -> BancoState {
        Arc::new(Self {
            backend_selector: BackendSelector::new().with_privacy(tier),
            router: SpilloverRouter::with_defaults(),
            circuit_breaker: CostCircuitBreaker::with_defaults(),
            context_manager: ContextManager::default(),
            template_engine: ChatTemplateEngine::default(),
            privacy_tier: tier,
            start_time: Instant::now(),
            conversations: ConversationStore::in_memory(),
            prompts: PromptStore::new(),
        })
    }

    /// Build a `HealthResponse` snapshot.
    #[must_use]
    pub fn health_status(&self) -> HealthResponse {
        let cb_state = match self.circuit_breaker.state() {
            CircuitState::Closed => "closed",
            CircuitState::Open => "open",
            CircuitState::HalfOpen => "half_open",
        };
        HealthResponse {
            status: "ok".to_string(),
            circuit_breaker_state: cb_state.to_string(),
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }

    /// Build a `ModelsResponse` from recommended backends.
    #[must_use]
    pub fn list_models(&self) -> ModelsResponse {
        let backends = self.backend_selector.recommend();
        let data = backends
            .iter()
            .map(|b| ModelInfo {
                id: format!("{b:?}").to_lowercase(),
                object: "model".to_string(),
                owned_by: "batuta".to_string(),
                local: b.is_local(),
            })
            .collect();
        ModelsResponse { object: "list".to_string(), data }
    }

    /// Build a `SystemResponse`.
    #[must_use]
    pub fn system_info(&self) -> SystemResponse {
        let backends = self.backend_selector.recommend();
        SystemResponse {
            privacy_tier: format!("{:?}", self.privacy_tier),
            backends: backends.iter().map(|b| format!("{b:?}")).collect(),
            gpu_available: backends.contains(&ServingBackend::Realizar),
            version: env!("CARGO_PKG_VERSION").to_string(),
            telemetry: false, // Banco never collects telemetry
        }
    }
}
