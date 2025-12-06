//! Cost Circuit Breaker
//!
//! Implements Toyota Way "Muda Elimination" (Waste Prevention).
//!
//! Prevents runaway API costs by tracking usage and enforcing daily budgets.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ============================================================================
// SERVE-CBR-001: Cost Model
// ============================================================================

/// Token pricing for a backend (per 1M tokens)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TokenPricing {
    /// Input token cost per 1M tokens (USD)
    pub input_per_million: f64,
    /// Output token cost per 1M tokens (USD)
    pub output_per_million: f64,
}

impl TokenPricing {
    /// Create new pricing
    #[must_use]
    pub const fn new(input_per_million: f64, output_per_million: f64) -> Self {
        Self {
            input_per_million,
            output_per_million,
        }
    }

    /// Calculate cost for given token counts
    #[must_use]
    pub fn calculate(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        let input_cost = (input_tokens as f64 / 1_000_000.0) * self.input_per_million;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * self.output_per_million;
        input_cost + output_cost
    }

    /// Known pricing for common models (approximate, as of late 2024)
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        let lower = model.to_lowercase();
        if lower.contains("gpt-4o") {
            Self::new(2.50, 10.00)
        } else if lower.contains("gpt-4-turbo") || lower.contains("gpt-4") {
            Self::new(10.00, 30.00)
        } else if lower.contains("gpt-3.5") {
            Self::new(0.50, 1.50)
        } else if lower.contains("claude-3-opus") {
            Self::new(15.00, 75.00)
        } else if lower.contains("claude-3-sonnet") || lower.contains("claude-3.5") {
            Self::new(3.00, 15.00)
        } else if lower.contains("claude-3-haiku") {
            Self::new(0.25, 1.25)
        } else if lower.contains("llama") || lower.contains("mistral") {
            // Together.ai / Groq pricing for open models
            Self::new(0.20, 0.20)
        } else {
            // Default conservative estimate
            Self::new(1.00, 2.00)
        }
    }
}

impl Default for TokenPricing {
    fn default() -> Self {
        Self::new(1.00, 2.00)
    }
}

// ============================================================================
// SERVE-CBR-002: Usage Tracking
// ============================================================================

/// Usage record for a single request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub timestamp: u64,
    pub backend: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
}

/// Daily usage summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DailyUsage {
    /// Date as YYYY-MM-DD
    pub date: String,
    /// Total input tokens
    pub total_input_tokens: u64,
    /// Total output tokens
    pub total_output_tokens: u64,
    /// Total cost in USD
    pub total_cost_usd: f64,
    /// Request count
    pub request_count: u64,
    /// Usage by model
    pub by_model: HashMap<String, f64>,
}

impl DailyUsage {
    /// Create for today
    #[must_use]
    pub fn today() -> Self {
        Self {
            date: Self::current_date(),
            ..Default::default()
        }
    }

    /// Get current date string
    #[must_use]
    pub fn current_date() -> String {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Simple date calculation (not timezone aware)
        let days = now / 86400;
        let year = 1970 + (days / 365); // Approximate
        let day_of_year = days % 365;
        let month = day_of_year / 30 + 1;
        let day = day_of_year % 30 + 1;
        format!("{}-{:02}-{:02}", year, month.min(12), day.min(31))
    }

    /// Add a usage record
    pub fn add(&mut self, record: &UsageRecord) {
        self.total_input_tokens += record.input_tokens;
        self.total_output_tokens += record.output_tokens;
        self.total_cost_usd += record.cost_usd;
        self.request_count += 1;
        *self.by_model.entry(record.model.clone()).or_insert(0.0) += record.cost_usd;
    }
}

// ============================================================================
// SERVE-CBR-003: Circuit Breaker
// ============================================================================

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CircuitState {
    /// Circuit is closed, requests allowed
    #[default]
    Closed,
    /// Circuit is open, requests blocked
    Open,
    /// Half-open, testing if budget allows
    HalfOpen,
}

/// Cost circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Daily budget in USD
    pub daily_budget_usd: f64,
    /// Warning threshold (percentage of budget)
    pub warning_threshold: f64,
    /// Per-request cost limit (USD)
    pub max_request_cost_usd: f64,
    /// Auto-reset after cooldown period
    pub cooldown_seconds: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            daily_budget_usd: 10.0,    // $10/day default
            warning_threshold: 0.8,    // Warn at 80%
            max_request_cost_usd: 1.0, // Max $1 per request
            cooldown_seconds: 3600,    // 1 hour cooldown
        }
    }
}

impl CircuitBreakerConfig {
    /// Create with specific daily budget
    #[must_use]
    pub fn with_budget(daily_budget_usd: f64) -> Self {
        Self {
            daily_budget_usd,
            ..Default::default()
        }
    }
}

/// Cost circuit breaker
///
/// Thread-safe circuit breaker that tracks API costs and prevents overspending.
pub struct CostCircuitBreaker {
    config: CircuitBreakerConfig,
    /// Accumulated cost in millicents (for atomic operations)
    accumulated_millicents: AtomicU64,
    /// Current date for reset tracking
    current_date: RwLock<String>,
    /// Circuit state
    state: RwLock<CircuitState>,
    /// Time when circuit was opened
    opened_at: RwLock<Option<u64>>,
}

impl CostCircuitBreaker {
    /// Create a new circuit breaker
    #[must_use]
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            accumulated_millicents: AtomicU64::new(0),
            current_date: RwLock::new(DailyUsage::current_date()),
            state: RwLock::new(CircuitState::Closed),
            opened_at: RwLock::new(None),
        }
    }

    /// Create with default config
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }

    /// Check if a request with estimated cost is allowed
    pub fn check(&self, estimated_cost_usd: f64) -> Result<(), CircuitBreakerError> {
        // Reset if new day
        self.maybe_reset_daily();

        // Check per-request limit
        if estimated_cost_usd > self.config.max_request_cost_usd {
            return Err(CircuitBreakerError::RequestTooExpensive {
                estimated: estimated_cost_usd,
                limit: self.config.max_request_cost_usd,
            });
        }

        // Check circuit state
        let state = *self.state.read().expect("circuit breaker state lock poisoned");
        match state {
            CircuitState::Open => {
                // Check if cooldown has passed
                if self.cooldown_elapsed() {
                    *self.state.write().expect("circuit breaker state lock poisoned") = CircuitState::HalfOpen;
                } else {
                    return Err(CircuitBreakerError::BudgetExceeded {
                        spent: self.accumulated_usd(),
                        budget: self.config.daily_budget_usd,
                    });
                }
            }
            CircuitState::HalfOpen | CircuitState::Closed => {}
        }

        // Check if adding this cost would exceed budget
        let current = self.accumulated_usd();
        if current + estimated_cost_usd > self.config.daily_budget_usd {
            *self.state.write().expect("circuit breaker state lock poisoned") = CircuitState::Open;
            *self.opened_at.write().expect("circuit breaker opened_at lock poisoned") = Some(Self::current_timestamp());
            return Err(CircuitBreakerError::BudgetExceeded {
                spent: current,
                budget: self.config.daily_budget_usd,
            });
        }

        Ok(())
    }

    /// Record actual cost after request completes
    pub fn record(&self, actual_cost_usd: f64) {
        let millicents = (actual_cost_usd * 100_000.0) as u64;
        self.accumulated_millicents
            .fetch_add(millicents, Ordering::SeqCst);

        // Check if we've hit the budget
        if self.accumulated_usd() >= self.config.daily_budget_usd {
            *self.state.write().expect("circuit breaker state lock poisoned") = CircuitState::Open;
            *self.opened_at.write().expect("circuit breaker opened_at lock poisoned") = Some(Self::current_timestamp());
        }
    }

    /// Get current accumulated cost in USD
    #[must_use]
    pub fn accumulated_usd(&self) -> f64 {
        self.accumulated_millicents.load(Ordering::SeqCst) as f64 / 100_000.0
    }

    /// Get remaining budget
    #[must_use]
    pub fn remaining_usd(&self) -> f64 {
        (self.config.daily_budget_usd - self.accumulated_usd()).max(0.0)
    }

    /// Get budget utilization percentage
    #[must_use]
    pub fn utilization(&self) -> f64 {
        self.accumulated_usd() / self.config.daily_budget_usd
    }

    /// Check if at warning threshold
    #[must_use]
    pub fn is_warning(&self) -> bool {
        self.utilization() >= self.config.warning_threshold
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> CircuitState {
        *self.state.read().expect("circuit breaker state lock poisoned")
    }

    /// Force reset (for testing or manual override)
    pub fn reset(&self) {
        self.accumulated_millicents.store(0, Ordering::SeqCst);
        *self.state.write().expect("circuit breaker state lock poisoned") = CircuitState::Closed;
        *self.opened_at.write().expect("circuit breaker opened_at lock poisoned") = None;
        *self.current_date.write().expect("circuit breaker current_date lock poisoned") = DailyUsage::current_date();
    }

    fn maybe_reset_daily(&self) {
        let today = DailyUsage::current_date();
        let current = self.current_date.read().expect("circuit breaker current_date lock poisoned").clone();
        if current != today {
            drop(current);
            self.reset();
        }
    }

    fn cooldown_elapsed(&self) -> bool {
        if let Some(opened) = *self.opened_at.read().expect("circuit breaker opened_at lock poisoned") {
            let now = Self::current_timestamp();
            now - opened >= self.config.cooldown_seconds
        } else {
            true
        }
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs()
    }
}

impl Default for CostCircuitBreaker {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Circuit breaker errors
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerError {
    /// Daily budget exceeded
    BudgetExceeded { spent: f64, budget: f64 },
    /// Single request too expensive
    RequestTooExpensive { estimated: f64, limit: f64 },
}

impl std::fmt::Display for CircuitBreakerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BudgetExceeded { spent, budget } => {
                write!(
                    f,
                    "Daily budget exceeded: ${:.2} spent of ${:.2} budget",
                    spent, budget
                )
            }
            Self::RequestTooExpensive { estimated, limit } => {
                write!(
                    f,
                    "Request too expensive: ${:.2} estimated, ${:.2} limit",
                    estimated, limit
                )
            }
        }
    }
}

impl std::error::Error for CircuitBreakerError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    // ========================================================================
    // SERVE-CBR-001: Token Pricing Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_001_pricing_calculate() {
        let pricing = TokenPricing::new(1.0, 2.0); // $1/M input, $2/M output
        let cost = pricing.calculate(1_000_000, 500_000);
        assert!((cost - 2.0).abs() < 0.001); // $1 input + $1 output
    }

    #[test]
    fn test_SERVE_CBR_001_pricing_small_amounts() {
        let pricing = TokenPricing::new(10.0, 30.0); // GPT-4 pricing
        let cost = pricing.calculate(1000, 500);
        // 1000 input = $0.01, 500 output = $0.015
        assert!((cost - 0.025).abs() < 0.001);
    }

    #[test]
    fn test_SERVE_CBR_001_pricing_for_model_gpt4() {
        let pricing = TokenPricing::for_model("gpt-4-turbo");
        assert_eq!(pricing.input_per_million, 10.0);
        assert_eq!(pricing.output_per_million, 30.0);
    }

    #[test]
    fn test_SERVE_CBR_001_pricing_for_model_claude() {
        let pricing = TokenPricing::for_model("claude-3-sonnet");
        assert_eq!(pricing.input_per_million, 3.0);
        assert_eq!(pricing.output_per_million, 15.0);
    }

    #[test]
    fn test_SERVE_CBR_001_pricing_for_model_llama() {
        let pricing = TokenPricing::for_model("llama-3.1-70b");
        assert_eq!(pricing.input_per_million, 0.20);
    }

    #[test]
    fn test_SERVE_CBR_001_pricing_default() {
        let pricing = TokenPricing::default();
        assert_eq!(pricing.input_per_million, 1.0);
        assert_eq!(pricing.output_per_million, 2.0);
    }

    // ========================================================================
    // SERVE-CBR-002: Daily Usage Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_002_daily_usage_add() {
        let mut usage = DailyUsage::today();
        let record = UsageRecord {
            timestamp: 0,
            backend: "openai".to_string(),
            model: "gpt-4".to_string(),
            input_tokens: 1000,
            output_tokens: 500,
            cost_usd: 0.025,
        };
        usage.add(&record);
        assert_eq!(usage.total_input_tokens, 1000);
        assert_eq!(usage.total_output_tokens, 500);
        assert!((usage.total_cost_usd - 0.025).abs() < 0.001);
        assert_eq!(usage.request_count, 1);
    }

    #[test]
    fn test_SERVE_CBR_002_daily_usage_by_model() {
        let mut usage = DailyUsage::today();
        usage.add(&UsageRecord {
            timestamp: 0,
            backend: "openai".to_string(),
            model: "gpt-4".to_string(),
            input_tokens: 1000,
            output_tokens: 500,
            cost_usd: 1.0,
        });
        usage.add(&UsageRecord {
            timestamp: 0,
            backend: "openai".to_string(),
            model: "gpt-3.5".to_string(),
            input_tokens: 1000,
            output_tokens: 500,
            cost_usd: 0.1,
        });
        assert_eq!(usage.by_model.get("gpt-4"), Some(&1.0));
        assert_eq!(usage.by_model.get("gpt-3.5"), Some(&0.1));
    }

    // ========================================================================
    // SERVE-CBR-003: Circuit Breaker Basic Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_003_default_config() {
        let config = CircuitBreakerConfig::default();
        assert_eq!(config.daily_budget_usd, 10.0);
        assert_eq!(config.warning_threshold, 0.8);
    }

    #[test]
    fn test_SERVE_CBR_003_check_allows_under_budget() {
        let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(10.0));
        assert!(cb.check(1.0).is_ok());
    }

    #[test]
    fn test_SERVE_CBR_003_check_blocks_over_budget() {
        let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(1.0));
        cb.record(0.9);
        let result = cb.check(0.2);
        assert!(result.is_err());
    }

    #[test]
    fn test_SERVE_CBR_003_record_accumulates() {
        let cb = CostCircuitBreaker::with_defaults();
        cb.record(1.0);
        cb.record(2.0);
        assert!((cb.accumulated_usd() - 3.0).abs() < 0.001);
    }

    // ========================================================================
    // SERVE-CBR-004: Circuit Breaker State Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_004_initial_state_closed() {
        let cb = CostCircuitBreaker::with_defaults();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_SERVE_CBR_004_opens_on_budget_exceed() {
        let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(1.0));
        cb.record(1.0);
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_SERVE_CBR_004_reset_closes_circuit() {
        let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(1.0));
        cb.record(1.0);
        assert_eq!(cb.state(), CircuitState::Open);
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!((cb.accumulated_usd()).abs() < 0.001);
    }

    // ========================================================================
    // SERVE-CBR-005: Request Limit Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_005_rejects_expensive_request() {
        let config = CircuitBreakerConfig {
            max_request_cost_usd: 0.5,
            ..Default::default()
        };
        let cb = CostCircuitBreaker::new(config);
        let result = cb.check(1.0);
        assert!(matches!(
            result,
            Err(CircuitBreakerError::RequestTooExpensive { .. })
        ));
    }

    #[test]
    fn test_SERVE_CBR_005_allows_cheap_request() {
        let config = CircuitBreakerConfig {
            max_request_cost_usd: 1.0,
            ..Default::default()
        };
        let cb = CostCircuitBreaker::new(config);
        assert!(cb.check(0.5).is_ok());
    }

    // ========================================================================
    // SERVE-CBR-006: Utilization Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_006_utilization_percentage() {
        let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(10.0));
        cb.record(5.0);
        assert!((cb.utilization() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_SERVE_CBR_006_remaining_budget() {
        let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(10.0));
        cb.record(3.0);
        assert!((cb.remaining_usd() - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_SERVE_CBR_006_warning_threshold() {
        let config = CircuitBreakerConfig {
            daily_budget_usd: 10.0,
            warning_threshold: 0.8,
            ..Default::default()
        };
        let cb = CostCircuitBreaker::new(config);
        cb.record(7.0);
        assert!(!cb.is_warning());
        cb.record(1.0);
        assert!(cb.is_warning());
    }

    // ========================================================================
    // SERVE-CBR-007: Error Display Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_007_budget_exceeded_display() {
        let err = CircuitBreakerError::BudgetExceeded {
            spent: 10.5,
            budget: 10.0,
        };
        let msg = err.to_string();
        assert!(msg.contains("10.50"));
        assert!(msg.contains("10.00"));
        assert!(msg.contains("exceeded"));
    }

    #[test]
    fn test_SERVE_CBR_007_request_expensive_display() {
        let err = CircuitBreakerError::RequestTooExpensive {
            estimated: 5.0,
            limit: 1.0,
        };
        let msg = err.to_string();
        assert!(msg.contains("5.00"));
        assert!(msg.contains("1.00"));
        assert!(msg.contains("expensive"));
    }
}
