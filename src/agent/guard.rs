//! Loop guard — prevents runaway agent loops (Jidoka pattern).
//!
//! The `LoopGuard` tracks iteration count, tool call hashes, cost,
//! and consecutive `MaxTokens` responses. It can `Allow`, `Warn`,
//! `Block` (single call), or `CircuitBreak` (terminate loop).
//!
//! Ping-pong detection uses `FxHash` (64-bit) on `(tool_name, input)`.
//! Theoretically grounded: Tacheny (arXiv:2512.10350) classifies
//! agentic loop dynamics as contractive, oscillatory, or exploratory.
//! Ping-pong detection identifies the oscillatory regime.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::result::TokenUsage;

/// Prevents runaway agent loops (Jidoka pattern).
pub struct LoopGuard {
    max_iterations: u32,
    current_iteration: u32,
    max_tool_calls: u32,
    total_tool_calls: u32,
    /// `FxHash` of `(tool_name, input)` → occurrence count.
    tool_call_counts: HashMap<u64, u32>,
    /// Consecutive `MaxTokens` responses (Jidoka: stop on repeated truncation).
    consecutive_max_tokens: u32,
    /// Accumulated token usage across all iterations.
    usage: TokenUsage,
    /// Maximum cost in USD (0.0 = unlimited sovereign).
    max_cost_usd: f64,
    /// Accumulated estimated cost in USD.
    accumulated_cost_usd: f64,
}

/// Verdict from the loop guard on whether to proceed.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopVerdict {
    /// Proceed with execution.
    Allow,
    /// Proceed but warn (approaching limits).
    Warn(String),
    /// Block this specific tool call (repeated pattern).
    Block(String),
    /// Hard stop the entire loop.
    CircuitBreak(String),
}

/// Configuration for ping-pong detection thresholds.
const PINGPONG_THRESHOLD: u32 = 3;
/// Maximum consecutive `MaxTokens` before circuit break.
const MAX_CONSECUTIVE_TRUNCATION: u32 = 5;
/// Warn when iteration count reaches this fraction of max.
const WARN_ITERATION_FRACTION: f64 = 0.8;

impl LoopGuard {
    /// Create a new guard from resource quotas.
    pub fn new(
        max_iterations: u32,
        max_tool_calls: u32,
        max_cost_usd: f64,
    ) -> Self {
        Self {
            max_iterations,
            current_iteration: 0,
            max_tool_calls,
            total_tool_calls: 0,
            tool_call_counts: HashMap::new(),
            consecutive_max_tokens: 0,
            usage: TokenUsage::default(),
            max_cost_usd,
            accumulated_cost_usd: 0.0,
        }
    }

    /// Check if another iteration is allowed.
    pub fn check_iteration(&mut self) -> LoopVerdict {
        self.current_iteration += 1;

        if self.current_iteration > self.max_iterations {
            return LoopVerdict::CircuitBreak(format!(
                "max iterations reached ({})",
                self.max_iterations
            ));
        }

        // Precision loss acceptable: max_iterations is small enough for f64
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_lossless)]
        let threshold =
            (self.max_iterations as f64 * WARN_ITERATION_FRACTION) as u32;
        if self.current_iteration >= threshold {
            return LoopVerdict::Warn(format!(
                "iteration {}/{} ({}% of budget)",
                self.current_iteration,
                self.max_iterations,
                self.current_iteration * 100 / self.max_iterations
            ));
        }

        LoopVerdict::Allow
    }

    /// Check if a tool call is allowed (ping-pong detection).
    pub fn check_tool_call(
        &mut self,
        tool_name: &str,
        input: &serde_json::Value,
    ) -> LoopVerdict {
        self.total_tool_calls += 1;

        if self.total_tool_calls > self.max_tool_calls {
            return LoopVerdict::CircuitBreak(format!(
                "max tool calls reached ({})",
                self.max_tool_calls
            ));
        }

        let hash = fx_hash_tool_call(tool_name, input);
        let count = self.tool_call_counts.entry(hash).or_insert(0);
        *count += 1;

        if *count >= PINGPONG_THRESHOLD {
            return LoopVerdict::Block(format!(
                "ping-pong detected: tool '{tool_name}' called \
                 {count} times with same input"
            ));
        }

        LoopVerdict::Allow
    }

    /// Record a `MaxTokens` stop reason. Returns `CircuitBreak` if
    /// consecutive truncations exceed threshold.
    pub fn record_max_tokens(&mut self) -> LoopVerdict {
        self.consecutive_max_tokens += 1;
        if self.consecutive_max_tokens >= MAX_CONSECUTIVE_TRUNCATION {
            LoopVerdict::CircuitBreak(format!(
                "{MAX_CONSECUTIVE_TRUNCATION} consecutive MaxTokens responses"
            ))
        } else {
            LoopVerdict::Allow
        }
    }

    /// Reset consecutive `MaxTokens` counter (on `EndTurn` or `ToolUse`).
    pub fn reset_max_tokens(&mut self) {
        self.consecutive_max_tokens = 0;
    }

    /// Record token usage from a completion.
    pub fn record_usage(&mut self, usage: &TokenUsage) {
        self.usage.accumulate(usage);
    }

    /// Record estimated cost and check budget.
    pub fn record_cost(&mut self, cost_usd: f64) -> LoopVerdict {
        self.accumulated_cost_usd += cost_usd;
        if self.max_cost_usd > 0.0
            && self.accumulated_cost_usd > self.max_cost_usd
        {
            LoopVerdict::CircuitBreak(format!(
                "cost budget exceeded: ${:.4} > ${:.4}",
                self.accumulated_cost_usd, self.max_cost_usd
            ))
        } else {
            LoopVerdict::Allow
        }
    }

    /// Get accumulated usage.
    pub fn usage(&self) -> &TokenUsage {
        &self.usage
    }

    /// Get current iteration count.
    pub fn current_iteration(&self) -> u32 {
        self.current_iteration
    }

    /// Get total tool calls made.
    pub fn total_tool_calls(&self) -> u32 {
        self.total_tool_calls
    }
}

/// `FxHash` a tool call for ping-pong detection.
///
/// Uses a simple multiplicative hash (non-cryptographic) — we only
/// need collision resistance across ~50 values, not security.
fn fx_hash_tool_call(tool_name: &str, input: &serde_json::Value) -> u64 {
    let mut hasher = FxHasher::default();
    tool_name.hash(&mut hasher);
    let input_str = input.to_string();
    input_str.hash(&mut hasher);
    hasher.finish()
}

/// Minimal `FxHash` implementation (no external dependency).
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x517c_c1b7_2722_0a95;

impl Hasher for FxHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.hash = (self.hash.rotate_left(5) ^ u64::from(byte))
                .wrapping_mul(FX_SEED);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iteration_limit() {
        let mut guard = LoopGuard::new(5, 100, 0.0);
        for _ in 0..3 {
            assert_eq!(guard.check_iteration(), LoopVerdict::Allow);
        }
        // Iteration 4 of 5 = 80% → Warn
        assert!(matches!(guard.check_iteration(), LoopVerdict::Warn(_)));
        // Iteration 5 of 5 = 100% → still Warn
        assert!(matches!(guard.check_iteration(), LoopVerdict::Warn(_)));
        // Iteration 6 → over limit → CircuitBreak
        assert!(matches!(
            guard.check_iteration(),
            LoopVerdict::CircuitBreak(_)
        ));
    }

    #[test]
    fn test_warn_at_80_percent() {
        let mut guard = LoopGuard::new(10, 100, 0.0);
        for _ in 0..7 {
            assert_eq!(guard.check_iteration(), LoopVerdict::Allow);
        }
        // Iteration 8 of 10 = 80% → Warn
        assert!(matches!(guard.check_iteration(), LoopVerdict::Warn(_)));
    }

    #[test]
    fn test_tool_call_limit() {
        let mut guard = LoopGuard::new(100, 2, 0.0);
        let input = serde_json::json!({"q": "a"});
        assert_eq!(
            guard.check_tool_call("t1", &input),
            LoopVerdict::Allow
        );
        assert_eq!(
            guard.check_tool_call("t2", &serde_json::json!({"q": "b"})),
            LoopVerdict::Allow
        );
        assert!(matches!(
            guard.check_tool_call("t3", &input),
            LoopVerdict::CircuitBreak(_)
        ));
    }

    #[test]
    fn test_pingpong_detection() {
        let mut guard = LoopGuard::new(100, 100, 0.0);
        let input = serde_json::json!({"query": "same"});

        assert_eq!(
            guard.check_tool_call("rag", &input),
            LoopVerdict::Allow
        );
        assert_eq!(
            guard.check_tool_call("rag", &input),
            LoopVerdict::Allow
        );
        // 3rd identical call → Block
        assert!(matches!(
            guard.check_tool_call("rag", &input),
            LoopVerdict::Block(_)
        ));
    }

    #[test]
    fn test_different_inputs_no_pingpong() {
        let mut guard = LoopGuard::new(100, 100, 0.0);
        for i in 0..10 {
            let input = serde_json::json!({"q": format!("query_{i}")});
            assert_eq!(
                guard.check_tool_call("rag", &input),
                LoopVerdict::Allow
            );
        }
    }

    #[test]
    fn test_consecutive_max_tokens() {
        let mut guard = LoopGuard::new(100, 100, 0.0);
        for _ in 0..4 {
            assert_eq!(guard.record_max_tokens(), LoopVerdict::Allow);
        }
        // 5th consecutive → CircuitBreak
        assert!(matches!(
            guard.record_max_tokens(),
            LoopVerdict::CircuitBreak(_)
        ));
    }

    #[test]
    fn test_max_tokens_reset() {
        let mut guard = LoopGuard::new(100, 100, 0.0);
        guard.record_max_tokens();
        guard.record_max_tokens();
        guard.reset_max_tokens();
        // After reset, counter starts over
        for _ in 0..4 {
            assert_eq!(guard.record_max_tokens(), LoopVerdict::Allow);
        }
        assert!(matches!(
            guard.record_max_tokens(),
            LoopVerdict::CircuitBreak(_)
        ));
    }

    #[test]
    fn test_cost_budget() {
        let mut guard = LoopGuard::new(100, 100, 1.0);
        assert_eq!(guard.record_cost(0.5), LoopVerdict::Allow);
        assert_eq!(guard.record_cost(0.3), LoopVerdict::Allow);
        // 0.5 + 0.3 + 0.3 = 1.1 > 1.0 → CircuitBreak
        assert!(matches!(
            guard.record_cost(0.3),
            LoopVerdict::CircuitBreak(_)
        ));
    }

    #[test]
    fn test_zero_cost_budget_unlimited() {
        let mut guard = LoopGuard::new(100, 100, 0.0);
        assert_eq!(guard.record_cost(1000.0), LoopVerdict::Allow);
    }

    #[test]
    fn test_usage_tracking() {
        let mut guard = LoopGuard::new(100, 100, 0.0);
        guard.record_usage(&TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
        });
        guard.record_usage(&TokenUsage {
            input_tokens: 200,
            output_tokens: 75,
        });
        assert_eq!(guard.usage().input_tokens, 300);
        assert_eq!(guard.usage().output_tokens, 125);
    }

    #[test]
    fn test_fx_hash_deterministic() {
        let input = serde_json::json!({"q": "hello"});
        let h1 = fx_hash_tool_call("rag", &input);
        let h2 = fx_hash_tool_call("rag", &input);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fx_hash_different_tools() {
        let input = serde_json::json!({"q": "hello"});
        let h1 = fx_hash_tool_call("rag", &input);
        let h2 = fx_hash_tool_call("memory", &input);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_counters() {
        let mut guard = LoopGuard::new(10, 10, 0.0);
        guard.check_iteration();
        guard.check_iteration();
        assert_eq!(guard.current_iteration(), 2);

        let input = serde_json::json!({});
        guard.check_tool_call("t", &input);
        assert_eq!(guard.total_tool_calls(), 1);
    }
}
