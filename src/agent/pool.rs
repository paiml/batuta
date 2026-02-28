//! Multi-agent orchestration pool.
//!
//! Manages concurrent agent instances with message passing
//! and fan-out/fan-in patterns. Each agent runs its own
//! perceive-reason-act loop in a separate tokio task.
//!
//! # Toyota Production System Principles
//!
//! - **Heijunka**: Load-level work across agents
//! - **Jidoka**: Each agent has its own `LoopGuard`
//! - **Muda**: Bounded concurrency prevents resource waste

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tracing::{debug, info, warn};

use super::driver::{LlmDriver, StreamEvent};
use super::manifest::AgentManifest;
use super::memory::{InMemorySubstrate, MemorySubstrate};
use super::result::{AgentError, AgentLoopResult};
use super::tool::ToolRegistry;

/// Unique identifier for a spawned agent.
pub type AgentId = u64;

/// Message sent between agents in the pool.
#[derive(Debug, Clone)]
pub struct AgentMessage {
    /// Source agent ID (0 = external/supervisor).
    pub from: AgentId,
    /// Target agent ID.
    pub to: AgentId,
    /// Message payload.
    pub content: String,
}

/// Configuration for a spawned agent.
pub struct SpawnConfig {
    /// Agent manifest.
    pub manifest: AgentManifest,
    /// Query to execute.
    pub query: String,
}

/// Multi-agent orchestration pool.
///
/// Manages concurrent agent instances, each running its own
/// perceive-reason-act loop. Supports fan-out (spawn many) and
/// fan-in (collect results) patterns.
///
/// ```rust,ignore
/// let mut pool = AgentPool::new(driver, 4);
/// pool.spawn(config1).await?;
/// pool.spawn(config2).await?;
/// let results = pool.join_all().await;
/// ```
pub struct AgentPool {
    driver: Arc<dyn LlmDriver>,
    memory: Arc<dyn MemorySubstrate>,
    next_id: AgentId,
    max_concurrent: usize,
    join_set: JoinSet<(AgentId, String, Result<AgentLoopResult, String>)>,
    stream_tx: Option<mpsc::Sender<StreamEvent>>,
}

impl AgentPool {
    /// Create a new agent pool with bounded concurrency.
    pub fn new(
        driver: Arc<dyn LlmDriver>,
        max_concurrent: usize,
    ) -> Self {
        Self {
            driver,
            memory: Arc::new(InMemorySubstrate::new()),
            next_id: 1,
            max_concurrent,
            join_set: JoinSet::new(),
            stream_tx: None,
        }
    }

    /// Set a shared memory substrate for all agents.
    #[must_use]
    pub fn with_memory(
        mut self,
        memory: Arc<dyn MemorySubstrate>,
    ) -> Self {
        self.memory = memory;
        self
    }

    /// Set a stream event channel for pool-level events.
    #[must_use]
    pub fn with_stream(
        mut self,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Self {
        self.stream_tx = Some(tx);
        self
    }

    /// Number of currently active agents.
    pub fn active_count(&self) -> usize {
        self.join_set.len()
    }

    /// Maximum concurrent agents allowed.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Spawn a new agent in the pool.
    ///
    /// Returns the `AgentId` assigned to this agent.
    /// Returns error if pool is at capacity.
    pub fn spawn(
        &mut self,
        config: SpawnConfig,
    ) -> Result<AgentId, AgentError> {
        if self.join_set.len() >= self.max_concurrent {
            return Err(AgentError::CircuitBreak(format!(
                "agent pool at capacity ({}/{})",
                self.join_set.len(),
                self.max_concurrent
            )));
        }

        let id = self.next_id;
        self.next_id += 1;

        let name = config.manifest.name.clone();
        let driver = Arc::clone(&self.driver);
        let memory = Arc::clone(&self.memory);
        let stream_tx = self.stream_tx.clone();

        info!(
            agent_id = id,
            name = %name,
            query_len = config.query.len(),
            "spawning agent"
        );

        self.join_set.spawn(async move {
            let tools = ToolRegistry::new();
            let result = super::runtime::run_agent_loop(
                &config.manifest,
                &config.query,
                driver.as_ref(),
                &tools,
                memory.as_ref(),
                stream_tx,
            )
            .await;

            // Map error to String to avoid Clone requirement
            let mapped = result.map_err(|e| e.to_string());
            (id, name, mapped)
        });

        Ok(id)
    }

    /// Fan-out: spawn multiple agents concurrently.
    ///
    /// Returns a list of `AgentId`s for the spawned agents.
    pub fn fan_out(
        &mut self,
        configs: Vec<SpawnConfig>,
    ) -> Result<Vec<AgentId>, AgentError> {
        let mut ids = Vec::with_capacity(configs.len());
        for config in configs {
            ids.push(self.spawn(config)?);
        }
        Ok(ids)
    }

    /// Fan-in: wait for all active agents to complete.
    ///
    /// Returns results keyed by `AgentId`. Agents that error
    /// are included with their error string.
    pub async fn join_all(
        &mut self,
    ) -> HashMap<AgentId, Result<AgentLoopResult, String>> {
        let mut results = HashMap::new();

        while let Some(outcome) = self.join_set.join_next().await {
            match outcome {
                Ok((id, name, result)) => {
                    debug!(
                        agent_id = id,
                        name = %name,
                        ok = result.is_ok(),
                        "agent completed"
                    );
                    results.insert(id, result);
                }
                Err(e) => {
                    warn!(error = %e, "agent task panicked");
                }
            }
        }

        results
    }

    /// Wait for the next agent to complete.
    ///
    /// Returns `None` if no agents are active.
    pub async fn join_next(
        &mut self,
    ) -> Option<(AgentId, Result<AgentLoopResult, String>)> {
        match self.join_set.join_next().await {
            Some(Ok((id, _name, result))) => Some((id, result)),
            Some(Err(e)) => {
                warn!(error = %e, "agent task panicked");
                None
            }
            None => None,
        }
    }

    /// Abort all running agents.
    pub fn abort_all(&mut self) {
        self.join_set.abort_all();
        info!("all agents aborted");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::driver::mock::MockDriver;
    use crate::agent::result::{StopReason, TokenUsage};
    use crate::agent::driver::CompletionResponse;

    fn test_manifest(name: &str) -> AgentManifest {
        let mut m = AgentManifest::default();
        m.name = name.to_string();
        m
    }

    /// Create a mock driver with N identical responses.
    fn mock_driver(text: &str, count: usize) -> Arc<MockDriver> {
        let responses: Vec<_> = (0..count)
            .map(|_| CompletionResponse {
                text: text.to_string(),
                stop_reason: StopReason::EndTurn,
                tool_calls: vec![],
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
            })
            .collect();
        Arc::new(MockDriver::new(responses))
    }

    #[tokio::test]
    async fn test_pool_spawn_single() {
        let driver = mock_driver("agent-1 done", 1);
        let mut pool = AgentPool::new(driver, 4);

        let id = pool
            .spawn(SpawnConfig {
                manifest: test_manifest("agent-1"),
                query: "hello".into(),
            })
            .expect("spawn failed");

        assert_eq!(id, 1);
        assert_eq!(pool.active_count(), 1);
    }

    #[tokio::test]
    async fn test_pool_join_all() {
        let driver = mock_driver("done", 4);
        let mut pool = AgentPool::new(driver, 4);

        pool.spawn(SpawnConfig {
            manifest: test_manifest("a1"),
            query: "q1".into(),
        })
        .expect("spawn a1");

        pool.spawn(SpawnConfig {
            manifest: test_manifest("a2"),
            query: "q2".into(),
        })
        .expect("spawn a2");

        let results = pool.join_all().await;
        assert_eq!(results.len(), 2);

        for (_, result) in &results {
            let r = result.as_ref().expect("agent should succeed");
            assert_eq!(r.text, "done");
        }
    }

    #[tokio::test]
    async fn test_pool_capacity_limit() {
        let driver = mock_driver("done", 4);
        let mut pool = AgentPool::new(driver, 1);

        pool.spawn(SpawnConfig {
            manifest: test_manifest("a1"),
            query: "q1".into(),
        })
        .expect("spawn a1");

        // Second spawn should fail — pool at capacity
        let err = pool
            .spawn(SpawnConfig {
                manifest: test_manifest("a2"),
                query: "q2".into(),
            })
            .unwrap_err();

        assert!(
            matches!(err, AgentError::CircuitBreak(_)),
            "expected CircuitBreak, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_pool_fan_out_fan_in() {
        let driver = mock_driver("result", 3);
        let mut pool = AgentPool::new(driver, 4);

        let configs = vec![
            SpawnConfig {
                manifest: test_manifest("w1"),
                query: "task1".into(),
            },
            SpawnConfig {
                manifest: test_manifest("w2"),
                query: "task2".into(),
            },
            SpawnConfig {
                manifest: test_manifest("w3"),
                query: "task3".into(),
            },
        ];

        let ids = pool.fan_out(configs).expect("fan_out");
        assert_eq!(ids.len(), 3);

        let results = pool.join_all().await;
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_pool_join_next() {
        let driver = mock_driver("one", 1);
        let mut pool = AgentPool::new(driver, 4);

        pool.spawn(SpawnConfig {
            manifest: test_manifest("single"),
            query: "q".into(),
        })
        .expect("spawn");

        let (id, result) =
            pool.join_next().await.expect("should have result");
        assert_eq!(id, 1);
        assert_eq!(
            result.expect("agent ok").text,
            "one"
        );

        // No more agents
        assert!(pool.join_next().await.is_none());
    }

    #[tokio::test]
    async fn test_pool_abort_all() {
        let driver = mock_driver("done", 4);
        let mut pool = AgentPool::new(driver, 4);

        pool.spawn(SpawnConfig {
            manifest: test_manifest("abort-me"),
            query: "q".into(),
        })
        .expect("spawn");

        pool.abort_all();
        // After abort, join_all returns whatever completed
        let results = pool.join_all().await;
        // Aborted tasks may or may not have completed
        assert!(results.len() <= 1);
    }

    #[tokio::test]
    async fn test_pool_with_shared_memory() {
        let driver = mock_driver("memorized", 1);
        let memory = Arc::new(InMemorySubstrate::new());
        let mut pool =
            AgentPool::new(driver, 4).with_memory(memory);

        pool.spawn(SpawnConfig {
            manifest: test_manifest("mem-agent"),
            query: "remember this".into(),
        })
        .expect("spawn");

        let results = pool.join_all().await;
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_pool_defaults() {
        let driver = mock_driver("x", 2);
        let pool = AgentPool::new(driver, 8);
        assert_eq!(pool.max_concurrent(), 8);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn test_agent_message_fields() {
        let msg = AgentMessage {
            from: 0,
            to: 1,
            content: "hello sub-agent".into(),
        };
        assert_eq!(msg.from, 0);
        assert_eq!(msg.to, 1);
        assert_eq!(msg.content, "hello sub-agent");
    }

    #[tokio::test]
    async fn test_pool_increments_ids() {
        let driver = mock_driver("x", 2);
        let mut pool = AgentPool::new(driver, 4);

        let id1 = pool
            .spawn(SpawnConfig {
                manifest: test_manifest("a"),
                query: "q".into(),
            })
            .expect("spawn");

        // Drain first to free capacity tracking
        let _ = pool.join_all().await;

        let id2 = pool
            .spawn(SpawnConfig {
                manifest: test_manifest("b"),
                query: "q".into(),
            })
            .expect("spawn");

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }
}
