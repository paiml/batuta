//! MCP Server — expose agent tools to external MCP clients.
//!
//! Implements handler dispatch for agent tools (memory, rag, compute)
//! so external LLM clients (Claude Code, other agents) can call
//! the agent's tools over MCP protocol.
//!
//! # Architecture
//!
//! ```text
//! External MCP Client
//!   → JSON-RPC 2.0 (stdio/SSE)
//!     → HandlerRegistry.dispatch(method, params)
//!       → MemoryHandler / RagHandler / ComputeHandler
//!         → ToolResult
//! ```
//!
//! # Phase 3 Implementation
//!
//! Uses a trait-based handler abstraction compatible with pforge.
//! When pforge is added as a dependency, handlers implement
//! `pforge_runtime::Handler` directly.
//!
//! # References
//!
//! - arXiv:2505.02279 — MCP interoperability survey
//! - arXiv:2503.23278 — MCP security analysis

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use super::ToolResult;
use crate::agent::memory::MemorySubstrate;

/// Handler for a single MCP tool endpoint.
///
/// Mirrors pforge `Handler` trait pattern for forward compatibility.
#[async_trait]
pub trait McpHandler: Send + Sync {
    /// Tool name as exposed via MCP (e.g., "memory_store").
    fn name(&self) -> &str;

    /// Human-readable description for tool discovery.
    fn description(&self) -> &str;

    /// JSON Schema for the tool's input parameters.
    fn input_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given parameters.
    async fn handle(
        &self,
        params: serde_json::Value,
    ) -> ToolResult;
}

/// Registry of MCP handlers for dispatch.
pub struct HandlerRegistry {
    handlers: HashMap<String, Box<dyn McpHandler>>,
}

impl HandlerRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register a handler.
    pub fn register(&mut self, handler: Box<dyn McpHandler>) {
        let name = handler.name().to_string();
        self.handlers.insert(name, handler);
    }

    /// Dispatch a tool call to the appropriate handler.
    pub async fn dispatch(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> ToolResult {
        match self.handlers.get(method) {
            Some(handler) => handler.handle(params).await,
            None => ToolResult::error(format!(
                "unknown method: {method}"
            )),
        }
    }

    /// List available tools for MCP discovery.
    pub fn list_tools(&self) -> Vec<McpToolInfo> {
        self.handlers
            .values()
            .map(|h| McpToolInfo {
                name: h.name().to_string(),
                description: h.description().to_string(),
                input_schema: h.input_schema(),
            })
            .collect()
    }

    /// Number of registered handlers.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }
}

impl Default for HandlerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Tool info returned by MCP tools/list.
#[derive(Debug, Clone, serde::Serialize)]
pub struct McpToolInfo {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON Schema for input.
    pub input_schema: serde_json::Value,
}

/// Memory handler — exposes agent memory via MCP.
///
/// Supports `store` (remember) and `recall` (search) actions.
pub struct MemoryHandler {
    memory: Arc<dyn MemorySubstrate>,
    agent_id: String,
}

impl MemoryHandler {
    /// Create a new memory handler.
    pub fn new(
        memory: Arc<dyn MemorySubstrate>,
        agent_id: impl Into<String>,
    ) -> Self {
        Self {
            memory,
            agent_id: agent_id.into(),
        }
    }
}

#[async_trait]
impl McpHandler for MemoryHandler {
    fn name(&self) -> &str {
        "memory"
    }

    fn description(&self) -> &str {
        "Store and recall agent memory fragments"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "recall"]
                },
                "content": { "type": "string" },
                "query": { "type": "string" },
                "limit": { "type": "integer" }
            },
            "required": ["action"]
        })
    }

    async fn handle(
        &self,
        params: serde_json::Value,
    ) -> ToolResult {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        match action {
            "store" => {
                let content = params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if content.is_empty() {
                    return ToolResult::error(
                        "content is required for store",
                    );
                }
                match self
                    .memory
                    .remember(
                        &self.agent_id,
                        content,
                        crate::agent::memory::MemorySource::User,
                        None,
                    )
                    .await
                {
                    Ok(id) => ToolResult::success(format!(
                        "Stored with id: {id}"
                    )),
                    Err(e) => ToolResult::error(format!(
                        "store failed: {e}"
                    )),
                }
            }
            "recall" => {
                let query = params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let limit = params
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(5) as usize;
                match self
                    .memory
                    .recall(query, limit, None, None)
                    .await
                {
                    Ok(fragments) => {
                        if fragments.is_empty() {
                            return ToolResult::success(
                                "No matching memories found.",
                            );
                        }
                        let mut out = String::new();
                        for f in &fragments {
                            out.push_str(&format!(
                                "- {} (score: {:.2})\n",
                                f.content, f.relevance_score,
                            ));
                        }
                        ToolResult::success(out)
                    }
                    Err(e) => ToolResult::error(format!(
                        "recall failed: {e}"
                    )),
                }
            }
            _ => ToolResult::error(format!(
                "unknown action: {action} (expected: store, recall)"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::in_memory::InMemorySubstrate;

    fn test_registry() -> HandlerRegistry {
        let memory =
            Arc::new(InMemorySubstrate::new());
        let mut registry = HandlerRegistry::new();
        registry.register(Box::new(MemoryHandler::new(
            memory, "test-agent",
        )));
        registry
    }

    #[test]
    fn test_registry_creation() {
        let registry = HandlerRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_register() {
        let registry = test_registry();
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_list_tools() {
        let registry = test_registry();
        let tools = registry.list_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "memory");
    }

    #[tokio::test]
    async fn test_dispatch_unknown_method() {
        let registry = test_registry();
        let result = registry
            .dispatch("nonexistent", serde_json::json!({}))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("unknown method"));
    }

    #[tokio::test]
    async fn test_memory_store() {
        let registry = test_registry();
        let result = registry
            .dispatch(
                "memory",
                serde_json::json!({
                    "action": "store",
                    "content": "test memory content"
                }),
            )
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("Stored with id"));
    }

    #[tokio::test]
    async fn test_memory_store_empty_content() {
        let registry = test_registry();
        let result = registry
            .dispatch(
                "memory",
                serde_json::json!({
                    "action": "store",
                    "content": ""
                }),
            )
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("required"));
    }

    #[tokio::test]
    async fn test_memory_recall_empty() {
        let registry = test_registry();
        let result = registry
            .dispatch(
                "memory",
                serde_json::json!({
                    "action": "recall",
                    "query": "nothing"
                }),
            )
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("No matching"));
    }

    #[tokio::test]
    async fn test_memory_store_then_recall() {
        let memory: Arc<dyn MemorySubstrate> =
            Arc::new(InMemorySubstrate::new());
        let mut registry = HandlerRegistry::new();
        registry.register(Box::new(MemoryHandler::new(
            Arc::clone(&memory),
            "test",
        )));

        // Store
        let store_result = registry
            .dispatch(
                "memory",
                serde_json::json!({
                    "action": "store",
                    "content": "Rust is a systems language"
                }),
            )
            .await;
        assert!(!store_result.is_error);

        // Recall
        let recall_result = registry
            .dispatch(
                "memory",
                serde_json::json!({
                    "action": "recall",
                    "query": "Rust",
                    "limit": 3
                }),
            )
            .await;
        assert!(!recall_result.is_error);
        assert!(recall_result.content.contains("systems language"));
    }

    #[tokio::test]
    async fn test_memory_unknown_action() {
        let registry = test_registry();
        let result = registry
            .dispatch(
                "memory",
                serde_json::json!({
                    "action": "delete"
                }),
            )
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("unknown action"));
    }

    #[test]
    fn test_memory_handler_schema() {
        let memory =
            Arc::new(InMemorySubstrate::new());
        let handler = MemoryHandler::new(memory, "test");
        let schema = handler.input_schema();
        assert!(schema.get("properties").is_some());
        assert_eq!(handler.name(), "memory");
        assert!(!handler.description().is_empty());
    }

    #[test]
    fn test_default_registry() {
        let registry = HandlerRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_mcp_tool_info_serialization() {
        let info = McpToolInfo {
            name: "test".into(),
            description: "Test tool".into(),
            input_schema: serde_json::json!({}),
        };
        let json = serde_json::to_string(&info)
            .expect("serialize");
        assert!(json.contains("\"name\":\"test\""));
    }
}
