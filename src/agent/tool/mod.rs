//! Tool system for agent actions.
//!
//! Tools are the agent's interface to the outside world. Each tool
//! declares a required capability; the agent manifest must grant
//! that capability for the tool to be available (Poka-Yoke).

pub mod memory;
#[cfg(feature = "rag")]
pub mod rag;

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Duration;

use super::capability::Capability;
use super::driver::ToolDefinition;

/// Result of a tool execution.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Result content as text.
    pub content: String,
    /// Whether the tool call errored.
    pub is_error: bool,
}

impl ToolResult {
    /// Create a successful result.
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    /// Create an error result.
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// Executable tool with capability enforcement.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (must match ToolDefinition name).
    fn name(&self) -> &str;

    /// JSON Schema definition for the LLM.
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with JSON input.
    async fn execute(
        &self,
        input: serde_json::Value,
    ) -> ToolResult;

    /// Required capability to invoke this tool (Poka-Yoke).
    fn required_capability(&self) -> Capability;

    /// Execution timeout (Jidoka: stop on timeout).
    fn timeout(&self) -> Duration {
        Duration::from_secs(120)
    }
}

/// Registry of available tools.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Get tool definitions filtered by granted capabilities.
    pub fn definitions_for(
        &self,
        capabilities: &[Capability],
    ) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .filter(|t| {
                super::capability::capability_matches(
                    capabilities,
                    &t.required_capability(),
                )
            })
            .map(|t| t.definition())
            .collect()
    }

    /// List all registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyTool;

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &str {
            "dummy"
        }

        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "dummy".into(),
                description: "A dummy tool".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            }
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
        ) -> ToolResult {
            ToolResult::success("dummy result")
        }

        fn required_capability(&self) -> Capability {
            Capability::Memory
        }
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(DummyTool));

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert!(registry.get("dummy").is_some());
        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn test_registry_definitions_filtered() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(DummyTool));

        // DummyTool requires Memory capability
        let with_memory =
            registry.definitions_for(&[Capability::Memory]);
        assert_eq!(with_memory.len(), 1);

        let without_memory = registry.definitions_for(&[Capability::Rag]);
        assert_eq!(without_memory.len(), 0);
    }

    #[test]
    fn test_registry_tool_names() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(DummyTool));
        assert!(registry.tool_names().contains(&"dummy"));
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("ok");
        assert_eq!(result.content, "ok");
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("failed");
        assert_eq!(result.content, "failed");
        assert!(result.is_error);
    }

    #[test]
    fn test_registry_default() {
        let registry = ToolRegistry::default();
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_dummy_tool_execute() {
        let tool = DummyTool;
        let result = tool.execute(serde_json::json!({})).await;
        assert_eq!(result.content, "dummy result");
        assert!(!result.is_error);
    }

    #[test]
    fn test_dummy_tool_timeout() {
        let tool = DummyTool;
        assert_eq!(tool.timeout(), Duration::from_secs(120));
    }
}
