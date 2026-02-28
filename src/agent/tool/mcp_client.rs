//! MCP Client Tool — wraps external MCP server tools.
//!
//! Each `McpClientTool` represents a single tool discovered from
//! an external MCP server. The tool proxies execute calls through
//! an `McpTransport` trait, which abstracts over stdio/SSE/HTTP.
//!
//! # Privacy Enforcement (Poka-Yoke)
//!
//! MCP servers are subject to `PrivacyTier` rules:
//! - **Sovereign**: Only `stdio` transport allowed (local process)
//! - **Private/Standard**: All transports allowed
//!
//! # References
//!
//! - arXiv:2505.02279 — MCP interoperability survey
//! - arXiv:2503.23278 — MCP security analysis

use std::time::Duration;

use async_trait::async_trait;

use super::{ToolResult, Tool};
use crate::agent::capability::Capability;
use crate::agent::driver::ToolDefinition;

/// Transport abstraction for MCP server communication.
///
/// Separates the tool from the transport layer so that:
/// - Tests use `MockMcpTransport`
/// - Production uses `StdioMcpTransport` (Phase 2: `pmcp::Client`)
/// - Future: SSE/WebSocket transports
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Call a tool on the MCP server.
    async fn call_tool(
        &self,
        tool_name: &str,
        input: serde_json::Value,
    ) -> Result<String, String>;

    /// Server name for capability matching.
    fn server_name(&self) -> &str;
}

/// MCP client tool that proxies calls to an external MCP server.
pub struct McpClientTool {
    /// MCP server name (for capability matching).
    server_name: String,
    /// Tool name on the MCP server.
    tool_name: String,
    /// Tool description.
    description: String,
    /// JSON Schema for tool input.
    input_schema: serde_json::Value,
    /// Transport for calling the MCP server.
    transport: Box<dyn McpTransport>,
    /// Execution timeout.
    timeout: Duration,
}

impl McpClientTool {
    /// Create a new MCP client tool.
    pub fn new(
        server_name: impl Into<String>,
        tool_name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
        transport: Box<dyn McpTransport>,
    ) -> Self {
        Self {
            server_name: server_name.into(),
            tool_name: tool_name.into(),
            description: description.into(),
            input_schema,
            transport,
            timeout: Duration::from_secs(60),
        }
    }

    /// Set the execution timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The prefixed tool name: `mcp_{server}_{tool}`.
    fn prefixed_name(&self) -> String {
        format!("mcp_{}_{}", self.server_name, self.tool_name)
    }
}

#[async_trait]
impl Tool for McpClientTool {
    fn name(&self) -> &'static str {
        // Leak the name to get 'static lifetime.
        // This is safe because tool names live for the process.
        Box::leak(self.prefixed_name().into_boxed_str())
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.prefixed_name(),
            description: format!(
                "[MCP:{}] {}",
                self.server_name, self.description
            ),
            input_schema: self.input_schema.clone(),
        }
    }

    async fn execute(
        &self,
        input: serde_json::Value,
    ) -> ToolResult {
        match self
            .transport
            .call_tool(&self.tool_name, input)
            .await
        {
            Ok(content) => ToolResult::success(content),
            Err(e) => ToolResult::error(format!(
                "MCP call to {}:{} failed: {}",
                self.server_name, self.tool_name, e
            )),
        }
    }

    fn required_capability(&self) -> Capability {
        Capability::Mcp {
            server: self.server_name.clone(),
            tool: self.tool_name.clone(),
        }
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Stdio MCP transport — launches a subprocess and communicates via stdin/stdout.
///
/// The subprocess is expected to speak JSON-RPC 2.0 with MCP tools/call messages.
/// Each `call_tool` sends a request line and reads a response line.
///
/// # Privacy
///
/// This transport is allowed in Sovereign tier because the subprocess
/// runs locally (no network egress).
pub struct StdioMcpTransport {
    server: String,
    command: Vec<String>,
}

impl StdioMcpTransport {
    /// Create a stdio transport for the given server.
    ///
    /// `command` is the full command line (e.g., `["node", "server.js"]`).
    pub fn new(
        server: impl Into<String>,
        command: Vec<String>,
    ) -> Self {
        Self {
            server: server.into(),
            command,
        }
    }
}

#[async_trait]
impl McpTransport for StdioMcpTransport {
    async fn call_tool(
        &self,
        tool_name: &str,
        input: serde_json::Value,
    ) -> Result<String, String> {
        if self.command.is_empty() {
            return Err("stdio transport: empty command".into());
        }

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": input,
            }
        });

        let request_str = serde_json::to_string(&request)
            .map_err(|e| format!("serialize request: {e}"))?;

        let output = tokio::process::Command::new(&self.command[0])
            .args(&self.command[1..])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", self.command[0]))?;

        // Write request to stdin
        let mut child = output;
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin
                .write_all(request_str.as_bytes())
                .await
                .map_err(|e| format!("write stdin: {e}"))?;
            stdin
                .write_all(b"\n")
                .await
                .map_err(|e| format!("write newline: {e}"))?;
            drop(stdin); // Close stdin to signal EOF
        }

        let result = child
            .wait_with_output()
            .await
            .map_err(|e| format!("wait: {e}"))?;

        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(format!(
                "process exited {}: {}",
                result.status,
                stderr.trim()
            ));
        }

        let stdout = String::from_utf8_lossy(&result.stdout);
        // Parse JSON-RPC response
        let response: serde_json::Value =
            serde_json::from_str(stdout.trim())
                .map_err(|e| format!("parse response: {e}"))?;

        // Extract result content
        if let Some(error) = response.get("error") {
            let msg = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            return Err(msg.to_string());
        }

        if let Some(result) = response.get("result") {
            // MCP tools/call returns { content: [{ text: "..." }] }
            if let Some(content) = result.get("content") {
                if let Some(arr) = content.as_array() {
                    let texts: Vec<&str> = arr
                        .iter()
                        .filter_map(|c| c.get("text").and_then(|t| t.as_str()))
                        .collect();
                    if !texts.is_empty() {
                        return Ok(texts.join("\n"));
                    }
                }
            }
            // Fallback: stringify the result
            Ok(serde_json::to_string(result)
                .unwrap_or_else(|_| "{}".to_string()))
        } else {
            Err("no result in response".into())
        }
    }

    fn server_name(&self) -> &str {
        &self.server
    }
}

/// Mock MCP transport for testing.
pub struct MockMcpTransport {
    server: String,
    responses: std::sync::Mutex<Vec<Result<String, String>>>,
}

impl MockMcpTransport {
    /// Create a mock transport with pre-configured responses.
    pub fn new(
        server: impl Into<String>,
        responses: Vec<Result<String, String>>,
    ) -> Self {
        Self {
            server: server.into(),
            responses: std::sync::Mutex::new(responses),
        }
    }
}

#[async_trait]
impl McpTransport for MockMcpTransport {
    async fn call_tool(
        &self,
        _tool_name: &str,
        _input: serde_json::Value,
    ) -> Result<String, String> {
        let mut responses = self.responses.lock().expect(
            "mock transport lock",
        );
        if responses.is_empty() {
            Err("mock transport exhausted".into())
        } else {
            responses.remove(0)
        }
    }

    fn server_name(&self) -> &str {
        &self.server
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_tool(
        responses: Vec<Result<String, String>>,
    ) -> McpClientTool {
        McpClientTool::new(
            "test-server",
            "search",
            "Search documents",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }),
            Box::new(MockMcpTransport::new(
                "test-server",
                responses,
            )),
        )
    }

    #[test]
    fn test_prefixed_name() {
        let tool = mock_tool(vec![]);
        assert_eq!(tool.prefixed_name(), "mcp_test-server_search");
    }

    #[test]
    fn test_definition() {
        let tool = mock_tool(vec![]);
        let def = tool.definition();
        assert_eq!(def.name, "mcp_test-server_search");
        assert!(def.description.contains("[MCP:test-server]"));
        assert!(def.description.contains("Search documents"));
    }

    #[test]
    fn test_required_capability() {
        let tool = mock_tool(vec![]);
        let cap = tool.required_capability();
        assert!(matches!(
            cap,
            Capability::Mcp { server, tool }
            if server == "test-server" && tool == "search"
        ));
    }

    #[tokio::test]
    async fn test_execute_success() {
        let tool = mock_tool(vec![Ok("found 3 results".into())]);
        let result = tool
            .execute(serde_json::json!({"query": "rust"}))
            .await;
        assert!(!result.is_error);
        assert_eq!(result.content, "found 3 results");
    }

    #[tokio::test]
    async fn test_execute_error() {
        let tool =
            mock_tool(vec![Err("connection refused".into())]);
        let result = tool
            .execute(serde_json::json!({"query": "test"}))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("MCP call"));
        assert!(result.content.contains("connection refused"));
    }

    #[tokio::test]
    async fn test_execute_transport_exhausted() {
        let tool = mock_tool(vec![]);
        let result = tool
            .execute(serde_json::json!({}))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("exhausted"));
    }

    #[test]
    fn test_timeout_default() {
        let tool = mock_tool(vec![]);
        assert_eq!(tool.timeout(), Duration::from_secs(60));
    }

    #[test]
    fn test_timeout_custom() {
        let tool =
            mock_tool(vec![]).with_timeout(Duration::from_secs(10));
        assert_eq!(tool.timeout(), Duration::from_secs(10));
    }

    #[test]
    fn test_capability_matches_with_registry() {
        use crate::agent::capability::capability_matches;

        let tool = mock_tool(vec![]);
        let cap = tool.required_capability();

        // Exact match
        let granted = vec![Capability::Mcp {
            server: "test-server".into(),
            tool: "search".into(),
        }];
        assert!(capability_matches(&granted, &cap));

        // Wildcard tool match
        let wildcard = vec![Capability::Mcp {
            server: "test-server".into(),
            tool: "*".into(),
        }];
        assert!(capability_matches(&wildcard, &cap));

        // Wrong server — denied
        let wrong = vec![Capability::Mcp {
            server: "other-server".into(),
            tool: "search".into(),
        }];
        assert!(!capability_matches(&wrong, &cap));
    }

    #[tokio::test]
    async fn test_stdio_transport_empty_command() {
        let transport = StdioMcpTransport::new("test", vec![]);
        let result = transport
            .call_tool("search", serde_json::json!({}))
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty command"));
    }

    #[tokio::test]
    async fn test_stdio_transport_nonexistent_command() {
        let transport = StdioMcpTransport::new(
            "test",
            vec!["__nonexistent_binary_42__".into()],
        );
        let result = transport
            .call_tool("search", serde_json::json!({}))
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("spawn"));
    }

    #[tokio::test]
    async fn test_stdio_transport_echo_jsonrpc() {
        // Use a shell command that echoes a valid JSON-RPC response
        let response = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": "hello from mcp"}]
            }
        });
        let transport = StdioMcpTransport::new(
            "echo-server",
            vec![
                "bash".into(),
                "-c".into(),
                format!("echo '{}'", response),
            ],
        );
        let result = transport
            .call_tool("greet", serde_json::json!({"name": "test"}))
            .await;
        assert!(result.is_ok(), "expected ok, got: {:?}", result);
        assert_eq!(result.unwrap(), "hello from mcp");
    }

    #[tokio::test]
    async fn test_stdio_transport_error_response() {
        let response = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "method not found"}
        });
        let transport = StdioMcpTransport::new(
            "err-server",
            vec![
                "bash".into(),
                "-c".into(),
                format!("echo '{}'", response),
            ],
        );
        let result = transport
            .call_tool("missing", serde_json::json!({}))
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("method not found"));
    }

    #[tokio::test]
    async fn test_stdio_transport_server_name() {
        let transport = StdioMcpTransport::new(
            "my-server",
            vec!["echo".into()],
        );
        assert_eq!(transport.server_name(), "my-server");
    }

    #[tokio::test]
    async fn test_multiple_calls() {
        let tool = mock_tool(vec![
            Ok("first".into()),
            Ok("second".into()),
        ]);

        let r1 = tool.execute(serde_json::json!({})).await;
        assert_eq!(r1.content, "first");

        let r2 = tool.execute(serde_json::json!({})).await;
        assert_eq!(r2.content, "second");
    }
}
