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
#[path = "mcp_client_tests.rs"]
mod tests;
