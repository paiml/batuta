use std::time::Duration;

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
    let result = tool.execute(serde_json::json!({})).await;
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
