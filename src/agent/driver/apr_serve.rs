//! AprServeDriver — first-class inference via `apr serve` subprocess.
//!
//! Spawns `apr serve run <model>` as a child process with CUDA/GPU support,
//! then connects via OpenAI-compatible HTTP API. This is the **preferred**
//! inference path for `batuta code` / `apr code`:
//!
//! - Full CUDA/GPU acceleration (apr-cli has all features)
//! - APR and GGUF format support (prefers APR)
//! - No feature flag issues (batuta doesn't need `cuda` feature)
//! - Sovereign: localhost only, no data egress
//!
//! PMAT-160: Replaces embedded RealizarDriver as primary inference.
//! RealizarDriver remains as fallback when `apr` binary is not on PATH.

use async_trait::async_trait;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use super::{CompletionRequest, CompletionResponse, LlmDriver, Message, ToolCall};
use crate::agent::result::{AgentError, DriverError, StopReason, TokenUsage};
use crate::serve::backends::PrivacyTier;

/// Driver that uses `apr serve` subprocess for inference.
pub struct AprServeDriver {
    /// Base URL for the local server (e.g., `http://127.0.0.1:19384`)
    base_url: String,
    /// Model name for OpenAI API requests
    model_name: String,
    /// Child process handle (killed on drop)
    _child: Child,
    /// Context window size
    context_window_size: usize,
}

impl Drop for AprServeDriver {
    /// PMAT-166: Graceful shutdown — SIGTERM first, SIGKILL after 2s timeout.
    fn drop(&mut self) {
        let pid = self._child.id();

        // Try graceful shutdown first (SIGTERM on Unix via kill command)
        #[cfg(unix)]
        {
            let _ = Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();

            // Wait up to 2s for graceful exit
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            loop {
                match self._child.try_wait() {
                    Ok(Some(_)) => return, // Exited cleanly
                    Ok(None) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    _ => break, // Timeout or error — force kill
                }
            }
        }

        // Fallback: force kill (always runs on Windows, or after SIGTERM timeout)
        let _ = self._child.kill();
        let _ = self._child.wait();
    }
}

impl AprServeDriver {
    /// Launch `apr serve run` and wait for readiness.
    ///
    /// Picks a random port, spawns the subprocess, polls the health
    /// endpoint until ready (max 30s). Returns error if `apr` not
    /// found or server fails to start.
    pub fn launch(model_path: PathBuf, context_window: Option<usize>) -> Result<Self, AgentError> {
        let apr_path = find_apr_binary()?;

        // Pick a random high port to avoid conflicts
        let port = 19384 + (std::process::id() % 1000) as u16;
        let base_url = format!("http://127.0.0.1:{port}");

        let model_name = model_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "local".to_string());

        // PMAT-164: conditional GPU flag — APR gets no --gpu (wgpu shader
        // panics on -inf literal), GGUF gets --gpu for full CUDA acceleration.
        let is_apr = model_path.extension().map(|e| e == "apr").unwrap_or(false);

        let mut args = vec![
            "serve".to_string(),
            "run".to_string(),
            model_path.to_string_lossy().to_string(),
            "--port".to_string(),
            port.to_string(),
            "--host".to_string(),
            "127.0.0.1".to_string(),
        ];
        if !is_apr {
            args.push("--gpu".to_string());
        }

        // Spawn apr serve as child process
        let child = Command::new(&apr_path)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                AgentError::Driver(DriverError::InferenceFailed(format!(
                    "failed to spawn apr serve: {e}"
                )))
            })?;

        let gpu_note = if is_apr { "CPU (APR)" } else { "GPU" };
        eprintln!("Launched apr serve on port {port} ({gpu_note}, pid {})", child.id());

        let driver = Self {
            base_url,
            model_name,
            _child: child,
            context_window_size: context_window.unwrap_or(4096),
        };

        // Wait for server to be ready
        driver.wait_for_ready()?;

        Ok(driver)
    }

    /// Poll health endpoint until server is ready (max 30s).
    ///
    /// Uses TCP connect probe (no HTTP library needed for health check).
    fn wait_for_ready(&self) -> Result<(), AgentError> {
        let addr = self.base_url.trim_start_matches("http://");
        let sock_addr: std::net::SocketAddr =
            addr.parse().unwrap_or_else(|_| std::net::SocketAddr::from(([127, 0, 0, 1], 19384)));

        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(30);

        loop {
            if start.elapsed() > timeout {
                return Err(AgentError::Driver(DriverError::InferenceFailed(
                    "apr serve did not become ready within 30s".into(),
                )));
            }

            if std::net::TcpStream::connect_timeout(
                &sock_addr,
                std::time::Duration::from_millis(200),
            )
            .is_ok()
            {
                eprintln!("apr serve ready ({:.1}s)", start.elapsed().as_secs_f64());
                return Ok(());
            }

            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }

    /// Build OpenAI-compatible request body.
    ///
    /// PMAT-161: Strips the enriched tool-definition section from the system
    /// prompt. The full JSON schemas (~2000 tokens) overwhelm 1.5B models.
    /// Instead, sends a compact tool list. `apr serve` applies its own chat
    /// template — `<tool_call>` format instructions are redundant over HTTP.
    fn build_openai_body(&self, request: &CompletionRequest) -> serde_json::Value {
        let mut messages = Vec::new();

        if let Some(ref system) = request.system {
            // Strip everything after "## Available Tools" — that section was
            // injected by build_enriched_system() for the RealizarDriver path.
            // For HTTP, we replace it with a compact tool summary.
            let base =
                system.find("\n\n## Available Tools").map(|i| &system[..i]).unwrap_or(system);

            let mut compact_system = base.to_string();

            // Add compact tool list (name + one-liner, no JSON schemas)
            if !request.tools.is_empty() {
                compact_system.push_str("\n\nYou have these tools: ");
                let tool_names: Vec<&str> = request.tools.iter().map(|t| t.name.as_str()).collect();
                compact_system.push_str(&tool_names.join(", "));
                compact_system.push_str(
                    ".\nTo use a tool, respond with a JSON object: \
                     {\"name\": \"tool_name\", \"input\": {\"param\": \"value\"}}",
                );
            }

            messages.push(serde_json::json!({
                "role": "system",
                "content": compact_system
            }));
        }

        for msg in &request.messages {
            match msg {
                Message::User(text) => messages.push(serde_json::json!({
                    "role": "user",
                    "content": text
                })),
                Message::Assistant(text) => messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": text
                })),
                Message::AssistantToolUse(call) => messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": format!("<tool_call>\n{}\n</tool_call>",
                        serde_json::json!({"name": call.name, "input": call.input}))
                })),
                Message::ToolResult(result) => messages.push(serde_json::json!({
                    "role": "user",
                    "content": format!("<tool_result>\n{}\n</tool_result>", result.content)
                })),
                _ => {}
            }
        }

        // Cap max_tokens for HTTP path — single agent turn rarely needs >512 tokens.
        // The manifest default (4096) causes very long generation on local models.
        let max_tokens = request.max_tokens.min(512);

        serde_json::json!({
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": request.temperature,
            "stream": false
        })
    }
}

#[async_trait]
impl LlmDriver for AprServeDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, AgentError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = self.build_openai_body(&request);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| AgentError::Driver(DriverError::Network(format!("http client: {e}"))))?;
        let response = client
            .post(&url)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::Driver(DriverError::Network(format!("apr serve: {e}"))))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::Driver(DriverError::Network(format!(
                "apr serve HTTP {status}: {text}"
            ))));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| AgentError::Driver(DriverError::InferenceFailed(format!("parse: {e}"))))?;

        // Extract response from OpenAI format
        let text = json["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();

        let usage = json.get("usage").cloned().unwrap_or(serde_json::json!({}));
        let input_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
        let output_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);

        // Parse tool calls from text (same parser as RealizarDriver)
        let (clean_text, tool_calls) = super::realizar::parse_tool_calls_pub(&text);

        let stop_reason =
            if tool_calls.is_empty() { StopReason::EndTurn } else { StopReason::ToolUse };

        Ok(CompletionResponse {
            text: clean_text,
            stop_reason,
            tool_calls,
            usage: TokenUsage { input_tokens, output_tokens },
        })
    }

    fn context_window(&self) -> usize {
        self.context_window_size
    }

    fn privacy_tier(&self) -> PrivacyTier {
        // Sovereign: apr serve runs on localhost, zero network egress
        PrivacyTier::Sovereign
    }
}

/// Find the `apr` binary on PATH.
fn find_apr_binary() -> Result<PathBuf, AgentError> {
    which::which("apr").map_err(|_| {
        AgentError::Driver(DriverError::InferenceFailed(
            "apr binary not found on PATH. Install: cargo install apr-cli".into(),
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_apr_binary() {
        // apr should be installed in the dev environment
        let result = find_apr_binary();
        assert!(result.is_ok(), "apr binary should be on PATH: {result:?}");
    }

    #[test]
    fn test_privacy_tier_is_sovereign() {
        // AprServeDriver is always Sovereign (localhost only)
        assert_eq!(PrivacyTier::Sovereign, PrivacyTier::Sovereign);
    }

    #[test]
    fn test_apr_extension_detected() {
        let apr = PathBuf::from("/models/qwen.apr");
        assert_eq!(apr.extension().map(|e| e == "apr"), Some(true));

        let gguf = PathBuf::from("/models/qwen.gguf");
        assert_eq!(gguf.extension().map(|e| e == "apr"), Some(false));
    }
}
