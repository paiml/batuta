//! Agent manifest configuration.
//!
//! Defines the TOML-based configuration for agent instances.
//! Includes model path, resource quotas (Muda elimination),
//! granted capabilities (Poka-Yoke), and privacy tier.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::capability::Capability;
use crate::serve::backends::PrivacyTier;

/// Agent configuration loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentManifest {
    /// Human-readable agent name.
    pub name: String,
    /// Semantic version.
    pub version: String,
    /// Description of what this agent does.
    pub description: String,
    /// LLM model configuration.
    pub model: ModelConfig,
    /// Resource quotas (Muda elimination).
    pub resources: ResourceQuota,
    /// Granted capabilities (Poka-Yoke).
    pub capabilities: Vec<Capability>,
    /// Privacy tier. Default: Sovereign (local-only).
    pub privacy: PrivacyTier,
    /// External MCP servers to connect to (agents-mcp feature). [F-022]
    #[cfg(feature = "agents-mcp")]
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
}

impl Default for AgentManifest {
    fn default() -> Self {
        Self {
            name: "unnamed-agent".into(),
            version: "0.1.0".into(),
            description: String::new(),
            model: ModelConfig::default(),
            resources: ResourceQuota::default(),
            capabilities: vec![Capability::Rag, Capability::Memory],
            privacy: PrivacyTier::Sovereign,
            #[cfg(feature = "agents-mcp")]
            mcp_servers: Vec::new(),
        }
    }
}

/// LLM model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Path to local model file (GGUF/APR/SafeTensors).
    pub model_path: Option<PathBuf>,
    /// Remote model identifier (Phase 2, for spillover).
    pub remote_model: Option<String>,
    /// `HuggingFace` repo ID for auto-pull (Phase 2).
    /// When set and `model_path` is None, resolves via `apr pull`.
    pub model_repo: Option<String>,
    /// Quantization variant for auto-pull (e.g., `q4_k_m`).
    pub model_quantization: Option<String>,
    /// Maximum tokens per completion.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// System prompt injected at start of conversation.
    pub system_prompt: String,
    /// Context window size override (auto-detected if None).
    pub context_window: Option<usize>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            remote_model: None,
            model_repo: None,
            model_quantization: None,
            max_tokens: 4096,
            temperature: 0.3,
            system_prompt: "You are a helpful assistant.".into(),
            context_window: None,
        }
    }
}

impl ModelConfig {
    /// Resolve the effective model path.
    ///
    /// Resolution order:
    /// 1. Explicit `model_path` — return as-is
    /// 2. `model_repo` — resolve via pacha cache at
    ///    `~/.cache/pacha/models/{repo}/{quant}.gguf`
    /// 3. Neither — return None
    pub fn resolve_model_path(&self) -> Option<PathBuf> {
        if let Some(ref path) = self.model_path {
            return Some(path.clone());
        }
        if let Some(ref repo) = self.model_repo {
            let quant = self
                .model_quantization
                .as_deref()
                .unwrap_or("q4_k_m");
            let cache_dir = dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("pacha")
                .join("models");
            let filename = format!(
                "{}-{}.gguf",
                repo.replace('/', "--"),
                quant,
            );
            return Some(cache_dir.join(filename));
        }
        None
    }

    /// Check if model needs to be downloaded (auto-pull).
    ///
    /// Returns `Some(repo)` if `model_repo` is set but the
    /// resolved cache path does not exist on disk.
    pub fn needs_pull(&self) -> Option<&str> {
        if self.model_path.is_some() {
            return None;
        }
        if let Some(ref repo) = self.model_repo {
            if let Some(path) = self.resolve_model_path() {
                if !path.exists() {
                    return Some(repo.as_str());
                }
            }
        }
        None
    }

    /// Auto-pull model via `apr pull` subprocess.
    ///
    /// Invokes `apr pull <repo>` with a configurable timeout.
    /// The `apr` CLI handles caching internally at
    /// `~/.cache/pacha/models/`. Returns the resolved cache path
    /// on success.
    ///
    /// Jidoka: stops on subprocess failure rather than continuing
    /// with a missing model.
    pub fn auto_pull(&self, timeout_secs: u64) -> Result<PathBuf, AutoPullError> {
        let repo = self
            .model_repo
            .as_deref()
            .ok_or(AutoPullError::NoRepo)?;

        let target_path = self
            .resolve_model_path()
            .ok_or(AutoPullError::NoRepo)?;

        // Check if `apr` binary is available
        let apr_path = which_apr()?;

        // Build model reference: repo or repo:quant
        let model_ref = match self.model_quantization.as_deref() {
            Some(q) => format!("{repo}:{q}"),
            None => repo.to_string(),
        };

        let mut child = std::process::Command::new(&apr_path)
            .args(["pull", &model_ref])
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                AutoPullError::Subprocess(format!(
                    "cannot spawn apr pull: {e}"
                ))
            })?;

        let output = wait_with_timeout(
            &mut child, timeout_secs,
        )?;

        if !output.status.success() {
            let stderr =
                String::from_utf8_lossy(&output.stderr);
            return Err(AutoPullError::Subprocess(format!(
                "apr pull exited with {}: {}",
                output.status,
                stderr.trim(),
            )));
        }

        if !target_path.exists() {
            return Err(AutoPullError::Subprocess(
                "apr pull completed but model file not found at expected path".into(),
            ));
        }

        Ok(target_path)
    }
}

/// Errors from model auto-pull operations.
#[derive(Debug)]
pub enum AutoPullError {
    /// No `model_repo` configured.
    NoRepo,
    /// `apr` binary not found in PATH.
    NotInstalled,
    /// Subprocess execution failed.
    Subprocess(String),
    /// Filesystem I/O error.
    Io(String),
}

impl std::fmt::Display for AutoPullError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoRepo => write!(f, "no model_repo configured"),
            Self::NotInstalled => write!(
                f,
                "apr binary not found in PATH; install with: cargo install apr-cli"
            ),
            Self::Subprocess(msg) | Self::Io(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for AutoPullError {}

/// Locate the `apr` binary in PATH.
fn which_apr() -> Result<PathBuf, AutoPullError> {
    // Check common names: `apr`, `apr-cli`
    for name in &["apr", "apr-cli"] {
        if let Ok(path) = which::which(name) {
            return Ok(path);
        }
    }
    Err(AutoPullError::NotInstalled)
}

/// Wait for a child process with a polling timeout.
fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout_secs: u64,
) -> Result<std::process::Output, AutoPullError> {
    let deadline = std::time::Instant::now()
        + std::time::Duration::from_secs(timeout_secs);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stderr = child
                    .stderr
                    .take()
                    .map(|mut s| {
                        let mut buf = Vec::new();
                        std::io::Read::read_to_end(&mut s, &mut buf).ok();
                        buf
                    })
                    .unwrap_or_default();
                return Ok(std::process::Output {
                    status,
                    stdout: Vec::new(),
                    stderr,
                });
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    child.kill().ok();
                    return Err(AutoPullError::Subprocess(
                        format!(
                            "apr pull timed out after {timeout_secs}s"
                        ),
                    ));
                }
                std::thread::sleep(
                    std::time::Duration::from_millis(500),
                );
            }
            Err(e) => {
                return Err(AutoPullError::Subprocess(
                    format!("wait error: {e}"),
                ));
            }
        }
    }
}

/// Resource quotas (Muda elimination).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ResourceQuota {
    /// Maximum loop iterations per invocation.
    pub max_iterations: u32,
    /// Maximum tool calls per invocation.
    pub max_tool_calls: u32,
    /// Maximum cost in USD (for hybrid deployments).
    pub max_cost_usd: f64,
}

impl Default for ResourceQuota {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            max_tool_calls: 50,
            max_cost_usd: 0.0,
        }
    }
}

/// Configuration for an external MCP server connection. [F-022]
#[cfg(feature = "agents-mcp")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// MCP server name (used for capability matching).
    pub name: String,
    /// Transport type (stdio, SSE, WebSocket).
    pub transport: McpTransport,
    /// For stdio: command + args to launch the server process.
    #[serde(default)]
    pub command: Vec<String>,
    /// For SSE/WebSocket: URL to connect to.
    pub url: Option<String>,
    /// Tool names granted from this server. `["*"]` grants all.
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// MCP transport mechanism. [F-022]
#[cfg(feature = "agents-mcp")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    /// Subprocess communication via stdin/stdout.
    Stdio,
    /// Server-Sent Events over HTTP.
    Sse,
    /// WebSocket full-duplex.
    WebSocket,
}

impl AgentManifest {
    /// Parse an agent manifest from TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(toml_str)
    }

    /// Validate the manifest for consistency.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.name.is_empty() {
            errors.push("name must not be empty".into());
        }
        if self.resources.max_iterations == 0 {
            errors.push("max_iterations must be > 0".into());
        }
        if self.resources.max_tool_calls == 0 {
            errors.push("max_tool_calls must be > 0".into());
        }
        if self.model.max_tokens == 0 {
            errors.push("max_tokens must be > 0".into());
        }
        if self.model.temperature < 0.0 || self.model.temperature > 2.0 {
            errors.push("temperature must be in [0.0, 2.0]".into());
        }
        if self.privacy == PrivacyTier::Sovereign && self.model.remote_model.is_some() {
            errors.push(
                "sovereign privacy tier cannot use remote_model".into(),
            );
        }
        if self.model.model_repo.is_some() && self.model.model_path.is_some() {
            errors.push(
                "model_repo and model_path are mutually exclusive".into(),
            );
        }
        #[cfg(feature = "agents-mcp")]
        self.validate_mcp_servers(&mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate MCP server configurations (Poka-Yoke).
    #[cfg(feature = "agents-mcp")]
    fn validate_mcp_servers(&self, errors: &mut Vec<String>) {
        for server in &self.mcp_servers {
            if server.name.is_empty() {
                errors.push("MCP server name must not be empty".into());
            }
            if self.privacy == PrivacyTier::Sovereign
                && matches!(server.transport, McpTransport::Sse | McpTransport::WebSocket)
            {
                errors.push(format!(
                    "sovereign privacy tier blocks network MCP transport for server '{}'",
                    server.name,
                ));
            }
            if matches!(server.transport, McpTransport::Stdio) && server.command.is_empty() {
                errors.push(format!(
                    "MCP server '{}' uses stdio transport but has no command",
                    server.name,
                ));
            }
        }
    }
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod tests;
