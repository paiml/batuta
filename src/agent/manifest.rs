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
