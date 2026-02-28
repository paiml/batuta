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
            max_tokens: 4096,
            temperature: 0.3,
            system_prompt: "You are a helpful assistant.".into(),
            context_window: None,
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

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_TOML: &str = r#"
name = "test-agent"
version = "0.1.0"
description = "A test agent"
privacy = "Sovereign"

[model]
max_tokens = 2048
temperature = 0.5
system_prompt = "You are a test agent."

[resources]
max_iterations = 10
max_tool_calls = 20
max_cost_usd = 0.0

[[capabilities]]
type = "rag"

[[capabilities]]
type = "memory"
"#;

    #[test]
    fn test_parse_minimal_toml() {
        let manifest = AgentManifest::from_toml(MINIMAL_TOML)
            .expect("parse failed");
        assert_eq!(manifest.name, "test-agent");
        assert_eq!(manifest.model.max_tokens, 2048);
        assert_eq!(manifest.resources.max_iterations, 10);
        assert_eq!(manifest.capabilities.len(), 2);
        assert_eq!(manifest.privacy, PrivacyTier::Sovereign);
    }

    #[test]
    fn test_defaults() {
        let manifest = AgentManifest::default();
        assert_eq!(manifest.name, "unnamed-agent");
        assert_eq!(manifest.model.max_tokens, 4096);
        assert_eq!(manifest.resources.max_iterations, 20);
        assert_eq!(manifest.privacy, PrivacyTier::Sovereign);
    }

    #[test]
    fn test_validate_valid() {
        let manifest = AgentManifest::from_toml(MINIMAL_TOML)
            .expect("parse failed");
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_validate_empty_name() {
        let mut manifest = AgentManifest::default();
        manifest.name = String::new();
        let errors = manifest.validate().unwrap_err();
        assert!(errors.iter().any(|e| e.contains("name")));
    }

    #[test]
    fn test_validate_zero_iterations() {
        let mut manifest = AgentManifest::default();
        manifest.resources.max_iterations = 0;
        let errors = manifest.validate().unwrap_err();
        assert!(errors.iter().any(|e| e.contains("max_iterations")));
    }

    #[test]
    fn test_validate_bad_temperature() {
        let mut manifest = AgentManifest::default();
        manifest.model.temperature = 3.0;
        let errors = manifest.validate().unwrap_err();
        assert!(errors.iter().any(|e| e.contains("temperature")));
    }

    #[test]
    fn test_validate_sovereign_with_remote() {
        let mut manifest = AgentManifest::default();
        manifest.privacy = PrivacyTier::Sovereign;
        manifest.model.remote_model = Some("gpt-4".into());
        let errors = manifest.validate().unwrap_err();
        assert!(errors.iter().any(|e| e.contains("sovereign")));
    }

    #[test]
    fn test_model_path_optional() {
        let toml = r#"
name = "no-model"
[model]
system_prompt = "hi"
"#;
        let manifest = AgentManifest::from_toml(toml)
            .expect("parse failed");
        assert!(manifest.model.model_path.is_none());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let manifest = AgentManifest::default();
        let toml_str =
            toml::to_string_pretty(&manifest).expect("serialize failed");
        let back = AgentManifest::from_toml(&toml_str)
            .expect("parse roundtrip failed");
        assert_eq!(back.name, manifest.name);
        assert_eq!(back.model.max_tokens, manifest.model.max_tokens);
    }

    #[test]
    fn test_parse_full_capabilities() {
        let toml = r#"
name = "full-agent"
version = "0.2.0"

[model]
max_tokens = 4096
system_prompt = "hi"

[resources]
max_iterations = 30
max_tool_calls = 100

[[capabilities]]
type = "memory"

[[capabilities]]
type = "rag"

[[capabilities]]
type = "shell"
allowed_commands = ["ls", "cat", "echo"]

[[capabilities]]
type = "compute"

[[capabilities]]
type = "browser"

privacy = "Sovereign"
"#;
        let manifest =
            AgentManifest::from_toml(toml).expect("parse failed");
        assert_eq!(manifest.capabilities.len(), 5);
        assert!(manifest.validate().is_ok());
        // Verify Shell has correct commands
        let shell_cap = manifest
            .capabilities
            .iter()
            .find(|c| {
                matches!(c, Capability::Shell { .. })
            })
            .expect("Shell capability");
        if let Capability::Shell { allowed_commands } = shell_cap {
            assert_eq!(allowed_commands.len(), 3);
            assert!(allowed_commands.contains(&"ls".to_string()));
        }
    }

    #[test]
    fn test_example_manifests_valid() {
        // Validate that example manifests in the repo parse correctly
        let basic = include_str!("../../examples/agent.toml");
        let manifest =
            AgentManifest::from_toml(basic).expect("basic manifest parse");
        assert!(manifest.validate().is_ok());

        let full = include_str!("../../examples/agent-full.toml");
        let manifest =
            AgentManifest::from_toml(full).expect("full manifest parse");
        assert!(manifest.validate().is_ok());
        assert_eq!(manifest.capabilities.len(), 5);
    }
}
