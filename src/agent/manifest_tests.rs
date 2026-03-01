//! Core parsing and configuration tests for agent manifest.
//!
//! Covers: TOML parsing, defaults, validation basics, capabilities,
//! MCP server config, model repo fields, path resolution.
//! See `manifest_tests_validation.rs` for edge cases and auto-pull.

use std::path::PathBuf;

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
    let shell_cap = manifest
        .capabilities
        .iter()
        .find(|c| matches!(c, Capability::Shell { .. }))
        .expect("Shell capability");
    if let Capability::Shell { allowed_commands } = shell_cap {
        assert_eq!(allowed_commands.len(), 3);
        assert!(allowed_commands.contains(&"ls".to_string()));
    }
}

#[cfg(feature = "agents-mcp")]
#[test]
fn test_mcp_server_config_parse() {
    let toml = r#"
name = "mcp-agent"
version = "0.1.0"
privacy = "Standard"

[model]
system_prompt = "hi"

[[capabilities]]
type = "memory"

[[mcp_servers]]
name = "code-search"
transport = "stdio"
command = ["node", "server.js"]
capabilities = ["*"]
"#;
    let manifest = AgentManifest::from_toml(toml)
        .expect("parse failed");
    assert_eq!(manifest.mcp_servers.len(), 1);
    assert_eq!(manifest.mcp_servers[0].name, "code-search");
    assert!(matches!(
        manifest.mcp_servers[0].transport,
        McpTransport::Stdio
    ));
    assert!(manifest.validate().is_ok());
}

#[cfg(feature = "agents-mcp")]
#[test]
fn test_mcp_sovereign_blocks_sse() {
    let toml = r#"
name = "sovereign-mcp"
privacy = "Sovereign"

[model]
system_prompt = "hi"

[[capabilities]]
type = "memory"

[[mcp_servers]]
name = "remote-server"
transport = "sse"
url = "https://api.example.com/mcp"
"#;
    let manifest = AgentManifest::from_toml(toml)
        .expect("parse failed");
    let errors = manifest.validate().unwrap_err();
    assert!(errors.iter().any(|e| e.contains("sovereign")));
    assert!(errors.iter().any(|e| e.contains("remote-server")));
}

#[cfg(feature = "agents-mcp")]
#[test]
fn test_mcp_stdio_needs_command() {
    let toml = r#"
name = "stdio-no-cmd"
privacy = "Standard"

[model]
system_prompt = "hi"

[[capabilities]]
type = "memory"

[[mcp_servers]]
name = "broken"
transport = "stdio"
"#;
    let manifest = AgentManifest::from_toml(toml)
        .expect("parse failed");
    let errors = manifest.validate().unwrap_err();
    assert!(errors.iter().any(|e| e.contains("no command")));
}

#[cfg(feature = "agents-mcp")]
#[test]
fn test_mcp_default_empty() {
    let manifest = AgentManifest::default();
    assert!(manifest.mcp_servers.is_empty());
}

#[test]
fn test_model_repo_fields() {
    let toml = r#"
name = "repo-agent"
[model]
model_repo = "meta-llama/Llama-3-8B-GGUF"
model_quantization = "q4_k_m"
system_prompt = "hi"
"#;
    let manifest =
        AgentManifest::from_toml(toml).expect("parse failed");
    assert_eq!(
        manifest.model.model_repo.as_deref(),
        Some("meta-llama/Llama-3-8B-GGUF"),
    );
    assert_eq!(
        manifest.model.model_quantization.as_deref(),
        Some("q4_k_m"),
    );
    assert!(manifest.validate().is_ok());
}

#[test]
fn test_model_repo_and_path_conflict() {
    let mut manifest = AgentManifest::default();
    manifest.model.model_repo =
        Some("meta-llama/Llama-3-8B".into());
    manifest.model.model_path =
        Some("/models/llama.gguf".into());
    let errors = manifest.validate().unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("mutually exclusive")));
}

#[test]
fn test_resolve_model_path_explicit() {
    let mut config = ModelConfig::default();
    config.model_path = Some("/models/llama.gguf".into());
    let resolved = config.resolve_model_path();
    assert_eq!(
        resolved,
        Some(PathBuf::from("/models/llama.gguf"))
    );
}

#[test]
fn test_resolve_model_path_from_repo() {
    let mut config = ModelConfig::default();
    config.model_repo =
        Some("meta-llama/Llama-3-8B-GGUF".into());
    config.model_quantization = Some("q4_k_m".into());
    let resolved = config.resolve_model_path();
    assert!(resolved.is_some());
    let path = resolved.expect("should resolve");
    assert!(path
        .to_string_lossy()
        .contains("meta-llama--Llama-3-8B-GGUF"));
    assert!(path.to_string_lossy().contains("q4_k_m"));
}
