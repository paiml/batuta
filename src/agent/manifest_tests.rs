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

#[test]
fn test_resolve_model_path_default_quant() {
    let mut config = ModelConfig::default();
    config.model_repo = Some("test/model".into());
    let resolved = config.resolve_model_path();
    assert!(resolved.is_some());
    assert!(resolved
        .expect("path")
        .to_string_lossy()
        .contains("q4_k_m"));
}

#[test]
fn test_resolve_model_path_none() {
    let config = ModelConfig::default();
    assert!(config.resolve_model_path().is_none());
}

#[test]
fn test_needs_pull_no_repo() {
    let config = ModelConfig::default();
    assert!(config.needs_pull().is_none());
}

#[test]
fn test_needs_pull_with_explicit_path() {
    let mut config = ModelConfig::default();
    config.model_path = Some("/models/llama.gguf".into());
    config.model_repo =
        Some("meta-llama/Llama-3-8B".into());
    assert!(config.needs_pull().is_none());
}

#[test]
fn test_needs_pull_with_repo() {
    let mut config = ModelConfig::default();
    config.model_repo =
        Some("meta-llama/Llama-3-8B-GGUF".into());
    assert_eq!(
        config.needs_pull(),
        Some("meta-llama/Llama-3-8B-GGUF"),
    );
}

#[test]
fn test_auto_pull_no_repo() {
    let config = ModelConfig::default();
    let err = config.auto_pull(10).unwrap_err();
    assert!(
        matches!(err, AutoPullError::NoRepo),
        "expected NoRepo, got: {err}",
    );
}

#[test]
fn test_auto_pull_error_display() {
    let no_repo = AutoPullError::NoRepo;
    assert!(no_repo.to_string().contains("no model_repo"));

    let not_installed = AutoPullError::NotInstalled;
    assert!(not_installed.to_string().contains("not found"));
    assert!(not_installed.to_string().contains("apr-cli"));

    let subprocess = AutoPullError::Subprocess("boom".into());
    assert_eq!(subprocess.to_string(), "boom");

    let io_err = AutoPullError::Io("disk full".into());
    assert_eq!(io_err.to_string(), "disk full");
}

#[test]
fn test_auto_pull_apr_not_in_path() {
    // With model_repo set but apr not in PATH,
    // auto_pull should return NotInstalled
    let mut config = ModelConfig::default();
    config.model_repo =
        Some("test-org/nonexistent-model".into());

    // This will fail because apr binary is not in PATH
    // (or if it is, the repo doesn't exist — either way, error)
    let result = config.auto_pull(5);
    assert!(result.is_err());
}

#[test]
fn test_validate_zero_tool_calls() {
    let mut manifest = AgentManifest::default();
    manifest.resources.max_tool_calls = 0;
    let errors = manifest.validate().unwrap_err();
    assert!(errors.iter().any(|e| e.contains("max_tool_calls")));
}

#[test]
fn test_validate_zero_max_tokens() {
    let mut manifest = AgentManifest::default();
    manifest.model.max_tokens = 0;
    let errors = manifest.validate().unwrap_err();
    assert!(errors.iter().any(|e| e.contains("max_tokens")));
}

#[test]
fn test_validate_negative_temperature() {
    let mut manifest = AgentManifest::default();
    manifest.model.temperature = -0.5;
    let errors = manifest.validate().unwrap_err();
    assert!(errors.iter().any(|e| e.contains("temperature")));
}

#[test]
fn test_validate_multiple_errors() {
    let mut manifest = AgentManifest::default();
    manifest.name = String::new();
    manifest.resources.max_iterations = 0;
    manifest.resources.max_tool_calls = 0;
    let errors = manifest.validate().unwrap_err();
    assert!(errors.len() >= 3);
}

#[test]
fn test_auto_pull_error_is_std_error() {
    let err: Box<dyn std::error::Error> =
        Box::new(AutoPullError::NoRepo);
    assert!(err.to_string().contains("no model_repo"));
}

#[test]
fn test_resource_quota_defaults() {
    let q = ResourceQuota::default();
    assert_eq!(q.max_iterations, 20);
    assert_eq!(q.max_tool_calls, 50);
    assert_eq!(q.max_cost_usd, 0.0);
    assert!(q.max_tokens_budget.is_none());
}

#[test]
fn test_resource_quota_with_token_budget() {
    let toml = r#"
name = "budget-agent"
[model]
system_prompt = "hi"
[resources]
max_iterations = 10
max_tool_calls = 20
max_tokens_budget = 100000
"#;
    let manifest =
        AgentManifest::from_toml(toml).expect("parse");
    assert_eq!(
        manifest.resources.max_tokens_budget,
        Some(100000),
    );
}

#[test]
fn test_model_config_defaults() {
    let config = ModelConfig::default();
    assert!(config.model_path.is_none());
    assert!(config.remote_model.is_none());
    assert!(config.model_repo.is_none());
    assert!(config.model_quantization.is_none());
    assert_eq!(config.max_tokens, 4096);
    assert!((config.temperature - 0.3).abs() < 0.001);
    assert!(
        config
            .system_prompt
            .contains("helpful assistant")
    );
    assert!(config.context_window.is_none());
}

#[test]
fn test_example_manifests_valid() {
    let basic = include_str!("../../examples/agent.toml");
    let manifest = AgentManifest::from_toml(basic)
        .expect("basic manifest parse");
    assert!(manifest.validate().is_ok());

    let full = include_str!("../../examples/agent-full.toml");
    let manifest = AgentManifest::from_toml(full)
        .expect("full manifest parse");
    assert!(manifest.validate().is_ok());
    assert_eq!(manifest.capabilities.len(), 5);
}
