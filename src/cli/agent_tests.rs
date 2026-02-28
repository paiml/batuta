use super::*;
use tempfile::NamedTempFile;

#[test]
fn test_load_valid_manifest() {
    let toml = r#"
name = "test-agent"
version = "0.1.0"
[model]
system_prompt = "You help."
[resources]
max_iterations = 5
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let manifest = load_manifest(&tmp.path().to_path_buf())
        .expect("should load");
    assert_eq!(manifest.name, "test-agent");
}

#[test]
fn test_load_missing_file() {
    let result =
        load_manifest(&PathBuf::from("/nonexistent/agent.toml"));
    assert!(result.is_err());
}

#[test]
fn test_build_guard_with_override() {
    let manifest = batuta::agent::AgentManifest::default();
    let (max_iter, _) = build_guard(&manifest, Some(99));
    assert_eq!(max_iter, 99);
}

#[test]
fn test_build_guard_from_manifest() {
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.resources.max_iterations = 42;
    let (max_iter, _) = build_guard(&manifest, None);
    assert_eq!(max_iter, 42);
}

#[test]
fn test_validate_command_valid() {
    let toml = r#"
name = "valid"
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 10
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result =
        cmd_agent_validate(&tmp.path().to_path_buf(), false, false);
    assert!(result.is_ok());
}

#[test]
fn test_validate_with_needs_pull() {
    let toml = r#"
name = "repo-agent"
[model]
model_repo = "meta-llama/Llama-3-8B-GGUF"
model_quantization = "q4_k_m"
system_prompt = "hi"
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    // Should pass validation (warns about download)
    let result =
        cmd_agent_validate(&tmp.path().to_path_buf(), false, false);
    assert!(result.is_ok());
}

#[test]
fn test_validate_check_model_no_path() {
    let toml = r#"
name = "no-model"
[model]
system_prompt = "hi"
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result =
        cmd_agent_validate(&tmp.path().to_path_buf(), true, false);
    // Should fail: no model configured
    assert!(result.is_err());
}

#[test]
fn test_validate_check_model_nonexistent() {
    let toml = r#"
name = "missing-model"
[model]
model_path = "/nonexistent/model.gguf"
system_prompt = "hi"
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result =
        cmd_agent_validate(&tmp.path().to_path_buf(), true, false);
    // Should fail: file not found (G0)
    assert!(result.is_err());
}

#[test]
fn test_detect_model_format() {
    // GGUF magic
    let gguf = [0x47u8, 0x47, 0x55, 0x46, 0, 0, 0, 0];
    assert_eq!(detect_model_format(&gguf), "GGUF");

    // APR v2
    let apr = [b'A', b'P', b'R', 0x02, 0, 0, 0, 0];
    assert_eq!(detect_model_format(&apr), "APR v2");

    // SafeTensors (8-byte length + '{')
    let st = [0u8, 0, 0, 0, 0, 0, 0, 0, b'{'];
    assert_eq!(detect_model_format(&st), "SafeTensors");

    // Unknown
    let unknown = [0u8; 4];
    assert_eq!(detect_model_format(&unknown), "unknown");
}

#[test]
fn test_run_command_executes_loop() {
    let toml = r#"
name = "run-test"
version = "1.0.0"
[model]
system_prompt = "You help."
[resources]
max_iterations = 10
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    // No model_path → dry-run mode with MockDriver
    let result = cmd_agent_run(
        &tmp.path().to_path_buf(),
        "hello",
        None,
        false,
        false,
    );
    assert!(result.is_ok());
}

#[test]
fn test_build_driver_no_model_returns_mock() {
    let manifest = batuta::agent::AgentManifest::default();
    let driver = build_driver(&manifest);
    assert!(
        driver.is_ok(),
        "should return MockDriver when no model_path"
    );
}

#[test]
fn test_build_tool_registry_memory() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Memory];
    let registry = build_tool_registry(&manifest);
    assert!(registry.get("memory").is_some());
}

#[test]
fn test_build_tool_registry_compute() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Compute];
    let registry = build_tool_registry(&manifest);
    assert!(registry.get("compute").is_some());
}

#[test]
fn test_build_tool_registry_shell() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Shell {
        allowed_commands: vec!["*".into()],
    }];
    let registry = build_tool_registry(&manifest);
    assert!(registry.get("shell").is_some());
}

#[test]
fn test_build_memory_substrate() {
    let memory = build_memory();
    // Should not panic — returns either TruenoMemory or InMemory
    let _ = memory;
}

#[test]
fn test_run_with_max_iterations_override() {
    let toml = r#"
name = "override-test"
version = "1.0.0"
[model]
system_prompt = "You help."
[resources]
max_iterations = 10
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = cmd_agent_run(
        &tmp.path().to_path_buf(),
        "hello",
        Some(3),
        false,
        false,
    );
    assert!(result.is_ok());
}

#[test]
fn test_validate_command_invalid() {
    let toml = r#"
name = ""
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 0
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result =
        cmd_agent_validate(&tmp.path().to_path_buf(), false, false);
    assert!(result.is_err());
}

#[test]
fn test_sign_and_verify_roundtrip() {
    let toml = r#"
name = "sign-test"
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 10
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");

    let sig_path = tmp.path().with_extension("toml.sig");
    let pk_path = sig_path.with_extension("pub");

    let result = cmd_agent_sign(
        &tmp.path().to_path_buf(),
        Some("tester"),
        Some(sig_path.clone()),
    );
    assert!(result.is_ok(), "sign failed: {result:?}");
    assert!(sig_path.exists(), "signature file not created");
    assert!(pk_path.exists(), "pubkey file not created");

    let result = cmd_agent_verify_sig(
        &tmp.path().to_path_buf(),
        Some(sig_path.clone()),
        &pk_path,
    );
    assert!(result.is_ok(), "verify failed: {result:?}");

    // Clean up
    let _ = std::fs::remove_file(&sig_path);
    let _ = std::fs::remove_file(&pk_path);
}

#[test]
fn test_verify_fails_on_tampered() {
    let toml = r#"
name = "tamper-test"
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 10
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");

    let sig_path = tmp.path().with_extension("toml.sig");
    let pk_path = sig_path.with_extension("pub");

    cmd_agent_sign(
        &tmp.path().to_path_buf(),
        None,
        Some(sig_path.clone()),
    )
    .expect("sign");

    // Tamper with manifest
    std::fs::write(tmp.path(), "name = \"tampered\"")
        .expect("tamper");

    let result = cmd_agent_verify_sig(
        &tmp.path().to_path_buf(),
        Some(sig_path.clone()),
        &pk_path,
    );
    assert!(result.is_err(), "should fail on tampered manifest");

    let _ = std::fs::remove_file(&sig_path);
    let _ = std::fs::remove_file(&pk_path);
}

#[test]
fn test_contracts_command() {
    let result = cmd_agent_contracts();
    assert!(result.is_ok());
}

#[test]
fn test_status_command() {
    let toml = r#"
name = "status-test"
version = "1.0.0"
privacy = "Sovereign"

[model]
max_tokens = 2048
temperature = 0.5
system_prompt = "hi"

[resources]
max_iterations = 15
max_tool_calls = 30
max_cost_usd = 1.50

[[capabilities]]
type = "rag"

[[capabilities]]
type = "memory"
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result =
        cmd_agent_status(&tmp.path().to_path_buf());
    assert!(result.is_ok());
}

#[test]
fn test_status_command_no_capabilities() {
    let toml = r#"
name = "empty-caps"
[model]
system_prompt = "x"
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result =
        cmd_agent_status(&tmp.path().to_path_buf());
    assert!(result.is_ok());
}

#[test]
fn test_pool_command_fan_out() {
    let toml = r#"
name = "pool-agent"
version = "1.0.0"
[model]
system_prompt = "You help."
[resources]
max_iterations = 5
"#;
    let tmp1 = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp1.path(), toml).expect("write");
    let tmp2 = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp2.path(), toml).expect("write");

    let manifests = vec![
        tmp1.path().to_path_buf(),
        tmp2.path().to_path_buf(),
    ];
    let result = cmd_agent_pool(&manifests, "hello", None);
    assert!(result.is_ok());
}

#[test]
fn test_pool_command_with_concurrency() {
    let toml = r#"
name = "pool-concurrent"
[model]
system_prompt = "hi"
[resources]
max_iterations = 3
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");

    let manifests = vec![tmp.path().to_path_buf()];
    let result =
        cmd_agent_pool(&manifests, "test", Some(1));
    assert!(result.is_ok());
}

#[test]
fn test_try_auto_pull_no_repo_is_noop() {
    let manifest =
        batuta::agent::AgentManifest::default();
    // No model_repo set → should be a no-op (Ok)
    let result = try_auto_pull(&manifest);
    assert!(result.is_ok());
}

#[test]
fn test_try_auto_pull_with_repo_errors() {
    let mut manifest =
        batuta::agent::AgentManifest::default();
    manifest.model.model_repo =
        Some("test-org/fake-model".into());
    // apr binary not available → should fail
    let result = try_auto_pull(&manifest);
    assert!(result.is_err());
    let err_msg =
        result.unwrap_err().to_string();
    assert!(
        err_msg.contains("auto-pull failed"),
        "unexpected error: {err_msg}",
    );
}

#[test]
fn test_run_command_with_auto_pull_flag() {
    let toml = r#"
name = "auto-pull-test"
[model]
system_prompt = "hi"
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");

    // auto_pull=true but no model_repo → no-op
    let result = cmd_agent_run(
        &tmp.path().to_path_buf(),
        "hello",
        None,
        false,
        true,
    );
    assert!(result.is_ok());
}

#[test]
fn test_validate_with_check_inference() {
    let toml = r#"
name = "g2-test"
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 5
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    // G2 with MockDriver (no model → dry-run) should pass
    let result = cmd_agent_validate(
        &tmp.path().to_path_buf(),
        false,
        true,
    );
    assert!(
        result.is_ok(),
        "G2 with mock driver should pass: {result:?}",
    );
}

#[test]
fn test_char_entropy() {
    use agent_helpers::char_entropy;

    // Single character repeated → low entropy
    let low = char_entropy("aaaaaaaaaa");
    assert!(low < 0.1, "repeated char entropy: {low}");

    // Normal English text → moderate entropy
    let normal = char_entropy("Hello, I am operational.");
    assert!(
        normal > 2.0 && normal < 5.0,
        "normal text entropy: {normal}",
    );

    // Empty string → zero
    assert_eq!(char_entropy(""), 0.0);
}

#[test]
fn test_truncate_str() {
    use agent_helpers::truncate_str;

    assert_eq!(truncate_str("short", 10), "short");
    assert_eq!(truncate_str("hello world!", 8), "hello...");
    assert_eq!(truncate_str("", 5), "");
}

#[test]
fn test_build_tool_registry_spawn() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![
        Capability::Spawn { max_depth: 2 },
    ];
    // build_tool_registry alone won't register spawn (needs driver)
    let registry = build_tool_registry(&manifest);
    assert!(
        registry.get("spawn_agent").is_none(),
        "spawn requires register_spawn_tool with driver"
    );
}

#[test]
fn test_register_spawn_tool_with_driver() {
    use batuta::agent::capability::Capability;
    use batuta::agent::driver::mock::MockDriver;

    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![
        Capability::Spawn { max_depth: 3 },
    ];
    let mut registry = build_tool_registry(&manifest);
    let driver: std::sync::Arc<dyn batuta::agent::driver::LlmDriver> =
        std::sync::Arc::new(MockDriver::single_response("x"));
    register_spawn_tool(&mut registry, &manifest, driver);
    assert!(registry.get("spawn_agent").is_some());
}

#[test]
#[cfg(feature = "agents-browser")]
fn test_build_tool_registry_browser() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Browser];
    let registry = build_tool_registry(&manifest);
    assert!(registry.get("browser").is_some());
}

#[test]
fn test_build_tool_registry_network() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Network {
        allowed_hosts: vec!["api.example.com".into()],
    }];
    let registry = build_tool_registry(&manifest);
    assert!(registry.get("network").is_some());
}

#[test]
#[cfg(feature = "rag")]
fn test_build_tool_registry_rag() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Rag];
    let registry = build_tool_registry(&manifest);
    assert!(registry.get("rag").is_some());
}

#[test]
fn test_register_inference_tool_with_driver() {
    use batuta::agent::capability::Capability;
    use batuta::agent::driver::mock::MockDriver;

    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Inference];
    let mut registry = build_tool_registry(&manifest);
    let driver: std::sync::Arc<dyn batuta::agent::driver::LlmDriver> =
        std::sync::Arc::new(MockDriver::single_response("x"));
    register_inference_tool(&mut registry, &manifest, driver);
    assert!(registry.get("inference").is_some());
}

#[test]
fn test_register_inference_tool_no_capability() {
    use batuta::agent::driver::mock::MockDriver;

    let manifest = batuta::agent::AgentManifest::default();
    let mut registry = build_tool_registry(&manifest);
    let driver: std::sync::Arc<dyn batuta::agent::driver::LlmDriver> =
        std::sync::Arc::new(MockDriver::single_response("x"));
    register_inference_tool(&mut registry, &manifest, driver);
    assert!(
        registry.get("inference").is_none(),
        "should not register without Inference capability"
    );
}

#[test]
fn test_register_spawn_tool_no_capability() {
    use batuta::agent::driver::mock::MockDriver;

    let manifest = batuta::agent::AgentManifest::default();
    let mut registry = build_tool_registry(&manifest);
    let driver: std::sync::Arc<dyn batuta::agent::driver::LlmDriver> =
        std::sync::Arc::new(MockDriver::single_response("x"));
    register_spawn_tool(&mut registry, &manifest, driver);
    assert!(
        registry.get("spawn_agent").is_none(),
        "should not register without Spawn capability"
    );
}
