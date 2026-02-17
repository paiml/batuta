//! Local sequential pipeline executor (PB-005)
//!
//! Orchestrates: parse → build DAG → load lock → for each stage in topo order:
//! resolve template → hash → check cache → execute → hash outputs → update lock → log event.
//!
//! Implements Jidoka: stop on first failure (only policy in Phase 1).
//! Remote targets return error: "Remote execution requires Phase 2 (PB-006)".

use super::cache::{self, CacheDecision};
use super::dag;
use super::eventlog;
use super::hasher;
use super::parser;
use super::template;
use super::types::*;
use anyhow::{bail, Context, Result};
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Configuration for a playbook run
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RunConfig {
    /// Path to the playbook YAML
    pub playbook_path: PathBuf,

    /// Only run these stages (None = all)
    pub stage_filter: Option<Vec<String>>,

    /// Force re-run (ignore cache)
    pub force: bool,

    /// Dry-run mode (Phase 6, no-op in Phase 1)
    pub dry_run: bool,

    /// Parameter overrides from CLI
    pub param_overrides: HashMap<String, serde_yaml::Value>,
}

/// Result of a playbook run
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RunResult {
    pub stages_run: u32,
    pub stages_cached: u32,
    pub stages_failed: u32,
    pub total_duration: std::time::Duration,
    pub lock_file: Option<LockFile>,
}

/// Execute a playbook
pub async fn run_playbook(config: &RunConfig) -> Result<RunResult> {
    let total_start = Instant::now();

    // Parse
    let mut playbook = parser::parse_playbook_file(&config.playbook_path)?;
    let warnings = parser::validate_playbook(&playbook)?;
    for w in &warnings {
        tracing::warn!("playbook validation: {}", w);
    }

    // Apply parameter overrides
    for (k, v) in &config.param_overrides {
        playbook.params.insert(k.clone(), v.clone());
    }

    // Build DAG
    let dag_result = dag::build_dag(&playbook)?;
    let run_id = eventlog::generate_run_id();

    // Log run started
    let _ = eventlog::append_event(
        &config.playbook_path,
        PipelineEvent::RunStarted {
            playbook: playbook.name.clone(),
            run_id: run_id.clone(),
            batuta_version: env!("CARGO_PKG_VERSION").to_string(),
        },
    );

    // Load existing lock file
    let existing_lock = cache::load_lock_file(&config.playbook_path)?;

    // Build new lock file
    let mut lock = LockFile {
        schema: "1.0".to_string(),
        playbook: playbook.name.clone(),
        generated_at: eventlog::now_iso8601(),
        generator: format!("batuta {}", env!("CARGO_PKG_VERSION")),
        blake3_version: "1.8".to_string(),
        params_hash: None,
        stages: IndexMap::new(),
    };

    // Copy over stage locks from existing lock for stages we won't re-run
    if let Some(ref el) = existing_lock {
        for (name, stage_lock) in &el.stages {
            lock.stages.insert(name.clone(), stage_lock.clone());
        }
    }

    let mut stages_run = 0u32;
    let mut stages_cached = 0u32;
    let mut stages_failed = 0u32;
    let mut rerun_stages: HashSet<String> = HashSet::new();

    // Determine which stages to run
    let stages_to_run: Vec<String> = if let Some(ref filter) = config.stage_filter {
        dag_result
            .topo_order
            .iter()
            .filter(|s| filter.contains(s))
            .cloned()
            .collect()
    } else {
        dag_result.topo_order.clone()
    };

    // Execute stages in topological order
    for stage_name in &stages_to_run {
        let stage = match playbook.stages.get(stage_name) {
            Some(s) => s,
            None => bail!("stage '{}' not found in playbook", stage_name),
        };

        // Check for remote target (Phase 2) — allow localhost (M8 fix)
        if let Some(ref target_name) = stage.target {
            if let Some(target) = playbook.targets.get(target_name) {
                if let Some(ref host) = target.host {
                    let is_local = host == "localhost" || host == "127.0.0.1";
                    if !is_local {
                        bail!(
                            "stage '{}' targets remote host '{}': Remote execution requires Phase 2 (PB-006)",
                            stage_name,
                            target_name
                        );
                    }
                }
            }
        }

        // Frozen stages always report CACHED unless --force (H8 fix)
        if stage.frozen && !config.force {
            stages_cached += 1;
            let _ = eventlog::append_event(
                &config.playbook_path,
                PipelineEvent::StageCached {
                    stage: stage_name.clone(),
                    cache_key: "frozen".to_string(),
                    reason: "stage is frozen".to_string(),
                },
            );
            println!("  {} CACHED (frozen)", stage_name);
            continue;
        }

        // Shell purification warning (Phase 2)
        tracing::debug!(
            "stage '{}': executing via raw sh -c (bashrs purification deferred to Phase 2)",
            stage_name
        );

        // Resolve template
        let resolved_cmd = template::resolve_template(
            &stage.cmd,
            &playbook.params,
            &stage.params,
            &stage.deps,
            &stage.outs,
        )
        .with_context(|| format!("stage '{}' template resolution failed", stage_name))?;

        // Hash command
        let cmd_hash = hasher::hash_cmd(&resolved_cmd);

        // Hash dependencies
        let mut dep_hashes: Vec<(String, String)> = Vec::new();
        let mut dep_locks: Vec<DepLock> = Vec::new();
        for dep in &stage.deps {
            let dep_path = Path::new(&dep.path);
            if dep_path.exists() {
                let result = hasher::hash_dep(dep_path)?;
                dep_hashes.push((dep.path.clone(), result.hash.clone()));
                dep_locks.push(DepLock {
                    path: dep.path.clone(),
                    hash: result.hash,
                    file_count: Some(result.file_count),
                    total_bytes: Some(result.total_bytes),
                });
            } else {
                // Dep doesn't exist yet — always a miss
                dep_hashes.push((dep.path.clone(), String::new()));
                dep_locks.push(DepLock {
                    path: dep.path.clone(),
                    hash: String::new(),
                    file_count: None,
                    total_bytes: None,
                });
            }
        }

        let deps_combined = hasher::combine_deps_hashes(
            &dep_hashes
                .iter()
                .map(|(_, h)| h.clone())
                .collect::<Vec<_>>(),
        );

        // Hash params — use effective_param_keys for granular invalidation (I3 invariant)
        let param_refs = hasher::effective_param_keys(&stage.params, &stage.cmd);
        let params_hash = hasher::hash_params(&playbook.params, &param_refs)?;

        // Compute cache key
        let cache_key = hasher::compute_cache_key(&cmd_hash, &deps_combined, &params_hash);

        // Determine upstream reruns for this stage
        let upstream_reruns: Vec<String> = dag_result
            .predecessors
            .get(stage_name)
            .map(|preds| {
                preds
                    .iter()
                    .filter(|p| rerun_stages.contains(p.as_str()))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        // Check cache
        let decision = cache::check_cache(
            stage_name,
            &cache_key,
            &cmd_hash,
            &dep_hashes,
            &params_hash,
            &existing_lock,
            config.force,
            &upstream_reruns,
        );

        match decision {
            CacheDecision::Hit => {
                stages_cached += 1;
                let _ = eventlog::append_event(
                    &config.playbook_path,
                    PipelineEvent::StageCached {
                        stage: stage_name.clone(),
                        cache_key: cache_key.clone(),
                        reason: "cache_key matches lock".to_string(),
                    },
                );
                println!("  {} CACHED", stage_name);
                continue;
            }
            CacheDecision::Miss { ref reasons } => {
                let reason_str: Vec<String> = reasons.iter().map(|r| r.to_string()).collect();
                let _ = eventlog::append_event(
                    &config.playbook_path,
                    PipelineEvent::StageStarted {
                        stage: stage_name.clone(),
                        target: stage.target.clone().unwrap_or_else(|| "local".to_string()),
                        cache_miss_reason: reason_str.join("; "),
                    },
                );
                println!("  {} RUNNING ({})", stage_name, reason_str.join("; "));
            }
        }

        // Capture start timestamp BEFORE execution (M5 fix)
        let started_at = eventlog::now_iso8601();
        let stage_start = Instant::now();

        // Execute the command
        let exec_result = execute_command(&resolved_cmd).await;
        let duration = stage_start.elapsed();
        let completed_at = eventlog::now_iso8601();

        match exec_result {
            Ok(()) => {
                stages_run += 1;
                rerun_stages.insert(stage_name.clone());

                // Hash outputs
                let mut out_locks: Vec<OutLock> = Vec::new();
                for out in &stage.outs {
                    let out_path = Path::new(&out.path);
                    if out_path.exists() {
                        let result = hasher::hash_dep(out_path)?;
                        out_locks.push(OutLock {
                            path: out.path.clone(),
                            hash: result.hash,
                            file_count: Some(result.file_count),
                            total_bytes: Some(result.total_bytes),
                            remote: out.remote.clone(),
                        });
                    } else {
                        tracing::warn!(
                            "stage '{}' completed but output '{}' does not exist",
                            stage_name,
                            out.path
                        );
                        out_locks.push(OutLock {
                            path: out.path.clone(),
                            hash: String::new(),
                            file_count: None,
                            total_bytes: None,
                            remote: out.remote.clone(),
                        });
                    }
                }

                let outs_hash = if out_locks.is_empty() {
                    None
                } else {
                    Some(hasher::combine_deps_hashes(
                        &out_locks.iter().map(|o| o.hash.clone()).collect::<Vec<_>>(),
                    ))
                };

                // Update lock
                lock.stages.insert(
                    stage_name.clone(),
                    StageLock {
                        status: StageStatus::Completed,
                        started_at: Some(started_at),
                        completed_at: Some(completed_at),
                        duration_seconds: Some(duration.as_secs_f64()),
                        target: stage.target.clone(),
                        deps: dep_locks,
                        params_hash: Some(params_hash.clone()),
                        outs: out_locks,
                        cmd_hash: Some(cmd_hash.clone()),
                        cache_key: Some(cache_key.clone()),
                    },
                );

                let _ = eventlog::append_event(
                    &config.playbook_path,
                    PipelineEvent::StageCompleted {
                        stage: stage_name.clone(),
                        duration_seconds: duration.as_secs_f64(),
                        outs_hash,
                    },
                );

                println!(
                    "  {} COMPLETED ({:.1}s)",
                    stage_name,
                    duration.as_secs_f64()
                );
            }
            Err(e) => {
                stages_failed += 1;

                let (exit_code, error_msg) = match e.downcast_ref::<CommandError>() {
                    Some(ce) => (ce.exit_code, ce.stderr.clone()),
                    None => (None, e.to_string()),
                };

                lock.stages.insert(
                    stage_name.clone(),
                    StageLock {
                        status: StageStatus::Failed,
                        started_at: Some(started_at),
                        completed_at: Some(completed_at),
                        duration_seconds: Some(duration.as_secs_f64()),
                        target: stage.target.clone(),
                        deps: dep_locks,
                        params_hash: Some(params_hash.clone()),
                        outs: vec![],
                        cmd_hash: Some(cmd_hash.clone()),
                        cache_key: None,
                    },
                );

                let _ = eventlog::append_event(
                    &config.playbook_path,
                    PipelineEvent::StageFailed {
                        stage: stage_name.clone(),
                        exit_code,
                        error: error_msg.clone(),
                        retry_attempt: None,
                    },
                );

                eprintln!("  {} FAILED: {}", stage_name, error_msg);

                // Jidoka: stop on first failure
                if playbook.policy.failure == FailurePolicy::StopOnFirst {
                    let _ = eventlog::append_event(
                        &config.playbook_path,
                        PipelineEvent::RunFailed {
                            playbook: playbook.name.clone(),
                            run_id: run_id.clone(),
                            error: format!("stage '{}' failed (Jidoka: stop_on_first)", stage_name),
                        },
                    );

                    // Save partial lock
                    if playbook.policy.lock_file {
                        lock.generated_at = eventlog::now_iso8601();
                        cache::save_lock_file(&lock, &config.playbook_path)?;
                    }

                    bail!(
                        "stage '{}' failed: {} (Jidoka: stop_on_first policy)",
                        stage_name,
                        error_msg
                    );
                }
            }
        }
    }

    let total_duration = total_start.elapsed();

    // Log run completed
    let _ = eventlog::append_event(
        &config.playbook_path,
        PipelineEvent::RunCompleted {
            playbook: playbook.name.clone(),
            run_id,
            stages_run,
            stages_cached,
            stages_failed,
            total_seconds: total_duration.as_secs_f64(),
        },
    );

    // Save lock file
    if playbook.policy.lock_file {
        lock.generated_at = eventlog::now_iso8601();
        // Compute global params hash
        let all_param_keys: Vec<String> = playbook.params.keys().cloned().collect();
        lock.params_hash = Some(hasher::hash_params(&playbook.params, &all_param_keys)?);
        cache::save_lock_file(&lock, &config.playbook_path)?;
    }

    Ok(RunResult {
        stages_run,
        stages_cached,
        stages_failed,
        total_duration,
        lock_file: Some(lock),
    })
}

/// Command execution error with exit code and stderr
#[derive(Debug, thiserror::Error)]
#[error("command failed (exit code: {exit_code:?}): {stderr}")]
struct CommandError {
    exit_code: Option<i32>,
    stderr: String,
}

/// Execute a shell command via `sh -c`
async fn execute_command(cmd: &str) -> Result<()> {
    let output = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
        .await
        .context("failed to spawn command")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(CommandError {
            exit_code: output.status.code(),
            stderr: if stderr.is_empty() {
                format!("exit code {}", output.status.code().unwrap_or(-1))
            } else {
                stderr
            },
        }
        .into());
    }

    Ok(())
}

/// Validate a playbook without executing (for `batuta playbook validate`)
pub fn validate_only(playbook_path: &Path) -> Result<(Playbook, Vec<ValidationWarning>)> {
    let playbook = parser::parse_playbook_file(playbook_path)?;
    let warnings = parser::validate_playbook(&playbook)?;
    let _ = dag::build_dag(&playbook)?; // Validates DAG (no cycles, etc.)
    Ok((playbook, warnings))
}

/// Show playbook status (for `batuta playbook status`)
pub fn show_status(playbook_path: &Path) -> Result<()> {
    let playbook = parser::parse_playbook_file(playbook_path)?;
    let lock = cache::load_lock_file(playbook_path)?;

    println!("Playbook: {} ({})", playbook.name, playbook_path.display());
    println!("Version: {}", playbook.version);
    println!("Stages: {}", playbook.stages.len());

    if let Some(ref lock) = lock {
        println!("\nLock file: {} ({})", lock.generator, lock.generated_at);
        println!("{}", "-".repeat(60));
        for (name, _stage) in &playbook.stages {
            if let Some(stage_lock) = lock.stages.get(name) {
                let status_str = match stage_lock.status {
                    StageStatus::Completed => "COMPLETED",
                    StageStatus::Failed => "FAILED",
                    StageStatus::Cached => "CACHED",
                    StageStatus::Running => "RUNNING",
                    StageStatus::Pending => "PENDING",
                    StageStatus::Hashing => "HASHING",
                    StageStatus::Validating => "VALIDATING",
                };
                let duration = stage_lock
                    .duration_seconds
                    .map(|d| format!("{:.1}s", d))
                    .unwrap_or_else(|| "-".to_string());
                println!("  {:20} {:12} {}", name, status_str, duration);
            } else {
                println!("  {:20} {:12}", name, "NOT RUN");
            }
        }
    } else {
        println!("\nNo lock file found (pipeline has not been run)");
    }

    Ok(())
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn test_PB005_run_config_defaults() {
        let config = RunConfig {
            playbook_path: PathBuf::from("/tmp/test.yaml"),
            stage_filter: None,
            force: false,
            dry_run: false,
            param_overrides: HashMap::new(),
        };
        assert!(!config.force);
        assert!(!config.dry_run);
        assert!(config.stage_filter.is_none());
    }

    #[tokio::test]
    async fn test_PB005_execute_command_success() {
        let result = execute_command("echo hello").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_PB005_execute_command_failure() {
        let result = execute_command("exit 1").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        let cmd_err = err.downcast_ref::<CommandError>().unwrap();
        assert_eq!(cmd_err.exit_code, Some(1));
    }

    #[tokio::test]
    async fn test_PB005_execute_command_with_output() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.txt");
        let cmd = format!("echo hello > {}", out.display());
        execute_command(&cmd).await.unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        assert_eq!(content.trim(), "hello");
    }

    #[test]
    fn test_PB005_validate_only() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            r#"
version: "1.0"
name: test
params: {}
targets: {}
stages:
  hello:
    cmd: "echo hello"
    deps: []
    outs:
      - path: /tmp/out.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
        )
        .unwrap();

        let (pb, warnings) = validate_only(&yaml_path).unwrap();
        assert_eq!(pb.name, "test");
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_PB005_validate_only_with_cycle() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            r#"
version: "1.0"
name: cycle
params: {}
targets: {}
stages:
  a:
    cmd: "echo a"
    deps:
      - path: /tmp/b.txt
    outs:
      - path: /tmp/a.txt
  b:
    cmd: "echo b"
    deps:
      - path: /tmp/a.txt
    outs:
      - path: /tmp/b.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
        )
        .unwrap();

        let err = validate_only(&yaml_path).unwrap_err();
        assert!(err.to_string().contains("cycle"));
    }

    #[tokio::test]
    async fn test_PB005_run_simple_pipeline() {
        let dir = tempfile::tempdir().unwrap();
        let out_dir = dir.path().join("outputs");
        std::fs::create_dir(&out_dir).unwrap();

        let out_file = out_dir.join("hello.txt");
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            format!(
                r#"
version: "1.0"
name: simple-test
params: {{}}
targets: {{}}
stages:
  hello:
    cmd: "echo hello > {}"
    deps: []
    outs:
      - path: {}
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
                out_file.display(),
                out_file.display(),
            ),
        )
        .unwrap();

        let config = RunConfig {
            playbook_path: yaml_path.clone(),
            stage_filter: None,
            force: false,
            dry_run: false,
            param_overrides: HashMap::new(),
        };

        let result = run_playbook(&config).await.unwrap();
        assert_eq!(result.stages_run, 1);
        assert_eq!(result.stages_cached, 0);
        assert_eq!(result.stages_failed, 0);

        // Output file should exist
        assert!(out_file.exists());
        assert_eq!(std::fs::read_to_string(&out_file).unwrap().trim(), "hello");

        // Lock file should exist
        let lock_path = cache::lock_file_path(&yaml_path);
        assert!(lock_path.exists());
    }

    #[tokio::test]
    async fn test_PB005_cached_rerun() {
        let dir = tempfile::tempdir().unwrap();
        let out_file = dir.path().join("out.txt");
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            format!(
                r#"
version: "1.0"
name: cache-test
params: {{}}
targets: {{}}
stages:
  write:
    cmd: "echo cached > {}"
    deps: []
    outs:
      - path: {}
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
                out_file.display(),
                out_file.display(),
            ),
        )
        .unwrap();

        let config = RunConfig {
            playbook_path: yaml_path.clone(),
            stage_filter: None,
            force: false,
            dry_run: false,
            param_overrides: HashMap::new(),
        };

        // First run: executes
        let r1 = run_playbook(&config).await.unwrap();
        assert_eq!(r1.stages_run, 1);
        assert_eq!(r1.stages_cached, 0);

        // Second run: cached
        let r2 = run_playbook(&config).await.unwrap();
        assert_eq!(r2.stages_run, 0);
        assert_eq!(r2.stages_cached, 1);
    }

    #[tokio::test]
    async fn test_PB005_jidoka_stop_on_failure() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            r#"
version: "1.0"
name: fail-test
params: {}
targets: {}
stages:
  fail:
    cmd: "exit 1"
    deps: []
    outs: []
  after_fail:
    cmd: "echo should-not-run"
    deps: []
    outs: []
    after:
      - fail
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
        )
        .unwrap();

        let config = RunConfig {
            playbook_path: yaml_path,
            stage_filter: None,
            force: false,
            dry_run: false,
            param_overrides: HashMap::new(),
        };

        let err = run_playbook(&config).await.unwrap_err();
        assert!(err.to_string().contains("Jidoka"));
    }

    #[test]
    fn test_PB005_show_status_no_lock() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            r#"
version: "1.0"
name: status-test
params: {}
targets: {}
stages:
  hello:
    cmd: "echo hello"
    deps: []
    outs: []
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
        )
        .unwrap();

        // No lock file exists — should print "No lock file found"
        let result = show_status(&yaml_path);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_PB005_show_status_with_lock() {
        let dir = tempfile::tempdir().unwrap();
        let out_file = dir.path().join("out.txt");
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            format!(
                r#"
version: "1.0"
name: status-lock-test
params: {{}}
targets: {{}}
stages:
  hello:
    cmd: "echo hi > {}"
    deps: []
    outs:
      - path: {}
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
                out_file.display(),
                out_file.display(),
            ),
        )
        .unwrap();

        // Run to create lock file
        let config = RunConfig {
            playbook_path: yaml_path.clone(),
            stage_filter: None,
            force: false,
            dry_run: false,
            param_overrides: HashMap::new(),
        };
        run_playbook(&config).await.unwrap();

        // Now show_status should display lock info
        let result = show_status(&yaml_path);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_PB005_frozen_stage_cached() {
        let dir = tempfile::tempdir().unwrap();
        let out_file = dir.path().join("frozen_out.txt");
        // Pre-create the output to avoid "output missing" issues
        std::fs::write(&out_file, "frozen content").unwrap();
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            format!(
                r#"
version: "1.0"
name: frozen-test
params: {{}}
targets: {{}}
stages:
  frozen_stage:
    cmd: "echo should-not-run > {}"
    deps: []
    outs:
      - path: {}
    frozen: true
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
                out_file.display(),
                out_file.display(),
            ),
        )
        .unwrap();

        let config = RunConfig {
            playbook_path: yaml_path.clone(),
            stage_filter: None,
            force: false,
            dry_run: false,
            param_overrides: HashMap::new(),
        };

        let result = run_playbook(&config).await.unwrap();
        // Frozen stage should be cached without ever running
        assert_eq!(result.stages_cached, 1);
        assert_eq!(result.stages_run, 0);
        // Output should still have original content (command never ran)
        assert_eq!(
            std::fs::read_to_string(&out_file).unwrap(),
            "frozen content"
        );
    }

    #[tokio::test]
    async fn test_PB005_frozen_stage_force_overrides() {
        let dir = tempfile::tempdir().unwrap();
        let out_file = dir.path().join("frozen_force.txt");
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            format!(
                r#"
version: "1.0"
name: frozen-force-test
params: {{}}
targets: {{}}
stages:
  frozen_stage:
    cmd: "echo forced > {}"
    deps: []
    outs:
      - path: {}
    frozen: true
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
                out_file.display(),
                out_file.display(),
            ),
        )
        .unwrap();

        let config = RunConfig {
            playbook_path: yaml_path.clone(),
            stage_filter: None,
            force: true,
            dry_run: false,
            param_overrides: HashMap::new(),
        };

        let result = run_playbook(&config).await.unwrap();
        // --force should override frozen
        assert_eq!(result.stages_run, 1);
        assert_eq!(result.stages_cached, 0);
        assert_eq!(std::fs::read_to_string(&out_file).unwrap().trim(), "forced");
    }

    #[tokio::test]
    async fn test_PB005_param_overrides() {
        let dir = tempfile::tempdir().unwrap();
        let out_file = dir.path().join("param_out.txt");
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            format!(
                r#"
version: "1.0"
name: param-override-test
params:
  greeting: "default"
targets: {{}}
stages:
  write:
    cmd: "echo '{{{{params.greeting}}}}' > {}"
    deps: []
    outs:
      - path: {}
    params:
      - greeting
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
                out_file.display(),
                out_file.display(),
            ),
        )
        .unwrap();

        // First run with default param
        let config = RunConfig {
            playbook_path: yaml_path.clone(),
            stage_filter: None,
            force: false,
            dry_run: false,
            param_overrides: HashMap::new(),
        };
        let r1 = run_playbook(&config).await.unwrap();
        assert_eq!(r1.stages_run, 1);
        assert_eq!(
            std::fs::read_to_string(&out_file).unwrap().trim(),
            "default"
        );

        // Second run with override
        let mut overrides = HashMap::new();
        overrides.insert(
            "greeting".to_string(),
            serde_yaml::Value::String("overridden".to_string()),
        );
        let config2 = RunConfig {
            playbook_path: yaml_path.clone(),
            stage_filter: None,
            force: false,
            dry_run: false,
            param_overrides: overrides,
        };
        let r2 = run_playbook(&config2).await.unwrap();
        // Should re-run because param changed
        assert_eq!(r2.stages_run, 1);
        assert_eq!(
            std::fs::read_to_string(&out_file).unwrap().trim(),
            "overridden"
        );
    }

    #[tokio::test]
    async fn test_PB005_remote_target_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            r#"
version: "1.0"
name: remote-test
params: {}
targets:
  gpu-box:
    host: "gpu-box.local"
    ssh_user: noah
stages:
  remote_stage:
    cmd: "echo remote"
    deps: []
    outs: []
    target: gpu-box
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
        )
        .unwrap();

        let config = RunConfig {
            playbook_path: yaml_path,
            stage_filter: None,
            force: false,
            dry_run: false,
            param_overrides: HashMap::new(),
        };

        let err = run_playbook(&config).await.unwrap_err();
        assert!(err
            .to_string()
            .contains("Remote execution requires Phase 2"));
    }

    #[tokio::test]
    async fn test_PB005_localhost_target_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let out_file = dir.path().join("local.txt");
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            format!(
                r#"
version: "1.0"
name: localhost-test
params: {{}}
targets:
  local:
    host: "localhost"
stages:
  local_stage:
    cmd: "echo local > {}"
    deps: []
    outs:
      - path: {}
    target: local
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
                out_file.display(),
                out_file.display(),
            ),
        )
        .unwrap();

        let config = RunConfig {
            playbook_path: yaml_path,
            stage_filter: None,
            force: false,
            dry_run: false,
            param_overrides: HashMap::new(),
        };

        let result = run_playbook(&config).await.unwrap();
        assert_eq!(result.stages_run, 1);
        assert_eq!(std::fs::read_to_string(&out_file).unwrap().trim(), "local");
    }

    #[tokio::test]
    async fn test_PB005_127_0_0_1_target_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let out_file = dir.path().join("loopback.txt");
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            format!(
                r#"
version: "1.0"
name: loopback-test
params: {{}}
targets:
  loopback:
    host: "127.0.0.1"
stages:
  loop_stage:
    cmd: "echo loopback > {}"
    deps: []
    outs:
      - path: {}
    target: loopback
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
                out_file.display(),
                out_file.display(),
            ),
        )
        .unwrap();

        let config = RunConfig {
            playbook_path: yaml_path,
            stage_filter: None,
            force: false,
            dry_run: false,
            param_overrides: HashMap::new(),
        };

        let result = run_playbook(&config).await.unwrap();
        assert_eq!(result.stages_run, 1);
    }

    #[tokio::test]
    async fn test_PB005_stage_filter() {
        let dir = tempfile::tempdir().unwrap();
        let out1 = dir.path().join("one.txt");
        let out2 = dir.path().join("two.txt");
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            format!(
                r#"
version: "1.0"
name: filter-test
params: {{}}
targets: {{}}
stages:
  one:
    cmd: "echo one > {}"
    deps: []
    outs:
      - path: {}
  two:
    cmd: "echo two > {}"
    deps: []
    outs:
      - path: {}
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
                out1.display(),
                out1.display(),
                out2.display(),
                out2.display(),
            ),
        )
        .unwrap();

        let config = RunConfig {
            playbook_path: yaml_path,
            stage_filter: Some(vec!["one".to_string()]),
            force: false,
            dry_run: false,
            param_overrides: HashMap::new(),
        };

        let result = run_playbook(&config).await.unwrap();
        assert_eq!(result.stages_run, 1);
        assert!(out1.exists());
        assert!(!out2.exists()); // Stage "two" was filtered out
    }

    #[tokio::test]
    async fn test_PB005_force_rerun() {
        let dir = tempfile::tempdir().unwrap();
        let out_file = dir.path().join("force.txt");
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            format!(
                r#"
version: "1.0"
name: force-test
params: {{}}
targets: {{}}
stages:
  write:
    cmd: "echo force > {}"
    deps: []
    outs:
      - path: {}
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
                out_file.display(),
                out_file.display(),
            ),
        )
        .unwrap();

        // First run
        let config = RunConfig {
            playbook_path: yaml_path.clone(),
            stage_filter: None,
            force: false,
            dry_run: false,
            param_overrides: HashMap::new(),
        };
        let r1 = run_playbook(&config).await.unwrap();
        assert_eq!(r1.stages_run, 1);

        // Second run without force — cached
        let r2 = run_playbook(&config).await.unwrap();
        assert_eq!(r2.stages_cached, 1);
        assert_eq!(r2.stages_run, 0);

        // Third run with force — re-runs
        let force_config = RunConfig {
            playbook_path: yaml_path.clone(),
            stage_filter: None,
            force: true,
            dry_run: false,
            param_overrides: HashMap::new(),
        };
        let r3 = run_playbook(&force_config).await.unwrap();
        assert_eq!(r3.stages_run, 1);
        assert_eq!(r3.stages_cached, 0);
    }

    #[tokio::test]
    async fn test_PB005_execute_command_stderr() {
        let result = execute_command("echo error >&2 && exit 42").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        let cmd_err = err.downcast_ref::<CommandError>().unwrap();
        assert_eq!(cmd_err.exit_code, Some(42));
        assert!(cmd_err.stderr.contains("error"));
    }

    #[test]
    fn test_PB005_show_status_all_stage_statuses() {
        // Test that show_status handles all StageStatus variants in lock
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("test.yaml");
        std::fs::write(
            &yaml_path,
            r#"
version: "1.0"
name: multi-status
params: {}
targets: {}
stages:
  completed:
    cmd: "echo a"
    deps: []
    outs: []
  failed:
    cmd: "echo b"
    deps: []
    outs: []
  running:
    cmd: "echo c"
    deps: []
    outs: []
  pending:
    cmd: "echo d"
    deps: []
    outs: []
  hashing:
    cmd: "echo e"
    deps: []
    outs: []
  validating:
    cmd: "echo f"
    deps: []
    outs: []
  not_run:
    cmd: "echo g"
    deps: []
    outs: []
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
        )
        .unwrap();

        // Manually create a lock file with various statuses
        let lock = LockFile {
            schema: "1.0".to_string(),
            playbook: "multi-status".to_string(),
            generated_at: "2026-02-16T14:00:00Z".to_string(),
            generator: "batuta test".to_string(),
            blake3_version: "1.8".to_string(),
            params_hash: None,
            stages: IndexMap::from([
                (
                    "completed".to_string(),
                    StageLock {
                        status: StageStatus::Completed,
                        started_at: None,
                        completed_at: None,
                        duration_seconds: Some(1.5),
                        target: None,
                        deps: vec![],
                        params_hash: None,
                        outs: vec![],
                        cmd_hash: None,
                        cache_key: None,
                    },
                ),
                (
                    "failed".to_string(),
                    StageLock {
                        status: StageStatus::Failed,
                        started_at: None,
                        completed_at: None,
                        duration_seconds: Some(0.3),
                        target: None,
                        deps: vec![],
                        params_hash: None,
                        outs: vec![],
                        cmd_hash: None,
                        cache_key: None,
                    },
                ),
                (
                    "running".to_string(),
                    StageLock {
                        status: StageStatus::Running,
                        started_at: None,
                        completed_at: None,
                        duration_seconds: None,
                        target: None,
                        deps: vec![],
                        params_hash: None,
                        outs: vec![],
                        cmd_hash: None,
                        cache_key: None,
                    },
                ),
                (
                    "pending".to_string(),
                    StageLock {
                        status: StageStatus::Pending,
                        started_at: None,
                        completed_at: None,
                        duration_seconds: None,
                        target: None,
                        deps: vec![],
                        params_hash: None,
                        outs: vec![],
                        cmd_hash: None,
                        cache_key: None,
                    },
                ),
                (
                    "hashing".to_string(),
                    StageLock {
                        status: StageStatus::Hashing,
                        started_at: None,
                        completed_at: None,
                        duration_seconds: None,
                        target: None,
                        deps: vec![],
                        params_hash: None,
                        outs: vec![],
                        cmd_hash: None,
                        cache_key: None,
                    },
                ),
                (
                    "validating".to_string(),
                    StageLock {
                        status: StageStatus::Validating,
                        started_at: None,
                        completed_at: None,
                        duration_seconds: None,
                        target: None,
                        deps: vec![],
                        params_hash: None,
                        outs: vec![],
                        cmd_hash: None,
                        cache_key: None,
                    },
                ),
            ]),
        };

        cache::save_lock_file(&lock, &yaml_path).unwrap();
        let result = show_status(&yaml_path);
        assert!(result.is_ok());
    }
}
