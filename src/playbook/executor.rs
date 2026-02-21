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


/// Internal context shared across the execution pipeline
struct ExecutionContext {
    run_id: String,
    playbook: Playbook,
    dag_result: dag::PlaybookDag,
    existing_lock: Option<LockFile>,
    lock: LockFile,
    stages_run: u32,
    stages_cached: u32,
    stages_failed: u32,
    rerun_stages: HashSet<String>,
}

/// Hashed inputs for a single stage, computed before cache check
struct StageHashes {
    resolved_cmd: String,
    cmd_hash: String,
    dep_hashes: Vec<(String, String)>,
    dep_locks: Vec<DepLock>,
    params_hash: String,
    cache_key: String,
}

/// Execute a playbook
pub async fn run_playbook(config: &RunConfig) -> Result<RunResult> {
    let total_start = Instant::now();

    let mut ctx = prepare_execution(config)?;
    let stages_to_run = select_stages(&ctx.dag_result, &config.stage_filter);

    for stage_name in &stages_to_run {
        let stage = ctx
            .playbook
            .stages
            .get(stage_name)
            .ok_or_else(|| anyhow::anyhow!("stage '{}' not found in playbook", stage_name))?;

        check_remote_target(stage_name, stage, &ctx.playbook)?;

        if handle_frozen(stage, config.force, stage_name, &config.playbook_path) {
            ctx.stages_cached += 1;
            continue;
        }

        tracing::debug!(
            "stage '{}': executing via raw sh -c (bashrs purification deferred to Phase 2)",
            stage_name
        );

        let hashes = compute_stage_hashes(stage_name, stage, &ctx.playbook)?;
        let cache_action = evaluate_cache(
            stage_name,
            stage,
            &hashes,
            &ctx.existing_lock,
            &ctx.dag_result,
            &ctx.rerun_stages,
            config,
        );

        match cache_action {
            CacheAction::Cached => {
                ctx.stages_cached += 1;
                continue;
            }
            CacheAction::Execute => {}
        }

        let started_at = eventlog::now_iso8601();
        let stage_start = Instant::now();
        let exec_result = execute_command(&hashes.resolved_cmd).await;
        let duration = stage_start.elapsed();
        let completed_at = eventlog::now_iso8601();

        match exec_result {
            Ok(()) => {
                ctx.stages_run += 1;
                ctx.rerun_stages.insert(stage_name.clone());
                handle_stage_success(
                    stage_name,
                    stage,
                    &hashes,
                    &started_at,
                    &completed_at,
                    duration,
                    &mut ctx.lock,
                    &config.playbook_path,
                )?;
            }
            Err(e) => {
                ctx.stages_failed += 1;
                handle_stage_failure(
                    stage_name,
                    stage,
                    &hashes,
                    &started_at,
                    &completed_at,
                    duration,
                    e,
                    &mut ctx.lock,
                    &ctx.playbook,
                    config,
                    &ctx.run_id,
                )?;
            }
        }
    }

    finalize_run(
        &ctx.playbook,
        &ctx.run_id,
        ctx.stages_run,
        ctx.stages_cached,
        ctx.stages_failed,
        total_start.elapsed(),
        ctx.lock,
        config,
    )
}

/// Parse playbook, build DAG, initialize lock file and run context.
fn prepare_execution(config: &RunConfig) -> Result<ExecutionContext> {
    let mut playbook = parser::parse_playbook_file(&config.playbook_path)?;
    let warnings = parser::validate_playbook(&playbook)?;
    for w in &warnings {
        tracing::warn!("playbook validation: {}", w);
    }

    for (k, v) in &config.param_overrides {
        playbook.params.insert(k.clone(), v.clone());
    }

    let dag_result = dag::build_dag(&playbook)?;
    let run_id = eventlog::generate_run_id();

    let _ = eventlog::append_event(
        &config.playbook_path,
        PipelineEvent::RunStarted {
            playbook: playbook.name.clone(),
            run_id: run_id.clone(),
            batuta_version: env!("CARGO_PKG_VERSION").to_string(),
        },
    );

    let existing_lock = cache::load_lock_file(&config.playbook_path)?;

    let mut lock = LockFile {
        schema: "1.0".to_string(),
        playbook: playbook.name.clone(),
        generated_at: eventlog::now_iso8601(),
        generator: format!("batuta {}", env!("CARGO_PKG_VERSION")),
        blake3_version: "1.8".to_string(),
        params_hash: None,
        stages: IndexMap::new(),
    };

    if let Some(ref el) = existing_lock {
        for (name, stage_lock) in &el.stages {
            lock.stages.insert(name.clone(), stage_lock.clone());
        }
    }

    Ok(ExecutionContext {
        run_id,
        playbook,
        dag_result,
        existing_lock,
        lock,
        stages_run: 0,
        stages_cached: 0,
        stages_failed: 0,
        rerun_stages: HashSet::new(),
    })
}

/// Filter and order stages according to DAG topology and optional stage filter.
fn select_stages(
    dag_result: &dag::PlaybookDag,
    stage_filter: &Option<Vec<String>>,
) -> Vec<String> {
    if let Some(ref filter) = stage_filter {
        dag_result
            .topo_order
            .iter()
            .filter(|s| filter.contains(s))
            .cloned()
            .collect()
    } else {
        dag_result.topo_order.clone()
    }
}

/// Reject stages targeting remote hosts (Phase 2 not yet available).
/// Allow localhost and 127.0.0.1 (M8 fix).
fn check_remote_target(stage_name: &str, stage: &Stage, playbook: &Playbook) -> Result<()> {
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
    Ok(())
}

/// Handle frozen stages: return true if stage should be skipped (cached).
fn handle_frozen(
    stage: &Stage,
    force: bool,
    stage_name: &str,
    playbook_path: &Path,
) -> bool {
    if stage.frozen && !force {
        let _ = eventlog::append_event(
            playbook_path,
            PipelineEvent::StageCached {
                stage: stage_name.to_string(),
                cache_key: "frozen".to_string(),
                reason: "stage is frozen".to_string(),
            },
        );
        println!("  {} CACHED (frozen)", stage_name);
        true
    } else {
        false
    }
}

/// Resolve templates and compute all hashes for a stage.
fn compute_stage_hashes(
    stage_name: &str,
    stage: &Stage,
    playbook: &Playbook,
) -> Result<StageHashes> {
    let resolved_cmd = template::resolve_template(
        &stage.cmd,
        &playbook.params,
        &stage.params,
        &stage.deps,
        &stage.outs,
    )
    .with_context(|| format!("stage '{}' template resolution failed", stage_name))?;

    let cmd_hash = hasher::hash_cmd(&resolved_cmd);

    let (dep_hashes, dep_locks) = hash_dependencies(&stage.deps)?;

    let deps_combined = hasher::combine_deps_hashes(
        &dep_hashes
            .iter()
            .map(|(_, h)| h.clone())
            .collect::<Vec<_>>(),
    );

    let param_refs = hasher::effective_param_keys(&stage.params, &stage.cmd);
    let params_hash = hasher::hash_params(&playbook.params, &param_refs)?;
    let cache_key = hasher::compute_cache_key(&cmd_hash, &deps_combined, &params_hash);

    Ok(StageHashes {
        resolved_cmd,
        cmd_hash,
        dep_hashes,
        dep_locks,
        params_hash,
        cache_key,
    })
}

/// Hash all dependencies for a stage, returning parallel vecs of hashes and locks.
fn hash_dependencies(deps: &[Dependency]) -> Result<(Vec<(String, String)>, Vec<DepLock>)> {
    let mut dep_hashes: Vec<(String, String)> = Vec::new();
    let mut dep_locks: Vec<DepLock> = Vec::new();

    for dep in deps {
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
            dep_hashes.push((dep.path.clone(), String::new()));
            dep_locks.push(DepLock {
                path: dep.path.clone(),
                hash: String::new(),
                file_count: None,
                total_bytes: None,
            });
        }
    }

    Ok((dep_hashes, dep_locks))
}

/// Whether to skip (cached) or execute the stage.
enum CacheAction {
    Cached,
    Execute,
}

/// Check cache and log the appropriate event, returning the action to take.
fn evaluate_cache(
    stage_name: &str,
    stage: &Stage,
    hashes: &StageHashes,
    existing_lock: &Option<LockFile>,
    dag_result: &dag::PlaybookDag,
    rerun_stages: &HashSet<String>,
    config: &RunConfig,
) -> CacheAction {
    let upstream_reruns: Vec<String> = dag_result
        .predecessors
        .get(stage_name)
        .map(|preds: &Vec<String>| {
            preds
                .iter()
                .filter(|p| rerun_stages.contains(p.as_str()))
                .cloned()
                .collect()
        })
        .unwrap_or_default();

    let decision = cache::check_cache(
        stage_name,
        &hashes.cache_key,
        &hashes.cmd_hash,
        &hashes.dep_hashes,
        &hashes.params_hash,
        existing_lock,
        config.force,
        &upstream_reruns,
    );

    match decision {
        CacheDecision::Hit => {
            let _ = eventlog::append_event(
                &config.playbook_path,
                PipelineEvent::StageCached {
                    stage: stage_name.to_string(),
                    cache_key: hashes.cache_key.clone(),
                    reason: "cache_key matches lock".to_string(),
                },
            );
            println!("  {} CACHED", stage_name);
            CacheAction::Cached
        }
        CacheDecision::Miss { ref reasons } => {
            let reason_str: Vec<String> = reasons.iter().map(|r| r.to_string()).collect();
            let _ = eventlog::append_event(
                &config.playbook_path,
                PipelineEvent::StageStarted {
                    stage: stage_name.to_string(),
                    target: stage.target.clone().unwrap_or_else(|| "local".to_string()),
                    cache_miss_reason: reason_str.join("; "),
                },
            );
            println!("  {} RUNNING ({})", stage_name, reason_str.join("; "));
            CacheAction::Execute
        }
    }
}

/// Process successful stage execution: hash outputs, update lock, log event.
#[allow(clippy::too_many_arguments)]
fn handle_stage_success(
    stage_name: &str,
    stage: &Stage,
    hashes: &StageHashes,
    started_at: &str,
    completed_at: &str,
    duration: std::time::Duration,
    lock: &mut LockFile,
    playbook_path: &Path,
) -> Result<()> {
    let out_locks = hash_outputs(stage_name, &stage.outs)?;

    let outs_hash = if out_locks.is_empty() {
        None
    } else {
        Some(hasher::combine_deps_hashes(
            &out_locks.iter().map(|o| o.hash.clone()).collect::<Vec<_>>(),
        ))
    };

    lock.stages.insert(
        stage_name.to_string(),
        StageLock {
            status: StageStatus::Completed,
            started_at: Some(started_at.to_string()),
            completed_at: Some(completed_at.to_string()),
            duration_seconds: Some(duration.as_secs_f64()),
            target: stage.target.clone(),
            deps: hashes.dep_locks.clone(),
            params_hash: Some(hashes.params_hash.clone()),
            outs: out_locks,
            cmd_hash: Some(hashes.cmd_hash.clone()),
            cache_key: Some(hashes.cache_key.clone()),
        },
    );

    let _ = eventlog::append_event(
        playbook_path,
        PipelineEvent::StageCompleted {
            stage: stage_name.to_string(),
            duration_seconds: duration.as_secs_f64(),
            outs_hash,
        },
    );

    println!(
        "  {} COMPLETED ({:.1}s)",
        stage_name,
        duration.as_secs_f64()
    );

    Ok(())
}

/// Hash all output artifacts for a completed stage.
fn hash_outputs(stage_name: &str, outs: &[Output]) -> Result<Vec<OutLock>> {
    let mut out_locks: Vec<OutLock> = Vec::new();
    for out in outs {
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
    Ok(out_locks)
}

/// Process failed stage execution: update lock, log event, enforce Jidoka policy.
#[allow(clippy::too_many_arguments)]
fn handle_stage_failure(
    stage_name: &str,
    stage: &Stage,
    hashes: &StageHashes,
    started_at: &str,
    completed_at: &str,
    duration: std::time::Duration,
    error: anyhow::Error,
    lock: &mut LockFile,
    playbook: &Playbook,
    config: &RunConfig,
    run_id: &str,
) -> Result<()> {
    let (exit_code, error_msg) = match error.downcast_ref::<CommandError>() {
        Some(ce) => (ce.exit_code, ce.stderr.clone()),
        None => (None, error.to_string()),
    };

    lock.stages.insert(
        stage_name.to_string(),
        StageLock {
            status: StageStatus::Failed,
            started_at: Some(started_at.to_string()),
            completed_at: Some(completed_at.to_string()),
            duration_seconds: Some(duration.as_secs_f64()),
            target: stage.target.clone(),
            deps: hashes.dep_locks.clone(),
            params_hash: Some(hashes.params_hash.clone()),
            outs: vec![],
            cmd_hash: Some(hashes.cmd_hash.clone()),
            cache_key: None,
        },
    );

    let _ = eventlog::append_event(
        &config.playbook_path,
        PipelineEvent::StageFailed {
            stage: stage_name.to_string(),
            exit_code,
            error: error_msg.clone(),
            retry_attempt: None,
        },
    );

    eprintln!("  {} FAILED: {}", stage_name, error_msg);

    if playbook.policy.failure == FailurePolicy::StopOnFirst {
        let _ = eventlog::append_event(
            &config.playbook_path,
            PipelineEvent::RunFailed {
                playbook: playbook.name.clone(),
                run_id: run_id.to_string(),
                error: format!("stage '{}' failed (Jidoka: stop_on_first)", stage_name),
            },
        );

        if playbook.policy.lock_file {
            lock.generated_at = eventlog::now_iso8601();
            cache::save_lock_file(lock, &config.playbook_path)?;
        }

        bail!(
            "stage '{}' failed: {} (Jidoka: stop_on_first policy)",
            stage_name,
            error_msg
        );
    }

    Ok(())
}

/// Save final lock file and return the run result.
fn finalize_run(
    playbook: &Playbook,
    run_id: &str,
    stages_run: u32,
    stages_cached: u32,
    stages_failed: u32,
    total_duration: std::time::Duration,
    mut lock: LockFile,
    config: &RunConfig,
) -> Result<RunResult> {
    let _ = eventlog::append_event(
        &config.playbook_path,
        PipelineEvent::RunCompleted {
            playbook: playbook.name.clone(),
            run_id: run_id.to_string(),
            stages_run,
            stages_cached,
            stages_failed,
            total_seconds: total_duration.as_secs_f64(),
        },
    );

    if playbook.policy.lock_file {
        lock.generated_at = eventlog::now_iso8601();
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
