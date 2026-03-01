//! Stack drift detection and enforcement for the batuta CLI.
//!
//! Checks for version drift across PAIML stack crates and either
//! warns (local dev) or blocks (CI/strict mode) when drift is found.

use tracing::warn;

use crate::ansi_colors::Colorize;
use crate::stack;

/// Check if the environment requests strict mode (BATUTA_STRICT=1).
fn is_strict_env() -> bool {
    std::env::var("BATUTA_STRICT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Format drift as a warning (non-blocking).
fn format_drift_warning(drifts: &[stack::DriftReport]) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "{}\n\n",
        "⚠️  Stack Drift Warning (non-blocking)"
            .bright_yellow()
            .bold()
    ));

    for drift in drifts {
        output.push_str(&format!("   {}\n", drift.display()));
    }

    output.push_str(&format!(
        "\n{}",
        "Run 'batuta stack drift --fix' to update dependencies.\n".dimmed()
    ));
    output.push_str(&format!(
        "{}",
        "Use --strict to enforce drift checking.\n".dimmed()
    ));

    output
}

/// Check for stack drift across PAIML crates.
///
/// Returns `None` if check cannot be performed (offline, etc.)
/// Returns `Some(empty)` if no drift detected
/// Returns `Some(drifts)` if drift detected
fn check_stack_drift() -> anyhow::Result<Option<Vec<stack::DriftReport>>> {
    // Create runtime for async operations
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return Ok(None), // Can't create runtime, skip check
    };

    rt.block_on(async {
        let mut client = stack::CratesIoClient::new().with_persistent_cache();
        let mut checker = stack::DriftChecker::new();

        match checker.detect_drift(&mut client).await {
            Ok(drifts) => Ok(Some(drifts)),
            Err(_) => Ok(None), // Network error or similar, skip check
        }
    })
}

/// Get the drift warning marker file path (workspace-scoped).
/// Uses workspace root hash to scope warnings per project.
fn drift_marker_path() -> std::path::PathBuf {
    // Hash the workspace root to scope warnings per project
    let workspace_id = std::env::current_dir()
        .ok()
        .and_then(|p| {
            p.to_str().map(|s| {
                // Simple hash: sum of bytes mod 100000
                s.bytes().map(|b| b as u64).sum::<u64>() % 100000
            })
        })
        .unwrap_or(0);
    std::env::temp_dir().join(format!("batuta-drift-shown-{}", workspace_id))
}

/// Check if drift warning was already shown this session.
fn drift_already_shown() -> bool {
    let marker = drift_marker_path();
    if marker.exists() {
        // Check if marker is less than 1 hour old
        if let Ok(meta) = std::fs::metadata(&marker) {
            if let Ok(modified) = meta.modified() {
                if let Ok(elapsed) = modified.elapsed() {
                    return elapsed.as_secs() < 3600; // 1 hour
                }
            }
        }
    }
    false
}

/// Mark drift warning as shown for this session.
fn mark_drift_shown() {
    let _ = std::fs::write(drift_marker_path(), "shown");
}

/// Enforce stack drift checking with smart tolerance.
///
/// Default: shows warning ONCE per hour, never blocks.
/// With --strict or BATUTA_STRICT=1: always blocks on drift.
pub(crate) fn enforce_drift_check(strict: bool) -> anyhow::Result<()> {
    let strict_mode = strict || is_strict_env();

    // Only check once per hour unless strict mode (Muda elimination)
    if !strict_mode && drift_already_shown() {
        return Ok(());
    }

    let Some(drifts) = check_stack_drift()? else {
        return Ok(());
    };
    if drifts.is_empty() {
        return Ok(());
    }

    if strict_mode {
        eprintln!("{}", stack::format_drift_errors(&drifts));
        std::process::exit(1);
    } else {
        // Default: warn once, never block (Muda: don't waste user's time)
        // Users who want enforcement opt in with --strict or BATUTA_STRICT=1
        if !drift_already_shown() {
            warn!("Stack drift detected (non-blocking)");
            eprintln!("{}", format_drift_warning(&drifts));
            mark_drift_shown();
        }
    }

    Ok(())
}
