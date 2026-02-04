//! Report formatting utilities.
//!
//! Provides text and JSON formatting for publish status reports.

use anyhow::Result;

use super::types::PublishStatusReport;

// ============================================================================
// PUB-007: Display Formatting
// ============================================================================

/// Format report as text table
#[allow(dead_code)] // Used by examples and re-exported in mod.rs
pub fn format_report_text(report: &PublishStatusReport) -> String {
    use std::fmt::Write;

    let mut out = String::new();

    // Header
    writeln!(
        out,
        "{:<20} {:>10} {:>10} {:>10} {:>12}",
        "Crate", "Local", "crates.io", "Git", "Action"
    )
    .unwrap();
    writeln!(out, "{}", "─".repeat(65)).unwrap();

    // Rows
    for status in &report.crates {
        let local = status.local_version.as_deref().unwrap_or("-");
        let remote = status.crates_io_version.as_deref().unwrap_or("-");
        let git = status.git_status.summary();

        writeln!(
            out,
            "{:<20} {:>10} {:>10} {:>10} {:>2} {:>9}",
            status.name,
            local,
            remote,
            git,
            status.action.symbol(),
            status.action.description()
        )
        .unwrap();
    }

    writeln!(out, "{}", "─".repeat(65)).unwrap();

    // Summary
    writeln!(out).unwrap();
    writeln!(
        out,
        "{} {} crates: {} publish, {} commit, {} up-to-date",
        "\u{1F4CA}", // Chart emoji
        report.total, report.needs_publish, report.needs_commit, report.up_to_date
    )
    .unwrap();
    writeln!(
        out,
        "{} {}ms (cache: {} hits, {} misses)",
        "\u{26A1}", // Lightning emoji
        report.elapsed_ms, report.cache_hits, report.cache_misses
    )
    .unwrap();

    out
}

/// Format report as JSON
pub fn format_report_json(report: &PublishStatusReport) -> Result<String> {
    Ok(serde_json::to_string_pretty(report)?)
}
