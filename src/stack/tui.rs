#![allow(dead_code)]
//! TUI Dashboard for PAIML Stack Status
//!
//! Provides an interactive terminal UI for visualizing stack health.
//!
//! ## Architecture (PROBAR-SPEC-009)
//!
//! Migrated from ratatui to presentar-terminal for stack consistency.
//! Uses direct CellBuffer rendering with crossterm for terminal handling.

use crate::stack::types::{CrateInfo, CrateStatus, StackHealthReport};
use anyhow::Result;
#[cfg(feature = "native")]
use std::io::{self, Write};
#[cfg(feature = "native")]
use std::time::Duration;

#[cfg(feature = "native")]
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

#[cfg(feature = "native")]
use presentar_terminal::{CellBuffer, Color, DiffRenderer, Modifiers};

/// CYAN color constant (not in presentar-terminal)
#[cfg(feature = "native")]
const CYAN: Color = Color {
    r: 0.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};

/// TUI Dashboard state
#[cfg(feature = "native")]
pub struct Dashboard {
    /// Health report to display
    report: StackHealthReport,
    /// Selected crate index
    selected: usize,
    /// Whether to show details panel
    show_details: bool,
    /// Cell buffer for rendering
    buffer: CellBuffer,
    /// Diff renderer for efficient updates
    renderer: DiffRenderer,
    /// Terminal width
    width: u16,
    /// Terminal height
    height: u16,
}

#[cfg(feature = "native")]
impl Dashboard {
    /// Create a new dashboard with a health report
    pub fn new(report: StackHealthReport) -> Self {
        let (width, height) = crossterm::terminal::size().unwrap_or((80, 24));
        Self {
            report,
            selected: 0,
            show_details: true,
            buffer: CellBuffer::new(width, height),
            renderer: DiffRenderer::new(),
            width,
            height,
        }
    }

    /// Run the TUI dashboard
    pub fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, cursor::Hide)?;

        // Run the main loop
        let result = self.run_loop(&mut stdout);

        // Restore terminal
        disable_raw_mode()?;
        execute!(stdout, LeaveAlternateScreen, cursor::Show)?;

        result
    }

    /// Main event loop
    fn run_loop(&mut self, stdout: &mut io::Stdout) -> Result<()> {
        loop {
            // Update terminal size
            let (w, h) = crossterm::terminal::size().unwrap_or((80, 24));
            if w != self.width || h != self.height {
                self.width = w;
                self.height = h;
                self.buffer.resize(w, h);
                self.renderer.reset();
            }

            // Clear and render
            self.buffer.clear();
            self.render();

            // Flush to terminal
            self.renderer.flush(&mut self.buffer, stdout)?;
            stdout.flush()?;

            // Poll for events with timeout
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                            KeyCode::Up | KeyCode::Char('k') => {
                                if self.selected > 0 {
                                    self.selected -= 1;
                                }
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                if self.selected < self.report.crates.len().saturating_sub(1) {
                                    self.selected += 1;
                                }
                            }
                            KeyCode::Char('d') => {
                                self.show_details = !self.show_details;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    /// Write a string with color
    fn write_str(&mut self, x: u16, y: u16, s: &str, fg: Color) {
        let mut cx = x;
        for ch in s.chars() {
            if cx >= self.width {
                break;
            }
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.buffer
                .update(cx, y, s, fg, Color::TRANSPARENT, Modifiers::NONE);
            cx = cx.saturating_add(1);
        }
    }

    /// Set a single character with color
    fn set_char(&mut self, x: u16, y: u16, ch: char, fg: Color) {
        if x < self.width && y < self.height {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.buffer
                .update(x, y, s, fg, Color::TRANSPARENT, Modifiers::NONE);
        }
    }

    /// Render the dashboard
    fn render(&mut self) {
        let w = self.width;
        let h = self.height;

        // Layout: Title(3) | Table(min 10) | Details(8 if shown) | Help(3)
        let title_h: u16 = 3;
        let help_h: u16 = 3;
        let details_h: u16 = if self.show_details { 8 } else { 0 };
        let table_h = h.saturating_sub(title_h + details_h + help_h);

        self.render_title(0, 0, w, title_h);
        self.render_table(0, title_h, w, table_h);

        if self.show_details {
            self.render_details(0, title_h + table_h, w, details_h);
            self.render_help(0, h.saturating_sub(help_h), w, help_h);
        } else {
            self.render_help(0, h.saturating_sub(help_h), w, help_h);
        }
    }

    /// Render the title bar
    fn render_title(&mut self, x: u16, y: u16, w: u16, h: u16) {
        self.draw_box(x, y, w, h, "");

        let summary = &self.report.summary;
        let status_text = if summary.error_count > 0 {
            format!("X {} errors", summary.error_count)
        } else if summary.warning_count > 0 {
            format!("! {} warnings", summary.warning_count)
        } else {
            "* All healthy".to_string()
        };

        let title = format!(
            "PAIML Stack Dashboard | {} crates | {}",
            summary.total_crates, status_text
        );

        let max_len = (w.saturating_sub(4)) as usize;
        self.write_str(x + 2, y + 1, &title[..title.len().min(max_len)], CYAN);
    }

    /// Render the crate table
    fn render_table(&mut self, x: u16, y: u16, w: u16, h: u16) {
        self.draw_box(x, y, w, h, " Crates ");

        // Header
        let header = format!(
            "{:4} {:15} {:12} {:12} {:8}",
            "Stat", "Crate", "Local", "Crates.io", "Issues"
        );
        let max_len = (w.saturating_sub(2)) as usize;
        self.write_str(
            x + 1,
            y + 1,
            &header[..header.len().min(max_len)],
            Color::YELLOW,
        );

        // Separator
        let sep = "─".repeat(max_len);
        self.write_str(x + 1, y + 2, &sep[..sep.len().min(max_len)], Color::WHITE);

        // Rows - collect data first to avoid borrow conflict
        let content_y = y + 3;
        let content_h = h.saturating_sub(4) as usize;

        let rows: Vec<_> = self
            .report
            .crates
            .iter()
            .take(content_h)
            .enumerate()
            .map(|(i, crate_info)| {
                let status_icon = match crate_info.status {
                    CrateStatus::Healthy => "*",
                    CrateStatus::Warning => "!",
                    CrateStatus::Error => "X",
                    CrateStatus::Unknown => "?",
                };

                let crates_io = crate_info
                    .crates_io_version
                    .as_ref()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "—".to_string());

                let issue_count = crate_info.issues.len();
                let issue_str = if issue_count > 0 {
                    issue_count.to_string()
                } else {
                    "—".to_string()
                };

                let is_selected = i == self.selected;
                let fg_color = if is_selected {
                    Color::YELLOW
                } else {
                    Color::WHITE
                };

                let row = format!(
                    "{:4} {:15} {:12} {:12} {:8}",
                    status_icon,
                    &crate_info.name[..crate_info.name.len().min(15)],
                    crate_info.local_version.to_string(),
                    crates_io,
                    issue_str
                );

                let marker = if is_selected { "> " } else { "  " };
                let full_row = format!("{}{}", marker, row);
                (i, full_row, fg_color)
            })
            .collect();

        for (i, full_row, fg_color) in rows {
            self.write_str(
                x + 1,
                content_y + i as u16,
                &full_row[..full_row.len().min(max_len)],
                fg_color,
            );
        }
    }

    /// Render the details panel
    fn render_details(&mut self, x: u16, y: u16, w: u16, h: u16) {
        self.draw_box(x, y, w, h, " Details ");
        let max_len = (w.saturating_sub(2)) as usize;
        let content_y = y + 1;

        // Extract data first to avoid borrow conflict
        let crate_data = self.report.crates.get(self.selected).map(|c| {
            let name_line = format!("Name: {}", c.name);
            let version_line = format!("Version: {}", c.local_version);
            let issue_lines: Vec<String> = c
                .issues
                .iter()
                .take(h.saturating_sub(5) as usize)
                .map(|issue| format!("  * {}", issue.message))
                .collect();
            (name_line, version_line, issue_lines)
        });

        if let Some((name_line, version_line, issue_lines)) = crate_data {
            // Name
            self.write_str(
                x + 1,
                content_y,
                &name_line[..name_line.len().min(max_len)],
                Color::WHITE,
            );

            // Version
            self.write_str(
                x + 1,
                content_y + 1,
                &version_line[..version_line.len().min(max_len)],
                Color::WHITE,
            );

            // Issues
            if !issue_lines.is_empty() {
                self.write_str(x + 1, content_y + 2, "Issues:", Color::RED);
                for (i, issue_line) in issue_lines.iter().enumerate() {
                    self.write_str(
                        x + 1,
                        content_y + 3 + i as u16,
                        &issue_line[..issue_line.len().min(max_len)],
                        Color::WHITE,
                    );
                }
            } else {
                self.write_str(x + 1, content_y + 2, "No issues", Color::GREEN);
            }
        } else {
            self.write_str(x + 1, y + 1, "No crate selected", Color::WHITE);
        }
    }

    /// Render help bar
    fn render_help(&mut self, x: u16, y: u16, w: u16, h: u16) {
        self.draw_box(x, y, w, h, "");

        let help = "^/k Up  v/j Down  d Toggle details  q/Esc Quit";
        let max_len = (w.saturating_sub(2)) as usize;
        self.write_str(x + 1, y + 1, &help[..help.len().min(max_len)], CYAN);
    }

    /// Draw a box with border and title
    fn draw_box(&mut self, x: u16, y: u16, w: u16, h: u16, title: &str) {
        if w < 2 || h < 2 {
            return;
        }

        // Top border
        self.set_char(x, y, '┌', Color::WHITE);
        for i in 1..w - 1 {
            self.set_char(x + i, y, '─', Color::WHITE);
        }
        self.set_char(x + w - 1, y, '┐', Color::WHITE);

        // Title
        if !title.is_empty() && w > title.len() as u16 + 2 {
            let title_x = x + 2;
            self.write_str(title_x, y, title, CYAN);
        }

        // Sides
        for i in 1..h - 1 {
            self.set_char(x, y + i, '│', Color::WHITE);
            self.set_char(x + w - 1, y + i, '│', Color::WHITE);
        }

        // Bottom border
        self.set_char(x, y + h - 1, '└', Color::WHITE);
        for i in 1..w - 1 {
            self.set_char(x + i, y + h - 1, '─', Color::WHITE);
        }
        self.set_char(x + w - 1, y + h - 1, '┘', Color::WHITE);
    }
}

/// Run the TUI dashboard with the given report
#[cfg(feature = "native")]
pub fn run_dashboard(report: StackHealthReport) -> Result<()> {
    let mut dashboard = Dashboard::new(report);
    dashboard.run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stack::types::{CrateIssue, HealthSummary, IssueSeverity, IssueType};
    use std::path::PathBuf;

    fn create_test_report() -> StackHealthReport {
        let mut crates = vec![
            CrateInfo::new("trueno", semver::Version::new(1, 2, 0), PathBuf::new()),
            CrateInfo::new("aprender", semver::Version::new(0, 14, 1), PathBuf::new()),
        ];

        crates[0].status = CrateStatus::Healthy;
        crates[0].crates_io_version = Some(semver::Version::new(1, 2, 0));

        crates[1].status = CrateStatus::Warning;
        crates[1].crates_io_version = Some(semver::Version::new(0, 14, 1));
        crates[1].issues.push(CrateIssue::new(
            IssueSeverity::Warning,
            IssueType::VersionBehind,
            "Test warning",
        ));

        StackHealthReport {
            crates,
            conflicts: vec![],
            summary: HealthSummary {
                total_crates: 2,
                healthy_count: 1,
                warning_count: 1,
                error_count: 0,
                path_dependency_count: 0,
                conflict_count: 0,
            },
            timestamp: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_dashboard_creation() {
        let report = create_test_report();
        let dashboard = Dashboard::new(report);

        assert_eq!(dashboard.selected, 0);
        assert!(dashboard.show_details);
        assert_eq!(dashboard.report.crates.len(), 2);
    }

    #[test]
    fn test_dashboard_navigation() {
        let report = create_test_report();
        let mut dashboard = Dashboard::new(report);

        assert_eq!(dashboard.selected, 0);
        dashboard.selected = 1;
        assert_eq!(dashboard.selected, 1);
        dashboard.selected = 0;
        assert_eq!(dashboard.selected, 0);
    }

    #[test]
    fn test_dashboard_toggle_details() {
        let report = create_test_report();
        let mut dashboard = Dashboard::new(report);

        assert!(dashboard.show_details);
        dashboard.show_details = !dashboard.show_details;
        assert!(!dashboard.show_details);
    }

    #[test]
    fn test_dashboard_with_empty_report() {
        let report = StackHealthReport {
            crates: vec![],
            conflicts: vec![],
            summary: HealthSummary {
                total_crates: 0,
                healthy_count: 0,
                warning_count: 0,
                error_count: 0,
                path_dependency_count: 0,
                conflict_count: 0,
            },
            timestamp: chrono::Utc::now(),
        };

        let dashboard = Dashboard::new(report);
        assert_eq!(dashboard.report.crates.len(), 0);
    }

    #[test]
    fn test_dashboard_with_errors() {
        let mut crates = vec![CrateInfo::new(
            "broken",
            semver::Version::new(0, 1, 0),
            PathBuf::new(),
        )];
        crates[0].status = CrateStatus::Error;
        crates[0].issues.push(CrateIssue::new(
            IssueSeverity::Error,
            IssueType::PathDependency,
            "Path dependency error",
        ));

        let report = StackHealthReport {
            crates,
            conflicts: vec![],
            summary: HealthSummary {
                total_crates: 1,
                healthy_count: 0,
                warning_count: 0,
                error_count: 1,
                path_dependency_count: 1,
                conflict_count: 0,
            },
            timestamp: chrono::Utc::now(),
        };

        let dashboard = Dashboard::new(report);
        assert_eq!(dashboard.report.summary.error_count, 1);
    }

    #[test]
    fn test_dashboard_selected_boundary() {
        let report = create_test_report();
        let mut dashboard = Dashboard::new(report);

        dashboard.selected = 1;
        assert_eq!(dashboard.selected, 1);
        dashboard.selected = 100;
        assert_eq!(dashboard.selected, 100);
    }

    #[test]
    fn test_dashboard_report_summary_types() {
        let report = StackHealthReport {
            crates: vec![],
            conflicts: vec![],
            summary: HealthSummary {
                total_crates: 0,
                healthy_count: 0,
                warning_count: 0,
                error_count: 0,
                path_dependency_count: 0,
                conflict_count: 0,
            },
            timestamp: chrono::Utc::now(),
        };

        let dashboard = Dashboard::new(report);
        assert_eq!(dashboard.report.summary.total_crates, 0);
        assert!(dashboard.show_details);
    }

    #[test]
    fn test_dashboard_with_unknown_status() {
        let mut crates = vec![CrateInfo::new(
            "unknown",
            semver::Version::new(0, 1, 0),
            PathBuf::new(),
        )];
        crates[0].status = CrateStatus::Unknown;

        let report = StackHealthReport {
            crates,
            conflicts: vec![],
            summary: HealthSummary::default(),
            timestamp: chrono::Utc::now(),
        };

        let dashboard = Dashboard::new(report);
        assert_eq!(dashboard.report.crates[0].status, CrateStatus::Unknown);
    }

    #[test]
    fn test_dashboard_multiple_crates() {
        let mut crates = vec![
            CrateInfo::new("a", semver::Version::new(1, 0, 0), PathBuf::new()),
            CrateInfo::new("b", semver::Version::new(2, 0, 0), PathBuf::new()),
            CrateInfo::new("c", semver::Version::new(3, 0, 0), PathBuf::new()),
        ];
        for c in &mut crates {
            c.status = CrateStatus::Healthy;
        }

        let report = StackHealthReport {
            crates,
            conflicts: vec![],
            summary: HealthSummary {
                total_crates: 3,
                healthy_count: 3,
                ..Default::default()
            },
            timestamp: chrono::Utc::now(),
        };

        let mut dashboard = Dashboard::new(report);

        assert_eq!(dashboard.selected, 0);
        dashboard.selected = 1;
        assert_eq!(dashboard.selected, 1);
        dashboard.selected = 2;
        assert_eq!(dashboard.selected, 2);
    }

    #[test]
    fn test_dashboard_crate_with_multiple_issues() {
        let mut crates = vec![CrateInfo::new(
            "problematic",
            semver::Version::new(0, 1, 0),
            PathBuf::new(),
        )];
        crates[0].status = CrateStatus::Warning;
        crates[0].issues.push(CrateIssue::new(
            IssueSeverity::Warning,
            IssueType::VersionBehind,
            "Version behind",
        ));
        crates[0].issues.push(CrateIssue::new(
            IssueSeverity::Warning,
            IssueType::VersionBehind,
            "Another warning",
        ));

        let report = StackHealthReport {
            crates,
            conflicts: vec![],
            summary: HealthSummary {
                total_crates: 1,
                warning_count: 1,
                ..Default::default()
            },
            timestamp: chrono::Utc::now(),
        };

        let dashboard = Dashboard::new(report);
        assert_eq!(dashboard.report.crates[0].issues.len(), 2);
    }

    mod render_tests {
        use super::*;

        #[test]
        fn test_render_full_with_details() {
            let report = create_test_report();
            let mut dashboard = Dashboard::new(report);
            dashboard.render();
            assert!(dashboard.width > 0);
            assert!(dashboard.height > 0);
        }

        #[test]
        fn test_render_full_without_details() {
            let report = create_test_report();
            let mut dashboard = Dashboard::new(report);
            dashboard.show_details = false;
            dashboard.render();
            assert!(dashboard.width > 0);
        }

        #[test]
        fn test_render_with_all_status_types() {
            let mut crates = vec![
                CrateInfo::new("healthy", semver::Version::new(1, 0, 0), PathBuf::new()),
                CrateInfo::new("warning", semver::Version::new(1, 0, 0), PathBuf::new()),
                CrateInfo::new("error", semver::Version::new(1, 0, 0), PathBuf::new()),
                CrateInfo::new("unknown", semver::Version::new(1, 0, 0), PathBuf::new()),
            ];
            crates[0].status = CrateStatus::Healthy;
            crates[1].status = CrateStatus::Warning;
            crates[2].status = CrateStatus::Error;
            crates[3].status = CrateStatus::Unknown;

            let report = StackHealthReport {
                crates,
                conflicts: vec![],
                summary: HealthSummary {
                    total_crates: 4,
                    healthy_count: 1,
                    warning_count: 1,
                    error_count: 1,
                    ..Default::default()
                },
                timestamp: chrono::Utc::now(),
            };

            let mut dashboard = Dashboard::new(report);
            dashboard.render();
            assert!(dashboard.width > 0);
        }

        #[test]
        fn test_render_table_with_selected() {
            let mut crates = vec![
                CrateInfo::new("first", semver::Version::new(1, 0, 0), PathBuf::new()),
                CrateInfo::new("second", semver::Version::new(2, 0, 0), PathBuf::new()),
            ];
            crates[0].status = CrateStatus::Healthy;
            crates[1].status = CrateStatus::Healthy;

            let report = StackHealthReport {
                crates,
                conflicts: vec![],
                summary: HealthSummary::default(),
                timestamp: chrono::Utc::now(),
            };

            let mut dashboard = Dashboard::new(report);
            dashboard.selected = 1;
            dashboard.render();
            assert!(dashboard.width > 0);
        }

        #[test]
        fn test_render_empty_report() {
            let report = StackHealthReport {
                crates: vec![],
                conflicts: vec![],
                summary: HealthSummary::default(),
                timestamp: chrono::Utc::now(),
            };

            let mut dashboard = Dashboard::new(report);
            dashboard.render();
            assert!(dashboard.width > 0);
        }

        #[test]
        fn test_render_details_with_issues() {
            let mut crates = vec![CrateInfo::new(
                "broken",
                semver::Version::new(0, 1, 0),
                PathBuf::new(),
            )];
            crates[0].issues.push(CrateIssue::new(
                IssueSeverity::Error,
                IssueType::PathDependency,
                "Test error message",
            ));

            let report = StackHealthReport {
                crates,
                conflicts: vec![],
                summary: HealthSummary::default(),
                timestamp: chrono::Utc::now(),
            };

            let mut dashboard = Dashboard::new(report);
            dashboard.render();
            assert!(dashboard.height > 0);
        }
    }
}
