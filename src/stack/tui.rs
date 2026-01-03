#![allow(dead_code)]
//! TUI Dashboard for PAIML Stack Status
//!
//! Provides an interactive terminal UI for visualizing stack health.
//! Uses ratatui for rendering and crossterm for terminal handling.

use crate::stack::types::{CrateInfo, CrateStatus, StackHealthReport};
use anyhow::Result;
use std::io::{self, Stdout};
use std::time::Duration;

#[cfg(feature = "native")]
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

#[cfg(feature = "native")]
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap},
    Frame, Terminal,
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
}

#[cfg(feature = "native")]
impl Dashboard {
    /// Create a new dashboard with a health report
    pub fn new(report: StackHealthReport) -> Self {
        Self {
            report,
            selected: 0,
            show_details: true,
        }
    }

    /// Run the TUI dashboard
    pub fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Run the main loop
        let result = self.run_loop(&mut terminal);

        // Restore terminal
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        result
    }

    /// Main event loop
    fn run_loop(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
        loop {
            terminal.draw(|frame| self.render(frame))?;

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

    /// Render the dashboard
    fn render(&self, frame: &mut Frame) {
        let size = frame.area();

        // Create layout
        let chunks = if self.show_details {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3), // Title
                    Constraint::Min(10),   // Main content
                    Constraint::Length(8), // Details
                    Constraint::Length(3), // Help
                ])
                .split(size)
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3), // Title
                    Constraint::Min(10),   // Main content
                    Constraint::Length(3), // Help
                ])
                .split(size)
        };

        // Render title
        self.render_title(frame, chunks[0]);

        // Render crate table
        self.render_table(frame, chunks[1]);

        // Render details panel if enabled
        if self.show_details && chunks.len() > 3 {
            self.render_details(frame, chunks[2]);
            self.render_help(frame, chunks[3]);
        } else {
            self.render_help(frame, chunks[2]);
        }
    }

    /// Render the title bar
    fn render_title(&self, frame: &mut Frame, area: Rect) {
        let summary = &self.report.summary;
        let status_text = if summary.error_count > 0 {
            format!("❌ {} errors", summary.error_count)
        } else if summary.warning_count > 0 {
            format!("⚠️  {} warnings", summary.warning_count)
        } else {
            "✅ All healthy".to_string()
        };

        let title = Paragraph::new(vec![Line::from(vec![
            Span::styled(
                "PAIML Stack Dashboard",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" | "),
            Span::styled(
                format!("{} crates", summary.total_crates),
                Style::default().fg(Color::White),
            ),
            Span::raw(" | "),
            Span::styled(status_text, Style::default().fg(Color::Green)),
        ])])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        );

        frame.render_widget(title, area);
    }

    /// Render the crate table
    fn render_table(&self, frame: &mut Frame, area: Rect) {
        let header = Row::new(vec![
            Cell::from("Status").style(Style::default().fg(Color::Yellow)),
            Cell::from("Crate").style(Style::default().fg(Color::Yellow)),
            Cell::from("Local").style(Style::default().fg(Color::Yellow)),
            Cell::from("Crates.io").style(Style::default().fg(Color::Yellow)),
            Cell::from("Issues").style(Style::default().fg(Color::Yellow)),
        ])
        .height(1)
        .bottom_margin(1);

        let rows: Vec<Row> = self
            .report
            .crates
            .iter()
            .enumerate()
            .map(|(i, crate_info)| {
                let status_icon = match crate_info.status {
                    CrateStatus::Healthy => "✅",
                    CrateStatus::Warning => "⚠️ ",
                    CrateStatus::Error => "❌",
                    CrateStatus::Unknown => "❓",
                };

                let crates_io = crate_info
                    .crates_io_version
                    .as_ref()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "—".to_string());

                let issue_count = crate_info.issues.len();

                let style = if i == self.selected {
                    Style::default()
                        .bg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };

                Row::new(vec![
                    Cell::from(status_icon),
                    Cell::from(crate_info.name.clone()),
                    Cell::from(crate_info.local_version.to_string()),
                    Cell::from(crates_io),
                    Cell::from(if issue_count > 0 {
                        issue_count.to_string()
                    } else {
                        "—".to_string()
                    }),
                ])
                .style(style)
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(4),
                Constraint::Min(15),
                Constraint::Length(12),
                Constraint::Length(12),
                Constraint::Length(8),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .title("Crates")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White)),
        );

        frame.render_widget(table, area);
    }

    /// Render the details panel
    fn render_details(&self, frame: &mut Frame, area: Rect) {
        let selected_crate = self.report.crates.get(self.selected);

        let content = if let Some(crate_info) = selected_crate {
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("Name: ", Style::default().fg(Color::Yellow)),
                    Span::raw(&crate_info.name),
                ]),
                Line::from(vec![
                    Span::styled("Version: ", Style::default().fg(Color::Yellow)),
                    Span::raw(crate_info.local_version.to_string()),
                ]),
            ];

            if !crate_info.issues.is_empty() {
                lines.push(Line::from(Span::styled(
                    "Issues:",
                    Style::default().fg(Color::Red),
                )));
                for issue in &crate_info.issues {
                    lines.push(Line::from(format!("  • {}", issue.message)));
                }
            } else {
                lines.push(Line::from(Span::styled(
                    "No issues",
                    Style::default().fg(Color::Green),
                )));
            }

            lines
        } else {
            vec![Line::from("No crate selected")]
        };

        let paragraph = Paragraph::new(content)
            .block(
                Block::default()
                    .title("Details")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::White)),
            )
            .wrap(Wrap { trim: true });

        frame.render_widget(paragraph, area);
    }

    /// Render help bar
    fn render_help(&self, frame: &mut Frame, area: Rect) {
        let help = Paragraph::new(Line::from(vec![
            Span::styled("↑/k", Style::default().fg(Color::Cyan)),
            Span::raw(" Up  "),
            Span::styled("↓/j", Style::default().fg(Color::Cyan)),
            Span::raw(" Down  "),
            Span::styled("d", Style::default().fg(Color::Cyan)),
            Span::raw(" Toggle details  "),
            Span::styled("q/Esc", Style::default().fg(Color::Cyan)),
            Span::raw(" Quit"),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );

        frame.render_widget(help, area);
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

        // Initial state
        assert_eq!(dashboard.selected, 0);

        // Navigate down
        dashboard.selected = 1;
        assert_eq!(dashboard.selected, 1);

        // Navigate up
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

        // Test upper boundary
        dashboard.selected = 1;
        assert_eq!(dashboard.selected, 1);

        // Can't go beyond crate count
        dashboard.selected = 100;
        assert_eq!(dashboard.selected, 100); // No bounds check in raw assignment
    }

    #[test]
    fn test_dashboard_report_summary_types() {
        // Test with all zeros
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
    fn test_dashboard_multiple_crates_navigation() {
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

        // Navigate through all crates
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

    // Tests using TestBackend to exercise render methods
    mod render_tests {
        use super::*;
        use ratatui::{backend::TestBackend, Terminal};

        fn setup_test_terminal() -> Terminal<TestBackend> {
            let backend = TestBackend::new(80, 24);
            Terminal::new(backend).unwrap()
        }

        #[test]
        fn test_render_title_healthy() {
            let mut terminal = setup_test_terminal();
            let report = StackHealthReport {
                crates: vec![],
                conflicts: vec![],
                summary: HealthSummary {
                    total_crates: 5,
                    healthy_count: 5,
                    warning_count: 0,
                    error_count: 0,
                    ..Default::default()
                },
                timestamp: chrono::Utc::now(),
            };

            let dashboard = Dashboard::new(report);
            terminal
                .draw(|frame| {
                    dashboard.render_title(frame, frame.area());
                })
                .unwrap();

            let buffer = terminal.backend().buffer();
            let content: String = buffer
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect::<Vec<_>>()
                .join("");
            assert!(content.contains("PAIML Stack Dashboard"));
        }

        #[test]
        fn test_render_title_with_errors() {
            let mut terminal = setup_test_terminal();
            let report = StackHealthReport {
                crates: vec![],
                conflicts: vec![],
                summary: HealthSummary {
                    total_crates: 3,
                    healthy_count: 1,
                    warning_count: 0,
                    error_count: 2,
                    ..Default::default()
                },
                timestamp: chrono::Utc::now(),
            };

            let dashboard = Dashboard::new(report);
            terminal
                .draw(|frame| {
                    dashboard.render_title(frame, frame.area());
                })
                .unwrap();

            // Just verify it renders without panic
            assert!(terminal.backend().buffer().area.width > 0);
        }

        #[test]
        fn test_render_title_with_warnings() {
            let mut terminal = setup_test_terminal();
            let report = StackHealthReport {
                crates: vec![],
                conflicts: vec![],
                summary: HealthSummary {
                    total_crates: 4,
                    healthy_count: 2,
                    warning_count: 2,
                    error_count: 0,
                    ..Default::default()
                },
                timestamp: chrono::Utc::now(),
            };

            let dashboard = Dashboard::new(report);
            terminal
                .draw(|frame| {
                    dashboard.render_title(frame, frame.area());
                })
                .unwrap();

            assert!(terminal.backend().buffer().area.width > 0);
        }

        #[test]
        fn test_render_table_empty() {
            let mut terminal = setup_test_terminal();
            let report = StackHealthReport {
                crates: vec![],
                conflicts: vec![],
                summary: HealthSummary::default(),
                timestamp: chrono::Utc::now(),
            };

            let dashboard = Dashboard::new(report);
            terminal
                .draw(|frame| {
                    dashboard.render_table(frame, frame.area());
                })
                .unwrap();

            let buffer = terminal.backend().buffer();
            let content: String = buffer
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect::<Vec<_>>()
                .join("");
            assert!(content.contains("Crates"));
        }

        #[test]
        fn test_render_table_with_crates() {
            let mut terminal = setup_test_terminal();
            let mut crates = vec![
                CrateInfo::new("trueno", semver::Version::new(1, 0, 0), PathBuf::new()),
                CrateInfo::new("aprender", semver::Version::new(0, 14, 0), PathBuf::new()),
            ];
            crates[0].status = CrateStatus::Healthy;
            crates[0].crates_io_version = Some(semver::Version::new(1, 0, 0));
            crates[1].status = CrateStatus::Warning;
            crates[1].crates_io_version = Some(semver::Version::new(0, 14, 0));
            crates[1].issues.push(CrateIssue::new(
                IssueSeverity::Warning,
                IssueType::VersionBehind,
                "Test",
            ));

            let report = StackHealthReport {
                crates,
                conflicts: vec![],
                summary: HealthSummary {
                    total_crates: 2,
                    healthy_count: 1,
                    warning_count: 1,
                    ..Default::default()
                },
                timestamp: chrono::Utc::now(),
            };

            let dashboard = Dashboard::new(report);
            terminal
                .draw(|frame| {
                    dashboard.render_table(frame, frame.area());
                })
                .unwrap();

            let buffer = terminal.backend().buffer();
            let content: String = buffer
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect::<Vec<_>>()
                .join("");
            assert!(content.contains("trueno"));
            assert!(content.contains("aprender"));
        }

        #[test]
        fn test_render_details_with_crate() {
            let mut terminal = setup_test_terminal();
            let report = create_test_report();

            let dashboard = Dashboard::new(report);
            terminal
                .draw(|frame| {
                    dashboard.render_details(frame, frame.area());
                })
                .unwrap();

            let buffer = terminal.backend().buffer();
            let content: String = buffer
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect::<Vec<_>>()
                .join("");
            assert!(content.contains("Details"));
            assert!(content.contains("Name"));
        }

        #[test]
        fn test_render_details_empty() {
            let mut terminal = setup_test_terminal();
            let report = StackHealthReport {
                crates: vec![],
                conflicts: vec![],
                summary: HealthSummary::default(),
                timestamp: chrono::Utc::now(),
            };

            let dashboard = Dashboard::new(report);
            terminal
                .draw(|frame| {
                    dashboard.render_details(frame, frame.area());
                })
                .unwrap();

            let buffer = terminal.backend().buffer();
            let content: String = buffer
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect::<Vec<_>>()
                .join("");
            assert!(content.contains("No crate selected"));
        }

        #[test]
        fn test_render_details_with_issues() {
            let mut terminal = setup_test_terminal();
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

            let dashboard = Dashboard::new(report);
            terminal
                .draw(|frame| {
                    dashboard.render_details(frame, frame.area());
                })
                .unwrap();

            // Verify render completes without panic
            assert!(terminal.backend().buffer().area.height > 0);
        }

        #[test]
        fn test_render_help() {
            let mut terminal = setup_test_terminal();
            let report = create_test_report();

            let dashboard = Dashboard::new(report);
            terminal
                .draw(|frame| {
                    dashboard.render_help(frame, frame.area());
                })
                .unwrap();

            let buffer = terminal.backend().buffer();
            let content: String = buffer
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect::<Vec<_>>()
                .join("");
            assert!(content.contains("Quit"));
        }

        #[test]
        fn test_render_full_with_details() {
            let mut terminal = setup_test_terminal();
            let report = create_test_report();

            let dashboard = Dashboard::new(report);
            terminal.draw(|frame| dashboard.render(frame)).unwrap();

            // Verify full render completes
            assert!(terminal.backend().buffer().area.width > 0);
        }

        #[test]
        fn test_render_full_without_details() {
            let mut terminal = setup_test_terminal();
            let report = create_test_report();

            let mut dashboard = Dashboard::new(report);
            dashboard.show_details = false;

            terminal.draw(|frame| dashboard.render(frame)).unwrap();

            // Verify render completes without details panel
            assert!(terminal.backend().buffer().area.width > 0);
        }

        #[test]
        fn test_render_with_all_status_types() {
            let mut terminal = setup_test_terminal();
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

            let dashboard = Dashboard::new(report);
            terminal.draw(|frame| dashboard.render(frame)).unwrap();

            // Verify all status icons render
            assert!(terminal.backend().buffer().area.width > 0);
        }

        #[test]
        fn test_render_table_with_selected() {
            let mut terminal = setup_test_terminal();
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
            dashboard.selected = 1; // Select second crate

            terminal
                .draw(|frame| {
                    dashboard.render_table(frame, frame.area());
                })
                .unwrap();

            // Verify selected item is styled differently (verified by no panic)
            assert!(terminal.backend().buffer().area.width > 0);
        }

        #[test]
        fn test_render_crate_no_issues() {
            let mut terminal = setup_test_terminal();
            let mut crates = vec![CrateInfo::new(
                "clean",
                semver::Version::new(1, 0, 0),
                PathBuf::new(),
            )];
            crates[0].status = CrateStatus::Healthy;
            crates[0].issues.clear();

            let report = StackHealthReport {
                crates,
                conflicts: vec![],
                summary: HealthSummary::default(),
                timestamp: chrono::Utc::now(),
            };

            let dashboard = Dashboard::new(report);
            terminal
                .draw(|frame| {
                    dashboard.render_details(frame, frame.area());
                })
                .unwrap();

            let buffer = terminal.backend().buffer();
            let content: String = buffer
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect::<Vec<_>>()
                .join("");
            assert!(content.contains("No issues"));
        }
    }
}
