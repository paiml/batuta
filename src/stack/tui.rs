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
}
