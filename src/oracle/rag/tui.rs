//! TUI Dashboard for RAG Oracle
//!
//! Interactive terminal UI for visualizing index health, query results,
//! and system metrics. Implements Toyota Way Principle 7: Visual Control.

#![allow(dead_code)]

#[cfg(feature = "native")]
use std::collections::VecDeque;
#[cfg(feature = "native")]
use std::io::{self, Stdout};
#[cfg(feature = "native")]
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
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Row, Sparkline, Table},
    Frame, Terminal,
};

use super::types::{IndexHealthMetrics, RelevanceMetrics};

/// Query record for history display
#[derive(Debug, Clone)]
pub struct QueryRecord {
    /// Query timestamp (Unix epoch ms)
    pub timestamp_ms: u64,
    /// Query text (truncated)
    pub query: String,
    /// Primary component matched
    pub component: String,
    /// Query latency (ms)
    pub latency_ms: u64,
    /// Success flag
    pub success: bool,
}

/// TUI Dashboard state
#[cfg(feature = "native")]
pub struct OracleDashboard {
    /// Index health metrics
    pub index_health: IndexHealthMetrics,
    /// Query history (most recent first)
    pub query_history: VecDeque<QueryRecord>,
    /// Latency samples for sparkline
    pub latency_samples: Vec<u64>,
    /// Retrieval quality metrics
    pub retrieval_metrics: RelevanceMetrics,
    /// Selected component index
    selected_component: usize,
    /// Max history size
    max_history: usize,
    /// Refresh interval
    refresh_interval: Duration,
}

#[cfg(feature = "native")]
impl OracleDashboard {
    /// Create a new dashboard
    pub fn new() -> Self {
        Self {
            index_health: IndexHealthMetrics::default(),
            query_history: VecDeque::new(),
            latency_samples: Vec::new(),
            retrieval_metrics: RelevanceMetrics::default(),
            selected_component: 0,
            max_history: 100,
            refresh_interval: Duration::from_millis(100),
        }
    }

    /// Add a query to history
    pub fn record_query(&mut self, record: QueryRecord) {
        self.latency_samples.push(record.latency_ms);
        if self.latency_samples.len() > 50 {
            self.latency_samples.remove(0);
        }
        self.query_history.push_front(record);
        if self.query_history.len() > self.max_history {
            self.query_history.pop_back();
        }
    }

    /// Update index health metrics
    pub fn update_health(&mut self, health: IndexHealthMetrics) {
        self.index_health = health;
    }

    /// Run the TUI dashboard
    pub fn run(&mut self) -> anyhow::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = self.run_loop(&mut terminal);

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        result
    }

    /// Main event loop
    fn run_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ) -> anyhow::Result<()> {
        loop {
            terminal.draw(|frame| self.render(frame))?;

            if event::poll(self.refresh_interval)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                            KeyCode::Up | KeyCode::Char('k') => {
                                if self.selected_component > 0 {
                                    self.selected_component -= 1;
                                }
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                let max =
                                    self.index_health.docs_per_component.len().saturating_sub(1);
                                if self.selected_component < max {
                                    self.selected_component += 1;
                                }
                            }
                            KeyCode::Char('r') => {
                                // Trigger refresh - placeholder
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
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(12),   // Main panels
                Constraint::Length(8), // Query history
                Constraint::Length(1), // Help
            ])
            .split(frame.area());

        self.render_header(frame, chunks[0]);
        self.render_panels(frame, chunks[1]);
        self.render_history(frame, chunks[2]);
        self.render_help(frame, chunks[3]);
    }

    /// Render header with overall health
    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let coverage = self.index_health.coverage_percent;
        let total_docs: usize = self
            .index_health
            .docs_per_component
            .iter()
            .map(|(_, c)| c)
            .sum();

        let gauge = Gauge::default()
            .block(
                Block::default()
                    .title(" Oracle RAG Dashboard ")
                    .borders(Borders::ALL),
            )
            .gauge_style(Style::default().fg(self.health_color(coverage)))
            .percent(coverage)
            .label(format!(
                "Index Health: {}%  |  Docs: {}",
                coverage, total_docs
            ));

        frame.render_widget(gauge, area);
    }

    /// Render main panels (index status, latency, quality)
    fn render_panels(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(40),
                Constraint::Percentage(30),
                Constraint::Percentage(30),
            ])
            .split(area);

        self.render_index_status(frame, chunks[0]);
        self.render_latency(frame, chunks[1]);
        self.render_quality(frame, chunks[2]);
    }

    /// Render index status by component
    fn render_index_status(&self, frame: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self
            .index_health
            .docs_per_component
            .iter()
            .enumerate()
            .map(|(i, (name, count))| {
                let bar = render_bar(*count, 500, 15);
                let style = if i == self.selected_component {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                let marker = if i == self.selected_component {
                    ">"
                } else {
                    " "
                };
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{} ", marker), style),
                    Span::styled(format!("{:12}", name), style),
                    Span::raw(" "),
                    Span::styled(bar, Style::default().fg(Color::Cyan)),
                    Span::raw(format!(" {}", count)),
                ]))
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .title(" Index Status ")
                .borders(Borders::ALL),
        );

        frame.render_widget(list, area);
    }

    /// Render latency sparkline
    fn render_latency(&self, frame: &mut Frame, area: Rect) {
        let inner = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(4), Constraint::Length(3)])
            .split(area);

        // Sparkline
        let data: Vec<u64> = self.latency_samples.clone();
        let sparkline = Sparkline::default()
            .block(
                Block::default()
                    .title(" Query Latency ")
                    .borders(Borders::ALL),
            )
            .data(&data)
            .style(Style::default().fg(Color::Cyan));

        frame.render_widget(sparkline, inner[0]);

        // Stats
        let (avg, p99) = if !self.latency_samples.is_empty() {
            let sum: u64 = self.latency_samples.iter().sum();
            let avg = sum / self.latency_samples.len() as u64;
            let mut sorted = self.latency_samples.clone();
            sorted.sort();
            let p99_idx = (sorted.len() as f64 * 0.99) as usize;
            let p99 = sorted
                .get(p99_idx.min(sorted.len() - 1))
                .copied()
                .unwrap_or(0);
            (avg, p99)
        } else {
            (0, 0)
        };

        let stats = Paragraph::new(format!("avg: {}ms  p99: {}ms", avg, p99))
            .block(Block::default().borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM));

        frame.render_widget(stats, inner[1]);
    }

    /// Render quality metrics
    fn render_quality(&self, frame: &mut Frame, area: Rect) {
        let metrics = &self.retrieval_metrics;

        // Create owned strings to avoid temporary value lifetime issues
        let mrr_val = format!("{:.3}", metrics.mrr);
        let mrr_bar = render_bar((metrics.mrr * 100.0) as usize, 100, 12);
        let ndcg_val = format!("{:.3}", metrics.ndcg_at_k);
        let ndcg_bar = render_bar((metrics.ndcg_at_k * 100.0) as usize, 100, 12);
        let recall_val = format!("{:.3}", metrics.recall_at_k);
        let recall_bar = render_bar((metrics.recall_at_k * 100.0) as usize, 100, 12);

        let rows = vec![
            Row::new(vec!["MRR", &mrr_val, &mrr_bar]),
            Row::new(vec!["NDCG", &ndcg_val, &ndcg_bar]),
            Row::new(vec!["R@10", &recall_val, &recall_bar]),
        ];

        let table = Table::new(
            rows,
            [
                Constraint::Length(6),
                Constraint::Length(6),
                Constraint::Min(12),
            ],
        )
        .block(
            Block::default()
                .title(" Retrieval Quality ")
                .borders(Borders::ALL),
        )
        .style(Style::default().fg(Color::White));

        frame.render_widget(table, area);
    }

    /// Render query history
    fn render_history(&self, frame: &mut Frame, area: Rect) {
        let rows: Vec<Row> = self
            .query_history
            .iter()
            .take(5)
            .map(|record| {
                let time = format_timestamp(record.timestamp_ms);
                let status = if record.success { "+" } else { "x" };
                let status_style = if record.success {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::Red)
                };

                Row::new(vec![
                    time,
                    truncate_query(&record.query, 30),
                    record.component.clone(),
                    format!("{}ms", record.latency_ms),
                    status.to_string(),
                ])
                .style(status_style)
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(10),
                Constraint::Min(30),
                Constraint::Length(12),
                Constraint::Length(8),
                Constraint::Length(3),
            ],
        )
        .block(
            Block::default()
                .title(" Recent Queries ")
                .borders(Borders::ALL),
        )
        .header(
            Row::new(vec!["Time", "Query", "Component", "Latency", ""])
                .style(Style::default().add_modifier(Modifier::BOLD)),
        );

        frame.render_widget(table, area);
    }

    /// Render help bar
    fn render_help(&self, frame: &mut Frame, area: Rect) {
        let help = Paragraph::new(" [q]uit  [r]efresh  [^/v]navigate ")
            .style(Style::default().fg(Color::DarkGray));

        frame.render_widget(help, area);
    }

    /// Get color based on health percentage
    fn health_color(&self, percent: u16) -> Color {
        match percent {
            0..=60 => Color::Red,
            61..=80 => Color::Yellow,
            _ => Color::Green,
        }
    }
}

#[cfg(feature = "native")]
impl Default for OracleDashboard {
    fn default() -> Self {
        Self::new()
    }
}

/// Render a horizontal bar
fn render_bar(value: usize, max: usize, width: usize) -> String {
    let filled = if max > 0 {
        (value * width / max).min(width)
    } else {
        0
    };
    let empty = width.saturating_sub(filled);
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

/// Format timestamp for display
fn format_timestamp(timestamp_ms: u64) -> String {
    // Simple formatting - in production would use chrono
    let secs = timestamp_ms / 1000;
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let secs = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, secs)
}

/// Truncate query for display
fn truncate_query(query: &str, max_len: usize) -> String {
    if query.len() <= max_len {
        query.to_string()
    } else {
        format!("{}...", &query[..max_len - 3])
    }
}

/// Inline visualizations for CLI output
pub mod inline {
    /// Render a horizontal bar chart
    pub fn bar(value: f64, max: f64, width: usize) -> String {
        let filled = if max > 0.0 {
            ((value / max) * width as f64) as usize
        } else {
            0
        };
        let empty = width.saturating_sub(filled);
        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
    }

    /// Render a sparkline from values
    pub fn sparkline(values: &[f64]) -> String {
        const BARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        if values.is_empty() {
            return String::new();
        }

        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = max - min;

        values
            .iter()
            .map(|v| {
                let idx = if range == 0.0 {
                    0
                } else {
                    ((v - min) / range * 7.0) as usize
                };
                BARS[idx.min(7)]
            })
            .collect()
    }

    /// Format a score as a bar with percentage
    pub fn score_bar(score: f64, width: usize) -> String {
        let pct = (score * 100.0) as usize;
        format!("{} {:3}%", bar(score, 1.0, width), pct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_bar() {
        let bar = render_bar(50, 100, 10);
        assert_eq!(bar.chars().filter(|c| *c == '█').count(), 5);
        assert_eq!(bar.chars().filter(|c| *c == '░').count(), 5);
    }

    #[test]
    fn test_render_bar_full() {
        let bar = render_bar(100, 100, 10);
        assert_eq!(bar.chars().filter(|c| *c == '█').count(), 10);
    }

    #[test]
    fn test_render_bar_empty() {
        let bar = render_bar(0, 100, 10);
        assert_eq!(bar.chars().filter(|c| *c == '░').count(), 10);
    }

    #[test]
    fn test_format_timestamp() {
        let ts = format_timestamp(45296000); // 12:34:56
        assert_eq!(ts, "12:34:56");
    }

    #[test]
    fn test_truncate_query_short() {
        let q = truncate_query("short", 10);
        assert_eq!(q, "short");
    }

    #[test]
    fn test_truncate_query_long() {
        let q = truncate_query("this is a very long query", 15);
        assert!(q.ends_with("..."));
        assert!(q.len() <= 15);
    }

    #[test]
    fn test_inline_bar() {
        let bar = inline::bar(0.5, 1.0, 10);
        assert_eq!(bar.chars().filter(|c| *c == '█').count(), 5);
    }

    #[test]
    fn test_inline_sparkline() {
        let spark = inline::sparkline(&[0.0, 0.5, 1.0, 0.5, 0.0]);
        assert_eq!(spark.chars().count(), 5); // Unicode chars, not bytes
        assert!(spark.contains('▁'));
        assert!(spark.contains('█'));
    }

    #[test]
    fn test_inline_sparkline_empty() {
        let spark = inline::sparkline(&[]);
        assert!(spark.is_empty());
    }

    #[test]
    fn test_inline_score_bar() {
        let bar = inline::score_bar(0.85, 10);
        assert!(bar.contains("85%"));
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_dashboard_creation() {
        let dashboard = OracleDashboard::new();
        assert!(dashboard.query_history.is_empty());
        assert!(dashboard.latency_samples.is_empty());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_dashboard_record_query() {
        let mut dashboard = OracleDashboard::new();
        dashboard.record_query(QueryRecord {
            timestamp_ms: 1234567890,
            query: "test query".to_string(),
            component: "trueno".to_string(),
            latency_ms: 50,
            success: true,
        });

        assert_eq!(dashboard.query_history.len(), 1);
        assert_eq!(dashboard.latency_samples.len(), 1);
    }
}
