//! Agent TUI Dashboard
//!
//! Interactive terminal dashboard for monitoring agent loop execution.
//! Uses presentar-terminal for rendering and crossterm for terminal control.
//!
//! Launched by `batuta agent status --tui` or during `batuta agent run --stream`.

use crate::agent::driver::StreamEvent;
use crate::agent::manifest::AgentManifest;
use crate::agent::phase::LoopPhase;
use crate::agent::result::{StopReason, TokenUsage};

/// Truncate a string to `max_len`, appending "..." if needed.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_owned()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Snapshot of agent loop state for TUI rendering.
#[derive(Debug, Clone)]
pub struct AgentDashboardState {
    /// Agent name from manifest.
    pub agent_name: String,
    /// Current loop phase.
    pub phase: LoopPhase,
    /// Current iteration number.
    pub iteration: u32,
    /// Maximum iterations allowed.
    pub max_iterations: u32,
    /// Cumulative token usage.
    pub usage: TokenUsage,
    /// Token budget (None = unlimited).
    pub token_budget: Option<u64>,
    /// Tool calls executed.
    pub tool_calls: u32,
    /// Max tool calls allowed.
    pub max_tool_calls: u32,
    /// Recent text fragments.
    pub recent_text: Vec<String>,
    /// Recent tool call log.
    pub tool_log: Vec<ToolLogEntry>,
    /// Accumulated cost (USD).
    pub cost_usd: f64,
    /// Max cost budget.
    pub max_cost_usd: f64,
    /// Whether the loop is still running.
    pub running: bool,
    /// Final stop reason (if completed).
    pub stop_reason: Option<StopReason>,
}

/// A log entry for a tool call.
#[derive(Debug, Clone)]
pub struct ToolLogEntry {
    /// Tool name.
    pub name: String,
    /// Brief input summary.
    pub input_summary: String,
    /// Result summary.
    pub result_summary: String,
}

impl AgentDashboardState {
    /// Create initial state from manifest.
    pub fn from_manifest(manifest: &AgentManifest) -> Self {
        Self {
            agent_name: manifest.name.clone(),
            phase: LoopPhase::Perceive,
            iteration: 0,
            max_iterations: manifest.resources.max_iterations,
            usage: TokenUsage {
                input_tokens: 0,
                output_tokens: 0,
            },
            token_budget: manifest.resources.max_tokens_budget,
            tool_calls: 0,
            max_tool_calls: manifest.resources.max_tool_calls,
            recent_text: Vec::new(),
            tool_log: Vec::new(),
            cost_usd: 0.0,
            max_cost_usd: manifest.resources.max_cost_usd,
            running: true,
            stop_reason: None,
        }
    }

    /// Apply a stream event to update state.
    pub fn apply_event(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::PhaseChange { phase } => {
                self.phase = phase.clone();
            }
            StreamEvent::TextDelta { text } => {
                self.push_text(text);
            }
            StreamEvent::ToolUseStart { name, .. } => {
                self.push_tool_start(name);
            }
            StreamEvent::ToolUseEnd {
                name, result, ..
            } => {
                self.complete_tool(name, result);
            }
            StreamEvent::ContentComplete {
                stop_reason,
                usage,
            } => {
                self.usage = usage.clone();
                self.stop_reason = Some(stop_reason.clone());
                self.running = false;
            }
        }
    }

    fn push_text(&mut self, text: &str) {
        self.recent_text.push(text.to_owned());
        if self.recent_text.len() > 20 {
            self.recent_text.remove(0);
        }
    }

    fn push_tool_start(&mut self, name: &str) {
        self.tool_calls += 1;
        self.tool_log.push(ToolLogEntry {
            name: name.to_owned(),
            input_summary: String::new(),
            result_summary: "running...".into(),
        });
        if self.tool_log.len() > 10 {
            self.tool_log.remove(0);
        }
    }

    fn complete_tool(&mut self, name: &str, result: &str) {
        let Some(entry) = self
            .tool_log
            .iter_mut()
            .rev()
            .find(|e| e.name == name)
        else {
            return;
        };
        entry.result_summary = truncate_str(result, 60);
    }

    /// Iteration progress as percentage (0-100).
    pub fn iteration_pct(&self) -> u32 {
        if self.max_iterations == 0 {
            return 0;
        }
        (self.iteration * 100) / self.max_iterations
    }

    /// Token budget usage percentage (0-100), or 0 if unlimited.
    pub fn token_budget_pct(&self) -> u32 {
        let Some(budget) = self.token_budget else {
            return 0;
        };
        if budget == 0 {
            return 0;
        }
        let total =
            self.usage.input_tokens + self.usage.output_tokens;
        ((total * 100) / budget) as u32
    }
}

// ============================================================================
// TUI rendering (feature-gated)
// ============================================================================

#[cfg(feature = "presentar-terminal")]
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode,
        EnterAlternateScreen, LeaveAlternateScreen,
    },
};

#[cfg(feature = "presentar-terminal")]
use presentar_terminal::{
    CellBuffer, Color, DiffRenderer, Modifiers,
};

#[cfg(feature = "presentar-terminal")]
use std::io::{self, Write};

#[cfg(feature = "presentar-terminal")]
use std::time::Duration;

#[cfg(feature = "presentar-terminal")]
const CYAN: Color = Color {
    r: 0.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};

#[cfg(feature = "presentar-terminal")]
const GREEN: Color = Color {
    r: 0.2,
    g: 0.9,
    b: 0.2,
    a: 1.0,
};

#[cfg(feature = "presentar-terminal")]
const YELLOW: Color = Color {
    r: 1.0,
    g: 0.9,
    b: 0.0,
    a: 1.0,
};

#[cfg(feature = "presentar-terminal")]
const RED: Color = Color {
    r: 1.0,
    g: 0.2,
    b: 0.2,
    a: 1.0,
};

/// Interactive agent TUI dashboard.
#[cfg(feature = "presentar-terminal")]
pub struct AgentDashboard {
    state: AgentDashboardState,
    buffer: CellBuffer,
    renderer: DiffRenderer,
    width: u16,
    height: u16,
}

#[cfg(feature = "presentar-terminal")]
impl AgentDashboard {
    /// Create a new dashboard from agent state.
    pub fn new(state: AgentDashboardState) -> Self {
        let (width, height) =
            crossterm::terminal::size().unwrap_or((80, 24));
        Self {
            state,
            buffer: CellBuffer::new(width, height),
            renderer: DiffRenderer::new(),
            width,
            height,
        }
    }

    /// Run the dashboard loop, receiving events from a channel.
    pub fn run(
        mut self,
        rx: &mut tokio::sync::mpsc::Receiver<StreamEvent>,
    ) -> anyhow::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            cursor::Hide
        )?;

        let result = self.run_loop(&mut stdout, rx);

        disable_raw_mode()?;
        execute!(
            stdout,
            LeaveAlternateScreen,
            cursor::Show
        )?;

        result
    }

    fn run_loop(
        &mut self,
        stdout: &mut io::Stdout,
        rx: &mut tokio::sync::mpsc::Receiver<StreamEvent>,
    ) -> anyhow::Result<()> {
        loop {
            self.drain_events(rx);
            self.handle_resize();
            self.flush_frame(stdout)?;

            if Self::poll_quit_key(Duration::from_millis(100))? {
                return Ok(());
            }

            if !self.state.running {
                self.flush_frame(stdout)?;
                Self::wait_for_any_key()?;
                return Ok(());
            }
        }
    }

    fn drain_events(
        &mut self,
        rx: &mut tokio::sync::mpsc::Receiver<StreamEvent>,
    ) {
        while let Ok(ev) = rx.try_recv() {
            self.state.apply_event(&ev);
        }
    }

    fn handle_resize(&mut self) {
        let (w, h) =
            crossterm::terminal::size().unwrap_or((80, 24));
        if w != self.width || h != self.height {
            self.width = w;
            self.height = h;
            self.buffer.resize(w, h);
            self.renderer.reset();
        }
    }

    fn flush_frame(
        &mut self,
        stdout: &mut io::Stdout,
    ) -> anyhow::Result<()> {
        self.buffer.clear();
        self.render();
        self.renderer.flush(&mut self.buffer, stdout)?;
        stdout.flush()?;
        Ok(())
    }

    /// Returns `true` if quit key (q/Esc) was pressed.
    fn poll_quit_key(
        timeout: Duration,
    ) -> anyhow::Result<bool> {
        if !event::poll(timeout)? {
            return Ok(false);
        }
        let Event::Key(key) = event::read()? else {
            return Ok(false);
        };
        if key.kind != KeyEventKind::Press {
            return Ok(false);
        }
        Ok(matches!(
            key.code,
            KeyCode::Char('q') | KeyCode::Esc
        ))
    }

    /// Block until any key is pressed.
    fn wait_for_any_key() -> anyhow::Result<()> {
        loop {
            if !event::poll(Duration::from_millis(200))? {
                continue;
            }
            let Event::Key(key) = event::read()? else {
                continue;
            };
            if key.kind == KeyEventKind::Press {
                return Ok(());
            }
        }
    }

    /// Write a string character-by-character into the cell buffer.
    fn write_str(
        &mut self,
        x: u16,
        y: u16,
        s: &str,
        fg: Color,
    ) {
        let mut cx = x;
        for ch in s.chars() {
            if cx >= self.width {
                break;
            }
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            self.buffer.update(
                cx,
                y,
                encoded,
                fg,
                Color::TRANSPARENT,
                Modifiers::NONE,
            );
            cx = cx.saturating_add(1);
        }
    }

    fn render(&mut self) {
        self.render_title_bar(0);
        self.render_phase_indicator(2);
        self.render_divider(3);
        self.render_progress_bars(4);
        self.render_token_usage(8);
        self.render_divider(10);
        self.render_tool_log(11);
        self.render_recent_text(
            self.height.saturating_sub(6),
        );
        self.render_help_bar(
            self.height.saturating_sub(1),
        );
    }

    fn render_divider(&mut self, row: u16) {
        let w = (self.width as usize).min(80);
        let divider: String = "─".repeat(w);
        self.write_str(0, row, &divider, Color::WHITE);
    }

    fn render_title_bar(&mut self, row: u16) {
        let title = format!(
            " Agent: {} ",
            self.state.agent_name,
        );
        // Title with CYAN background
        for (i, ch) in title.chars().enumerate() {
            if (i as u16) >= self.width {
                break;
            }
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.buffer.update(
                i as u16,
                row,
                s,
                Color::BLACK,
                CYAN,
                Modifiers::BOLD,
            );
        }

        let status = if self.state.running {
            " RUNNING "
        } else {
            " DONE "
        };
        let status_color = if self.state.running {
            GREEN
        } else {
            CYAN
        };
        let x =
            self.width.saturating_sub(status.len() as u16 + 1);
        for (i, ch) in status.chars().enumerate() {
            let cx = x + i as u16;
            if cx >= self.width {
                break;
            }
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.buffer.update(
                cx,
                row,
                s,
                Color::BLACK,
                status_color,
                Modifiers::BOLD,
            );
        }
    }

    fn render_phase_indicator(&mut self, row: u16) {
        let phase_str = format!("{:?}", self.state.phase);
        let label = format!(
            "Phase: {} | Iteration: {}/{}",
            phase_str,
            self.state.iteration,
            self.state.max_iterations,
        );
        self.write_str(1, row, &label, CYAN);

        if let Some(ref sr) = self.state.stop_reason {
            let reason = format!(" Stop: {sr:?}");
            let x = label.len() as u16 + 3;
            self.write_str(x, row, &reason, YELLOW);
        }
    }

    fn render_progress_bars(&mut self, row: u16) {
        let bar_width = 30usize;

        // Iteration progress
        let iter_pct = self.state.iteration_pct();
        let iter_label =
            format!("Iterations: {:>3}%", iter_pct);
        self.write_str(1, row, &iter_label, Color::WHITE);
        self.render_bar(
            iter_label.len() as u16 + 2,
            row,
            bar_width,
            iter_pct,
        );

        // Tool calls progress
        let tool_pct = if self.state.max_tool_calls > 0 {
            (self.state.tool_calls * 100)
                / self.state.max_tool_calls
        } else {
            0
        };
        let tool_label =
            format!("Tool calls: {:>3}%", tool_pct);
        self.write_str(1, row + 1, &tool_label, Color::WHITE);
        self.render_bar(
            tool_label.len() as u16 + 2,
            row + 1,
            bar_width,
            tool_pct,
        );

        // Token budget (if set)
        if self.state.token_budget.is_some() {
            let tok_pct = self.state.token_budget_pct();
            let tok_label =
                format!("Token budget: {:>3}%", tok_pct);
            self.write_str(
                1,
                row + 2,
                &tok_label,
                Color::WHITE,
            );
            self.render_bar(
                tok_label.len() as u16 + 2,
                row + 2,
                bar_width,
                tok_pct,
            );
        }
    }

    fn render_bar(
        &mut self,
        x: u16,
        y: u16,
        width: usize,
        pct: u32,
    ) {
        let filled = (pct as usize * width) / 100;
        let color = if pct >= 90 {
            RED
        } else if pct >= 70 {
            YELLOW
        } else {
            GREEN
        };
        let bar: String = "█"
            .repeat(filled)
            + &"░".repeat(width.saturating_sub(filled));
        self.write_str(
            x,
            y,
            &format!("[{bar}]"),
            color,
        );
    }

    fn render_token_usage(&mut self, row: u16) {
        let total = self.state.usage.input_tokens
            + self.state.usage.output_tokens;
        let usage_str = format!(
            "Tokens: {} in / {} out = {} total",
            self.state.usage.input_tokens,
            self.state.usage.output_tokens,
            total,
        );
        self.write_str(1, row, &usage_str, Color::WHITE);

        if self.state.max_cost_usd > 0.0 {
            let cost_str = format!(
                "  Cost: ${:.4} / ${:.4}",
                self.state.cost_usd,
                self.state.max_cost_usd,
            );
            let x = usage_str.len() as u16 + 2;
            self.write_str(x, row, &cost_str, YELLOW);
        }
    }

    fn render_tool_log(&mut self, row: u16) {
        self.write_str(1, row, "Tool Log:", CYAN);

        let max_entries = (self
            .height
            .saturating_sub(row + 8) as usize)
            .min(self.state.tool_log.len());
        let max_w = self.width.saturating_sub(2) as usize;

        let lines: Vec<String> = self
            .state
            .tool_log
            .iter()
            .rev()
            .take(max_entries)
            .map(|entry| {
                let line = format!(
                    "  {} → {}",
                    entry.name, entry.result_summary,
                );
                if line.len() > max_w {
                    format!(
                        "{}...",
                        &line[..max_w.saturating_sub(3)]
                    )
                } else {
                    line
                }
            })
            .collect();

        for (i, line) in lines.iter().enumerate() {
            self.write_str(
                1,
                row + 1 + i as u16,
                line,
                Color::WHITE,
            );
        }
    }

    fn render_recent_text(&mut self, row: u16) {
        self.write_str(1, row, "Output:", CYAN);

        let text: String =
            self.state.recent_text.join("");
        let w = self.width.saturating_sub(4) as usize;
        let max_chars = w * 4;
        let display = if text.len() > max_chars {
            &text[text.len() - max_chars..]
        } else {
            &text
        };

        for (i, chunk) in display
            .as_bytes()
            .chunks(w.max(1))
            .take(4)
            .enumerate()
        {
            let s = String::from_utf8_lossy(chunk);
            self.write_str(
                2,
                row + 1 + i as u16,
                &s,
                GREEN,
            );
        }
    }

    fn render_help_bar(&mut self, row: u16) {
        let help = if self.state.running {
            " q: quit "
        } else {
            " Press any key to exit "
        };
        // White background for help bar
        for (i, ch) in help.chars().enumerate() {
            if (i as u16) >= self.width {
                break;
            }
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.buffer.update(
                i as u16,
                row,
                s,
                Color::BLACK,
                Color::WHITE,
                Modifiers::NONE,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_state_from_manifest() {
        let manifest = AgentManifest::default();
        let state =
            AgentDashboardState::from_manifest(&manifest);
        assert!(state.running);
        assert_eq!(state.iteration, 0);
        assert_eq!(
            state.max_iterations,
            manifest.resources.max_iterations,
        );
    }

    #[test]
    fn test_apply_text_delta() {
        let mut state = AgentDashboardState::from_manifest(
            &AgentManifest::default(),
        );
        state.apply_event(&StreamEvent::TextDelta {
            text: "hello".into(),
        });
        assert_eq!(state.recent_text.len(), 1);
        assert_eq!(state.recent_text[0], "hello");
    }

    #[test]
    fn test_apply_content_complete() {
        let mut state = AgentDashboardState::from_manifest(
            &AgentManifest::default(),
        );
        state.apply_event(&StreamEvent::ContentComplete {
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
            },
        });
        assert!(!state.running);
        assert_eq!(state.usage.input_tokens, 100);
        assert!(matches!(
            state.stop_reason,
            Some(StopReason::EndTurn)
        ));
    }

    #[test]
    fn test_apply_tool_use_events() {
        let mut state = AgentDashboardState::from_manifest(
            &AgentManifest::default(),
        );
        state.apply_event(&StreamEvent::ToolUseStart {
            id: "1".into(),
            name: "rag".into(),
        });
        assert_eq!(state.tool_calls, 1);
        assert_eq!(state.tool_log.len(), 1);
        assert_eq!(state.tool_log[0].name, "rag");

        state.apply_event(&StreamEvent::ToolUseEnd {
            id: "1".into(),
            name: "rag".into(),
            result: "found 3 results".into(),
        });
        assert_eq!(
            state.tool_log[0].result_summary,
            "found 3 results",
        );
    }

    #[test]
    fn test_apply_phase_change() {
        let mut state = AgentDashboardState::from_manifest(
            &AgentManifest::default(),
        );
        state.apply_event(&StreamEvent::PhaseChange {
            phase: LoopPhase::Act {
                tool_name: "rag".into(),
            },
        });
        assert!(matches!(state.phase, LoopPhase::Act { .. }));
    }

    #[test]
    fn test_iteration_pct() {
        let mut state = AgentDashboardState::from_manifest(
            &AgentManifest::default(),
        );
        state.max_iterations = 10;
        state.iteration = 3;
        assert_eq!(state.iteration_pct(), 30);
    }

    #[test]
    fn test_iteration_pct_zero_max() {
        let mut state = AgentDashboardState::from_manifest(
            &AgentManifest::default(),
        );
        state.max_iterations = 0;
        assert_eq!(state.iteration_pct(), 0);
    }

    #[test]
    fn test_token_budget_pct() {
        let mut state = AgentDashboardState::from_manifest(
            &AgentManifest::default(),
        );
        state.token_budget = Some(1000);
        state.usage = TokenUsage {
            input_tokens: 400,
            output_tokens: 100,
        };
        assert_eq!(state.token_budget_pct(), 50);
    }

    #[test]
    fn test_token_budget_pct_unlimited() {
        let state = AgentDashboardState::from_manifest(
            &AgentManifest::default(),
        );
        assert_eq!(state.token_budget_pct(), 0);
    }

    #[test]
    fn test_recent_text_capped_at_20() {
        let mut state = AgentDashboardState::from_manifest(
            &AgentManifest::default(),
        );
        for i in 0..25 {
            state.apply_event(&StreamEvent::TextDelta {
                text: format!("t{i}"),
            });
        }
        assert_eq!(state.recent_text.len(), 20);
        assert_eq!(state.recent_text[0], "t5");
    }

    #[test]
    fn test_tool_log_capped_at_10() {
        let mut state = AgentDashboardState::from_manifest(
            &AgentManifest::default(),
        );
        for i in 0..12 {
            state.apply_event(&StreamEvent::ToolUseStart {
                id: format!("{i}"),
                name: format!("tool_{i}"),
            });
        }
        assert_eq!(state.tool_log.len(), 10);
        assert_eq!(state.tool_log[0].name, "tool_2");
    }
}
