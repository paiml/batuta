//! ANSI Color Support
//!
//! Zero-dependency replacement for the `colored` crate.
//! Provides ANSI escape code based text styling via trait extension.
//!
//! Created as part of DEP-REDUCE to eliminate external dependencies.

#![allow(dead_code)] // Intentionally provide full color palette

use std::fmt;

/// ANSI escape codes for terminal colors and styles
pub mod codes {
    // Reset
    pub const RESET: &str = "\x1b[0m";

    // Styles
    pub const BOLD: &str = "\x1b[1m";
    pub const DIMMED: &str = "\x1b[2m";

    // Standard colors (foreground)
    pub const RED: &str = "\x1b[31m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const CYAN: &str = "\x1b[36m";
    pub const WHITE: &str = "\x1b[37m";

    // Bright colors (foreground)
    pub const BRIGHT_RED: &str = "\x1b[91m";
    pub const BRIGHT_GREEN: &str = "\x1b[92m";
    pub const BRIGHT_YELLOW: &str = "\x1b[93m";
    pub const BRIGHT_BLUE: &str = "\x1b[94m";
    pub const BRIGHT_MAGENTA: &str = "\x1b[95m";
    pub const BRIGHT_CYAN: &str = "\x1b[96m";
    pub const BRIGHT_WHITE: &str = "\x1b[97m";

    // Background colors
    pub const ON_RED: &str = "\x1b[41m";
}

/// A styled string that wraps content with ANSI codes
#[derive(Clone)]
pub struct StyledString {
    content: String,
    styles: Vec<&'static str>,
}

impl StyledString {
    fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            styles: Vec::new(),
        }
    }

    fn with_style(mut self, style: &'static str) -> Self {
        self.styles.push(style);
        self
    }

    // Standard colors
    pub fn red(self) -> Self {
        self.with_style(codes::RED)
    }
    pub fn green(self) -> Self {
        self.with_style(codes::GREEN)
    }
    pub fn yellow(self) -> Self {
        self.with_style(codes::YELLOW)
    }
    pub fn blue(self) -> Self {
        self.with_style(codes::BLUE)
    }
    pub fn magenta(self) -> Self {
        self.with_style(codes::MAGENTA)
    }
    pub fn cyan(self) -> Self {
        self.with_style(codes::CYAN)
    }
    pub fn white(self) -> Self {
        self.with_style(codes::WHITE)
    }

    // Bright colors
    pub fn bright_red(self) -> Self {
        self.with_style(codes::BRIGHT_RED)
    }
    pub fn bright_green(self) -> Self {
        self.with_style(codes::BRIGHT_GREEN)
    }
    pub fn bright_yellow(self) -> Self {
        self.with_style(codes::BRIGHT_YELLOW)
    }
    pub fn bright_blue(self) -> Self {
        self.with_style(codes::BRIGHT_BLUE)
    }
    pub fn bright_magenta(self) -> Self {
        self.with_style(codes::BRIGHT_MAGENTA)
    }
    pub fn bright_cyan(self) -> Self {
        self.with_style(codes::BRIGHT_CYAN)
    }
    pub fn bright_white(self) -> Self {
        self.with_style(codes::BRIGHT_WHITE)
    }

    // Background colors
    pub fn on_red(self) -> Self {
        self.with_style(codes::ON_RED)
    }

    // Styles
    pub fn bold(self) -> Self {
        self.with_style(codes::BOLD)
    }
    pub fn dimmed(self) -> Self {
        self.with_style(codes::DIMMED)
    }
}

impl fmt::Display for StyledString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Apply all styles
        for style in &self.styles {
            write!(f, "{}", style)?;
        }
        // Write content
        write!(f, "{}", self.content)?;
        // Reset
        write!(f, "{}", codes::RESET)
    }
}

/// Extension trait to add color methods to strings
/// Uses AsRef<str> to work with both &str, String, and &String without moving
pub trait Colorize {
    fn to_styled(&self) -> StyledString;

    // Standard colors
    fn red(&self) -> StyledString {
        self.to_styled().red()
    }
    fn green(&self) -> StyledString {
        self.to_styled().green()
    }
    fn yellow(&self) -> StyledString {
        self.to_styled().yellow()
    }
    fn blue(&self) -> StyledString {
        self.to_styled().blue()
    }
    fn magenta(&self) -> StyledString {
        self.to_styled().magenta()
    }
    fn cyan(&self) -> StyledString {
        self.to_styled().cyan()
    }
    fn white(&self) -> StyledString {
        self.to_styled().white()
    }

    // Bright colors
    fn bright_red(&self) -> StyledString {
        self.to_styled().bright_red()
    }
    fn bright_green(&self) -> StyledString {
        self.to_styled().bright_green()
    }
    fn bright_yellow(&self) -> StyledString {
        self.to_styled().bright_yellow()
    }
    fn bright_blue(&self) -> StyledString {
        self.to_styled().bright_blue()
    }
    fn bright_magenta(&self) -> StyledString {
        self.to_styled().bright_magenta()
    }
    fn bright_cyan(&self) -> StyledString {
        self.to_styled().bright_cyan()
    }
    fn bright_white(&self) -> StyledString {
        self.to_styled().bright_white()
    }

    // Background colors
    fn on_red(&self) -> StyledString {
        self.to_styled().on_red()
    }

    // Styles
    fn bold(&self) -> StyledString {
        self.to_styled().bold()
    }
    fn dimmed(&self) -> StyledString {
        self.to_styled().dimmed()
    }
}

impl Colorize for str {
    fn to_styled(&self) -> StyledString {
        StyledString::new(self)
    }
}

impl Colorize for String {
    fn to_styled(&self) -> StyledString {
        StyledString::new(self.as_str())
    }
}

impl Colorize for StyledString {
    fn to_styled(&self) -> StyledString {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_color() {
        let s = "hello".red();
        assert!(s.to_string().contains("\x1b[31m"));
        assert!(s.to_string().contains("hello"));
        assert!(s.to_string().contains("\x1b[0m"));
    }

    #[test]
    fn test_chained_styles() {
        let s = "hello".bright_cyan().bold();
        let output = s.to_string();
        assert!(output.contains("\x1b[96m")); // bright cyan
        assert!(output.contains("\x1b[1m")); // bold
        assert!(output.contains("hello"));
        assert!(output.contains("\x1b[0m")); // reset
    }

    #[test]
    fn test_string_owned() {
        let s = String::from("world").green().dimmed();
        assert!(s.to_string().contains("\x1b[32m"));
        assert!(s.to_string().contains("\x1b[2m"));
    }

    #[test]
    fn test_borrowed_string() {
        let owned = String::from("borrowed");
        let s = owned.yellow(); // Should not move owned
        assert!(s.to_string().contains("\x1b[33m"));
        // owned is still usable
        assert_eq!(owned, "borrowed");
    }
}
