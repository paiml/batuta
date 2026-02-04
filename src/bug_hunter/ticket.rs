//! PMAT Work Ticket Integration (BH-12)
//!
//! Parses PMAT work tickets to scope bug hunting to active development areas.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A parsed PMAT work ticket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmatTicket {
    /// Ticket identifier (e.g., "PMAT-1234")
    pub id: String,
    /// Ticket title
    pub title: String,
    /// Description
    pub description: String,
    /// Affected files/paths
    pub affected_paths: Vec<PathBuf>,
    /// Expected behavior description
    pub expected_behavior: Option<String>,
    /// Acceptance criteria
    pub acceptance_criteria: Vec<String>,
    /// Priority
    pub priority: TicketPriority,
}

/// Ticket priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TicketPriority {
    Critical,
    High,
    #[default]
    Medium,
    Low,
}

impl PmatTicket {
    /// Parse a ticket from file path or ID.
    pub fn parse(ticket_ref: &str, project_path: &Path) -> Result<Self, String> {
        // Check if it's a file path
        let ticket_path = if ticket_ref.ends_with(".md") || ticket_ref.ends_with(".yaml") {
            PathBuf::from(ticket_ref)
        } else {
            // Try to find ticket by ID in .pmat/tickets/
            let pmat_dir = project_path.join(".pmat/tickets");
            let md_path = pmat_dir.join(format!("{}.md", ticket_ref));
            let yaml_path = pmat_dir.join(format!("{}.yaml", ticket_ref));

            if md_path.exists() {
                md_path
            } else if yaml_path.exists() {
                yaml_path
            } else {
                // Try GitHub issue
                return Self::from_github_issue(ticket_ref);
            }
        };

        if ticket_path.extension().map(|e| e == "yaml").unwrap_or(false) {
            Self::from_yaml(&ticket_path)
        } else {
            Self::from_markdown(&ticket_path)
        }
    }

    /// Parse from YAML file.
    fn from_yaml(path: &Path) -> Result<Self, String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("Failed to read ticket: {}", e))?;

        serde_yaml::from_str(&content).map_err(|e| format!("Failed to parse YAML ticket: {}", e))
    }

    /// Parse from Markdown file.
    fn from_markdown(path: &Path) -> Result<Self, String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("Failed to read ticket: {}", e))?;

        parse_markdown_ticket(&content, path)
    }

    /// Parse from GitHub issue (placeholder - would use gh CLI).
    fn from_github_issue(issue_ref: &str) -> Result<Self, String> {
        // Extract issue number
        let issue_num: u32 = issue_ref
            .trim_start_matches("PMAT-")
            .trim_start_matches('#')
            .parse()
            .map_err(|_| format!("Invalid issue reference: {}", issue_ref))?;

        // For now, return a placeholder - in production would call `gh issue view`
        Ok(Self {
            id: format!("PMAT-{}", issue_num),
            title: format!("GitHub Issue #{}", issue_num),
            description: "Loaded from GitHub".to_string(),
            affected_paths: vec![PathBuf::from("src")],
            expected_behavior: None,
            acceptance_criteria: Vec::new(),
            priority: TicketPriority::Medium,
        })
    }

    /// Get target paths for scoped analysis.
    pub fn target_paths(&self) -> Vec<PathBuf> {
        if self.affected_paths.is_empty() {
            vec![PathBuf::from("src")]
        } else {
            self.affected_paths.clone()
        }
    }
}

/// Parse a markdown ticket file.
fn parse_markdown_ticket(content: &str, path: &Path) -> Result<PmatTicket, String> {
    let mut ticket = PmatTicket {
        id: path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "UNKNOWN".to_string()),
        title: String::new(),
        description: String::new(),
        affected_paths: Vec::new(),
        expected_behavior: None,
        acceptance_criteria: Vec::new(),
        priority: TicketPriority::Medium,
    };

    let mut current_section = "";
    let mut description_lines: Vec<&str> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();

        // Parse title from first # header
        if trimmed.starts_with("# ") && ticket.title.is_empty() {
            ticket.title = trimmed[2..].to_string();
            continue;
        }

        // Track sections
        if trimmed.starts_with("## ") {
            current_section = trimmed[3..].trim();
            continue;
        }

        // Parse content based on section
        match current_section.to_lowercase().as_str() {
            "description" | "summary" => {
                if !trimmed.is_empty() {
                    description_lines.push(trimmed);
                }
            }
            "affected files" | "files" | "paths" | "scope" => {
                if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
                    let path_str = trimmed[2..].trim().trim_matches('`');
                    ticket.affected_paths.push(PathBuf::from(path_str));
                }
            }
            "expected behavior" | "expected" => {
                if !trimmed.is_empty() {
                    ticket.expected_behavior = Some(trimmed.to_string());
                }
            }
            "acceptance criteria" | "criteria" => {
                if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
                    ticket.acceptance_criteria.push(trimmed[2..].to_string());
                }
            }
            "priority" => {
                ticket.priority = match trimmed.to_lowercase().as_str() {
                    "critical" => TicketPriority::Critical,
                    "high" => TicketPriority::High,
                    "medium" => TicketPriority::Medium,
                    "low" => TicketPriority::Low,
                    _ => TicketPriority::Medium,
                };
            }
            _ => {}
        }
    }

    ticket.description = description_lines.join(" ");

    Ok(ticket)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_markdown_ticket() {
        let content = r#"
# Fix Authentication Bug

## Description

Users cannot login when using special characters.

## Affected Files

- `src/auth/login.rs`
- `src/auth/token.rs`

## Expected Behavior

Login should work with any valid password.

## Acceptance Criteria

- [ ] Special characters handled correctly
- [ ] No regression in existing tests

## Priority

High
"#;

        let ticket = parse_markdown_ticket(content, Path::new("PMAT-123.md")).unwrap();
        assert_eq!(ticket.id, "PMAT-123");
        assert_eq!(ticket.title, "Fix Authentication Bug");
        assert_eq!(ticket.affected_paths.len(), 2);
        assert_eq!(ticket.priority, TicketPriority::High);
    }

    #[test]
    fn test_ticket_priority_default() {
        assert_eq!(TicketPriority::default(), TicketPriority::Medium);
    }

    #[test]
    fn test_target_paths_empty() {
        let ticket = PmatTicket {
            id: "TEST".to_string(),
            title: "Test".to_string(),
            description: String::new(),
            affected_paths: Vec::new(),
            expected_behavior: None,
            acceptance_criteria: Vec::new(),
            priority: TicketPriority::Medium,
        };
        let paths = ticket.target_paths();
        assert_eq!(paths, vec![PathBuf::from("src")]);
    }
}
