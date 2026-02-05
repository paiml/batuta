//! Git Blame Integration
//!
//! Provides git blame information for findings to help identify who introduced
//! bugs and when, enabling better triage and assignment.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// Git blame information for a single line.
#[derive(Debug, Clone, Default)]
pub struct BlameInfo {
    /// Author name
    pub author: String,
    /// Abbreviated commit hash
    pub commit: String,
    /// Date of the commit (YYYY-MM-DD format)
    pub date: String,
}

/// Cache for blame lookups to avoid repeated git calls.
#[derive(Debug, Default)]
pub struct BlameCache {
    /// Cache: (file, line) -> BlameInfo
    cache: HashMap<(String, usize), BlameInfo>,
}

impl BlameCache {
    /// Create a new empty blame cache.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Get blame info for a specific file and line, using cache.
    pub fn get_blame(&mut self, project_path: &Path, file: &Path, line: usize) -> Option<BlameInfo> {
        let file_str = file.to_string_lossy().to_string();
        let key = (file_str.clone(), line);

        if let Some(cached) = self.cache.get(&key) {
            return Some(cached.clone());
        }

        // Not in cache, fetch from git
        let blame = get_blame_for_line(project_path, file, line)?;
        self.cache.insert(key, blame.clone());
        Some(blame)
    }

    /// Prefetch blame info for multiple lines in a file (batch optimization).
    #[allow(dead_code)]
    pub fn prefetch_file(&mut self, project_path: &Path, file: &Path, lines: &[usize]) {
        if lines.is_empty() {
            return;
        }

        let file_str = file.to_string_lossy().to_string();

        // Check which lines we don't have cached
        let uncached: Vec<usize> = lines
            .iter()
            .filter(|&&l| !self.cache.contains_key(&(file_str.clone(), l)))
            .copied()
            .collect();

        if uncached.is_empty() {
            return;
        }

        // Batch fetch all blame info for the file
        if let Some(all_blames) = get_blame_for_file(project_path, file) {
            for (line, blame) in all_blames {
                self.cache.insert((file_str.clone(), line), blame);
            }
        }
    }
}

/// Get blame info for a specific line using git blame.
fn get_blame_for_line(project_path: &Path, file: &Path, line: usize) -> Option<BlameInfo> {
    let output = Command::new("git")
        .current_dir(project_path)
        .args([
            "blame",
            "-L",
            &format!("{},{}", line, line),
            "--porcelain",
            &file.to_string_lossy(),
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    parse_porcelain_blame(&String::from_utf8_lossy(&output.stdout))
}

/// Get blame info for an entire file (more efficient for multiple lines).
fn get_blame_for_file(project_path: &Path, file: &Path) -> Option<HashMap<usize, BlameInfo>> {
    let output = Command::new("git")
        .current_dir(project_path)
        .args(["blame", "--porcelain", &file.to_string_lossy()])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    Some(parse_porcelain_blame_full(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

/// Parse porcelain blame output for a single line.
fn parse_porcelain_blame(output: &str) -> Option<BlameInfo> {
    let mut author = String::new();
    let mut commit = String::new();
    let mut date = String::new();

    for line in output.lines() {
        if line.starts_with("author ") {
            author = line.strip_prefix("author ").unwrap_or("").to_string();
        } else if line.starts_with("author-time ") {
            // Convert Unix timestamp to YYYY-MM-DD
            if let Ok(timestamp) = line
                .strip_prefix("author-time ")
                .unwrap_or("0")
                .parse::<i64>()
            {
                date = format_timestamp(timestamp);
            }
        } else if commit.is_empty() && line.len() >= 40 {
            // Commit lines start with 40-char hex hash: <hash> <orig_line> <final_line> [count]
            let first_40: String = line.chars().take(40).collect();
            if first_40.chars().all(|c| c.is_ascii_hexdigit()) {
                commit = line[..7.min(line.len())].to_string();
            }
        }
    }

    if author.is_empty() && commit.is_empty() {
        return None;
    }

    Some(BlameInfo {
        author,
        commit,
        date,
    })
}

/// Parse porcelain blame output for full file.
fn parse_porcelain_blame_full(output: &str) -> HashMap<usize, BlameInfo> {
    let mut results = HashMap::new();
    let mut current_line = 0usize;
    let mut current_author = String::new();
    let mut current_commit = String::new();
    let mut current_date = String::new();

    for line in output.lines() {
        // Lines starting with a hash are commit lines: <hash> <orig_line> <final_line> [count]
        if line.len() >= 40 && line.chars().take(40).all(|c| c.is_ascii_hexdigit()) {
            // Save previous entry if we have one
            if current_line > 0 && !current_commit.is_empty() {
                results.insert(
                    current_line,
                    BlameInfo {
                        author: current_author.clone(),
                        commit: current_commit.clone(),
                        date: current_date.clone(),
                    },
                );
            }

            // Parse new commit line
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                current_commit = parts[0][..7.min(parts[0].len())].to_string();
                if let Ok(line_num) = parts[2].parse::<usize>() {
                    current_line = line_num;
                }
            }
            // Reset for new commit's metadata
            current_author.clear();
            current_date.clear();
        } else if line.starts_with("author ") {
            current_author = line.strip_prefix("author ").unwrap_or("").to_string();
        } else if line.starts_with("author-time ") {
            if let Ok(timestamp) = line
                .strip_prefix("author-time ")
                .unwrap_or("0")
                .parse::<i64>()
            {
                current_date = format_timestamp(timestamp);
            }
        }
    }

    // Don't forget the last entry
    if current_line > 0 && !current_commit.is_empty() {
        results.insert(
            current_line,
            BlameInfo {
                author: current_author,
                commit: current_commit,
                date: current_date,
            },
        );
    }

    results
}

/// Format Unix timestamp as YYYY-MM-DD.
fn format_timestamp(timestamp: i64) -> String {
    if timestamp < 0 {
        return String::new();
    }

    // Convert to date using chrono if available, otherwise use simple calculation
    let secs_per_day = 86400u64;
    let days_since_epoch = (timestamp as u64) / secs_per_day;

    // Simple date calculation (approximate, good enough for display)
    let mut year = 1970i32;
    let mut remaining_days = days_since_epoch as i32;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let mut month = 1;
    let days_in_months = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    for days in days_in_months.iter() {
        if remaining_days < *days {
            break;
        }
        remaining_days -= days;
        month += 1;
    }

    let day = remaining_days + 1;
    format!("{:04}-{:02}-{:02}", year, month, day)
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_porcelain_blame_basic() {
        let output = r#"abc1234567890123456789012345678901234567 1 1 1
author John Doe
author-mail <john@example.com>
author-time 1704067200
author-tz +0000
committer John Doe
committer-mail <john@example.com>
committer-time 1704067200
committer-tz +0000
summary Initial commit
filename src/main.rs
	fn main() {}
"#;

        let blame = parse_porcelain_blame(output).unwrap();
        assert_eq!(blame.author, "John Doe");
        assert_eq!(blame.commit, "abc1234");
        assert_eq!(blame.date, "2024-01-01");
    }

    #[test]
    fn test_parse_porcelain_blame_empty() {
        let output = "";
        assert!(parse_porcelain_blame(output).is_none());
    }

    #[test]
    fn test_format_timestamp() {
        // 2024-01-01 00:00:00 UTC
        assert_eq!(format_timestamp(1704067200), "2024-01-01");
        // 2000-06-15
        assert_eq!(format_timestamp(961027200), "2000-06-15");
        // 1970-01-01
        assert_eq!(format_timestamp(0), "1970-01-01");
    }

    #[test]
    fn test_is_leap_year() {
        assert!(is_leap_year(2000)); // Divisible by 400
        assert!(is_leap_year(2024)); // Divisible by 4, not by 100
        assert!(!is_leap_year(1900)); // Divisible by 100, not by 400
        assert!(!is_leap_year(2023)); // Not divisible by 4
    }

    #[test]
    fn test_blame_cache_new() {
        let cache = BlameCache::new();
        assert!(cache.cache.is_empty());
    }

    #[test]
    fn test_parse_porcelain_blame_full() {
        let output = r#"abc1234567890123456789012345678901234567 1 1 1
author Alice
author-time 1704067200
summary Line 1
filename test.rs
	line 1 content
def5678901234567890123456789012345678901 2 2 1
author Bob
author-time 1704153600
summary Line 2
filename test.rs
	line 2 content
"#;

        let blames = parse_porcelain_blame_full(output);
        assert_eq!(blames.len(), 2);
        assert_eq!(blames.get(&1).unwrap().author, "Alice");
        assert_eq!(blames.get(&2).unwrap().author, "Bob");
    }
}
