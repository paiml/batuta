use crate::types::*;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;

#[cfg(feature = "native")]
use tracing::{debug, info, warn};

#[cfg(feature = "native")]
use walkdir::WalkDir;

// Stub macros for WASM build
#[cfg(not(feature = "native"))]
macro_rules! info {
    ($($arg:tt)*) => {{}};
}

#[cfg(not(feature = "native"))]
macro_rules! debug {
    ($($arg:tt)*) => {{}};
}

#[cfg(not(feature = "native"))]
macro_rules! warn {
    ($($arg:tt)*) => {{}};
}

/// Analyze a project directory
pub fn analyze_project(
    path: &Path,
    include_tdg: bool,
    include_languages: bool,
    include_dependencies: bool,
) -> Result<ProjectAnalysis> {
    info!("Starting project analysis at {:?}", path);

    let mut analysis = ProjectAnalysis::new(path.to_path_buf());

    if include_languages {
        info!("Detecting languages...");
        let stats = detect_languages(path)?;
        analysis.languages = stats;

        // Determine primary language (most lines of code)
        if let Some(primary) = analysis.languages.first() {
            analysis.primary_language = Some(primary.language.clone());
        }

        // Calculate total files and lines
        analysis.total_files = analysis.languages.iter().map(|s| s.file_count).sum();
        analysis.total_lines = analysis.languages.iter().map(|s| s.line_count).sum();
    }

    if include_dependencies {
        info!("Analyzing dependencies...");
        analysis.dependencies = detect_dependencies(path)?;
    }

    if include_tdg {
        info!("Calculating TDG score...");
        analysis.tdg_score = calculate_tdg_score(path);
    }

    Ok(analysis)
}

/// Detect programming languages in the project
#[cfg(feature = "native")]
fn detect_languages(path: &Path) -> Result<Vec<LanguageStats>> {
    let mut language_stats: HashMap<Language, (usize, usize)> = HashMap::new();

    for entry in WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| !is_ignored(e.path()))
    {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }

        if let Some(lang) = detect_language_from_path(entry.path()) {
            let line_count = count_lines(entry.path()).unwrap_or(0);
            let stats = language_stats.entry(lang).or_insert((0, 0));
            stats.0 += 1; // file count
            stats.1 += line_count; // line count
        }
    }

    // Convert to LanguageStats and sort by line count (descending)
    let total_lines: usize = language_stats.values().map(|(_, lines)| lines).sum();
    let mut stats: Vec<LanguageStats> = language_stats
        .into_iter()
        .map(|(language, (file_count, line_count))| LanguageStats {
            language,
            file_count,
            line_count,
            percentage: if total_lines > 0 {
                (line_count as f64 / total_lines as f64) * 100.0
            } else {
                0.0
            },
        })
        .collect();

    stats.sort_by(|a, b| b.line_count.cmp(&a.line_count));

    Ok(stats)
}

/// Detect language from file extension
fn detect_language_from_path(path: &Path) -> Option<Language> {
    let extension = path.extension()?.to_str()?;

    match extension {
        "py" | "pyx" | "pyi" => Some(Language::Python),
        "c" | "h" => Some(Language::C),
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => Some(Language::Cpp),
        "rs" => Some(Language::Rust),
        "sh" | "bash" | "zsh" => Some(Language::Shell),
        "js" | "jsx" | "mjs" => Some(Language::JavaScript),
        "ts" | "tsx" => Some(Language::TypeScript),
        "go" => Some(Language::Go),
        "java" => Some(Language::Java),
        _ => None,
    }
}

/// Count non-empty lines in a file
fn count_lines(path: &Path) -> Result<usize> {
    let content = fs::read_to_string(path).context("Failed to read file")?;
    Ok(content.lines().filter(|line| !line.trim().is_empty()).count())
}

/// Check if path should be ignored (common directories to skip)
fn is_ignored(path: &Path) -> bool {
    let ignore_names = [
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        "target",
        "build",
        "dist",
        "__pycache__",
        ".pytest_cache",
        ".venv",
        "venv",
        ".idea",
        ".vscode",
    ];

    path.components().any(|c| {
        if let Some(name) = c.as_os_str().to_str() {
            ignore_names.contains(&name)
        } else {
            false
        }
    })
}

/// Detect dependency managers and files
fn detect_dependencies(path: &Path) -> Result<Vec<DependencyInfo>> {
    let mut deps = Vec::new();

    // Python
    if let Some(info) = check_dependency_file(path, "requirements.txt", DependencyManager::Pip) {
        deps.push(info);
    }
    if let Some(info) = check_dependency_file(path, "Pipfile", DependencyManager::Pipenv) {
        deps.push(info);
    }
    if let Some(info) = check_poetry_deps(path) {
        deps.push(info);
    }
    if let Some(info) = check_dependency_file(path, "environment.yml", DependencyManager::Conda) {
        deps.push(info);
    }

    // Rust
    if let Some(info) = check_dependency_file(path, "Cargo.toml", DependencyManager::Cargo) {
        deps.push(info);
    }

    // JavaScript/Node
    if let Some(info) = check_dependency_file(path, "package.json", DependencyManager::Npm) {
        deps.push(info);
    }
    if let Some(info) = check_dependency_file(path, "yarn.lock", DependencyManager::Yarn) {
        deps.push(info);
    }

    // Go
    if let Some(info) = check_dependency_file(path, "go.mod", DependencyManager::GoMod) {
        deps.push(info);
    }

    // Java
    if let Some(info) = check_dependency_file(path, "pom.xml", DependencyManager::Maven) {
        deps.push(info);
    }
    if let Some(info) = check_dependency_file(path, "build.gradle", DependencyManager::Gradle) {
        deps.push(info);
    }

    // C/C++
    if let Some(info) = check_dependency_file(path, "Makefile", DependencyManager::Make) {
        deps.push(info);
    }

    Ok(deps)
}

fn check_dependency_file(
    base_path: &Path,
    filename: &str,
    manager: DependencyManager,
) -> Option<DependencyInfo> {
    let file_path = base_path.join(filename);
    if file_path.exists() {
        debug!("Found dependency file: {:?}", file_path);
        let count = count_dependencies(&file_path, &manager);
        Some(DependencyInfo {
            manager,
            file_path,
            count,
        })
    } else {
        None
    }
}

fn check_poetry_deps(base_path: &Path) -> Option<DependencyInfo> {
    let file_path = base_path.join("pyproject.toml");
    if file_path.exists() {
        if let Ok(content) = fs::read_to_string(&file_path) {
            if content.contains("[tool.poetry]") {
                debug!("Found Poetry project: {:?}", file_path);
                let count = count_dependencies(&file_path, &DependencyManager::Poetry);
                return Some(DependencyInfo {
                    manager: DependencyManager::Poetry,
                    file_path,
                    count,
                });
            }
        }
    }
    None
}

fn count_dependencies(path: &Path, manager: &DependencyManager) -> Option<usize> {
    let content = fs::read_to_string(path).ok()?;

    match manager {
        DependencyManager::Pip => {
            // Count non-comment, non-empty lines in requirements.txt
            Some(
                content
                    .lines()
                    .filter(|line| {
                        let trimmed = line.trim();
                        !trimmed.is_empty() && !trimmed.starts_with('#')
                    })
                    .count(),
            )
        }
        DependencyManager::Cargo => {
            // Count [dependencies] section entries
            let lines: Vec<&str> = content.lines().collect();
            let mut in_deps = false;
            let mut count = 0;

            for line in lines {
                let trimmed = line.trim();
                if trimmed == "[dependencies]" || trimmed == "[dev-dependencies]" {
                    in_deps = true;
                } else if trimmed.starts_with('[') {
                    in_deps = false;
                } else if in_deps && !trimmed.is_empty() && !trimmed.starts_with('#') {
                    count += 1;
                }
            }
            Some(count)
        }
        DependencyManager::Npm => {
            // Parse package.json for dependencies count
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                let deps = json.get("dependencies").and_then(|d| d.as_object());
                let dev_deps = json.get("devDependencies").and_then(|d| d.as_object());
                Some(deps.map(|d| d.len()).unwrap_or(0) + dev_deps.map(|d| d.len()).unwrap_or(0))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Calculate TDG score using PMAT
fn calculate_tdg_score(path: &Path) -> Option<f64> {
    debug!("Running PMAT TDG analysis...");

    let output = Command::new("pmat")
        .arg("tdg")
        .arg(path)
        .output()
        .ok()?;

    if !output.status.success() {
        warn!("PMAT TDG command failed");
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse output for score line: "Overall Score: 100.0/100 (A+)"
    for line in stdout.lines() {
        if line.contains("Overall Score:") {
            if let Some(score_str) = line.split(':').nth(1) {
                // Extract "100.0/100" part
                if let Some(score) = score_str.trim().split('/').next() {
                    if let Ok(score_val) = score.trim().parse::<f64>() {
                        return Some(score_val);
                    }
                }
            }
        }
    }

    None
}
