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
#[allow(clippy::cognitive_complexity)]
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
    Ok(content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count())
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

fn count_pip_dependencies(content: &str) -> usize {
    content
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .count()
}

fn count_cargo_dependencies(content: &str) -> usize {
    let mut in_deps = false;
    let mut count = 0;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[dependencies]" || trimmed == "[dev-dependencies]" {
            in_deps = true;
        } else if trimmed.starts_with('[') {
            in_deps = false;
        } else if in_deps && !trimmed.is_empty() && !trimmed.starts_with('#') {
            count += 1;
        }
    }
    count
}

fn count_npm_dependencies(content: &str) -> Option<usize> {
    let json: serde_json::Value = serde_json::from_str(content).ok()?;
    let deps = json.get("dependencies").and_then(|d| d.as_object());
    let dev_deps = json.get("devDependencies").and_then(|d| d.as_object());
    Some(deps.map(|d| d.len()).unwrap_or(0) + dev_deps.map(|d| d.len()).unwrap_or(0))
}

fn count_dependencies(path: &Path, manager: &DependencyManager) -> Option<usize> {
    let content = fs::read_to_string(path).ok()?;

    match manager {
        DependencyManager::Pip => Some(count_pip_dependencies(&content)),
        DependencyManager::Cargo => Some(count_cargo_dependencies(&content)),
        DependencyManager::Npm => count_npm_dependencies(&content),
        _ => None,
    }
}

/// Calculate TDG score using PMAT, with fallback for when PMAT is unavailable
fn calculate_tdg_score(path: &Path) -> Option<f64> {
    debug!("Running PMAT TDG analysis...");

    // Try to use PMAT first
    if let Some(score) = calculate_tdg_with_pmat(path) {
        return Some(score);
    }

    // Fallback: basic heuristic TDG score when PMAT unavailable
    debug!("PMAT unavailable, using fallback TDG calculation");
    calculate_tdg_fallback(path)
}

/// Parse PMAT output for score line: "Overall Score: 100.0/100 (A+)"
fn parse_pmat_score_line(line: &str) -> Option<f64> {
    if !line.contains("Overall Score:") {
        return None;
    }
    let score_str = line.split(':').nth(1)?;
    let score = score_str.trim().split('/').next()?;
    score.trim().parse::<f64>().ok()
}

/// Calculate TDG using external PMAT tool
fn calculate_tdg_with_pmat(path: &Path) -> Option<f64> {
    let output = Command::new("pmat").arg("tdg").arg(path).output().ok()?;

    if !output.status.success() {
        warn!("PMAT TDG command failed");
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().find_map(parse_pmat_score_line)
}

/// Fallback TDG calculation using basic heuristics
/// This provides a reasonable estimate when PMAT is not available
fn calculate_tdg_fallback(path: &Path) -> Option<f64> {
    let mut score: f64 = 100.0;

    // Check for tests directory or #[test] presence
    let has_tests = path.join("tests").exists()
        || WalkDir::new(path)
            .max_depth(3)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
            .take(10)
            .any(|e| {
                std::fs::read_to_string(e.path())
                    .ok()
                    .is_some_and(|content| content.contains("#[test]"))
            });

    if !has_tests {
        score -= 10.0; // Deduct for no tests
    }

    // Check for README
    if !path.join("README.md").exists() && !path.join("README").exists() {
        score -= 5.0;
    }

    // Check for CI configuration
    let has_ci = path.join(".github/workflows").exists()
        || path.join(".gitlab-ci.yml").exists()
        || path.join(".circleci").exists();
    if !has_ci {
        score -= 5.0;
    }

    // Check for license
    let has_license = path.join("LICENSE").exists()
        || path.join("LICENSE.md").exists()
        || path.join("LICENSE.txt").exists();
    if !has_license {
        score -= 5.0;
    }

    Some(score.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // ============================================================================
    // LANGUAGE DETECTION TESTS
    // ============================================================================

    #[test]
    fn test_detect_language_from_path_python() {
        assert_eq!(
            detect_language_from_path(&PathBuf::from("test.py")),
            Some(Language::Python)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("module.pyx")),
            Some(Language::Python)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("types.pyi")),
            Some(Language::Python)
        );
    }

    #[test]
    fn test_detect_language_from_path_c() {
        assert_eq!(
            detect_language_from_path(&PathBuf::from("main.c")),
            Some(Language::C)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("header.h")),
            Some(Language::C)
        );
    }

    #[test]
    fn test_detect_language_from_path_cpp() {
        assert_eq!(
            detect_language_from_path(&PathBuf::from("main.cpp")),
            Some(Language::Cpp)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("main.cc")),
            Some(Language::Cpp)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("main.cxx")),
            Some(Language::Cpp)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("header.hpp")),
            Some(Language::Cpp)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("header.hxx")),
            Some(Language::Cpp)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("header.hh")),
            Some(Language::Cpp)
        );
    }

    #[test]
    fn test_detect_language_from_path_rust() {
        assert_eq!(
            detect_language_from_path(&PathBuf::from("main.rs")),
            Some(Language::Rust)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("lib.rs")),
            Some(Language::Rust)
        );
    }

    #[test]
    fn test_detect_language_from_path_shell() {
        assert_eq!(
            detect_language_from_path(&PathBuf::from("script.sh")),
            Some(Language::Shell)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("script.bash")),
            Some(Language::Shell)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("script.zsh")),
            Some(Language::Shell)
        );
    }

    #[test]
    fn test_detect_language_from_path_javascript() {
        assert_eq!(
            detect_language_from_path(&PathBuf::from("app.js")),
            Some(Language::JavaScript)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("component.jsx")),
            Some(Language::JavaScript)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("module.mjs")),
            Some(Language::JavaScript)
        );
    }

    #[test]
    fn test_detect_language_from_path_typescript() {
        assert_eq!(
            detect_language_from_path(&PathBuf::from("app.ts")),
            Some(Language::TypeScript)
        );
        assert_eq!(
            detect_language_from_path(&PathBuf::from("component.tsx")),
            Some(Language::TypeScript)
        );
    }

    #[test]
    fn test_detect_language_from_path_go() {
        assert_eq!(
            detect_language_from_path(&PathBuf::from("main.go")),
            Some(Language::Go)
        );
    }

    #[test]
    fn test_detect_language_from_path_java() {
        assert_eq!(
            detect_language_from_path(&PathBuf::from("Main.java")),
            Some(Language::Java)
        );
    }

    #[test]
    fn test_detect_language_from_path_unknown() {
        assert_eq!(detect_language_from_path(&PathBuf::from("file.txt")), None);
        assert_eq!(detect_language_from_path(&PathBuf::from("README.md")), None);
        assert_eq!(detect_language_from_path(&PathBuf::from("Makefile")), None);
        assert_eq!(
            detect_language_from_path(&PathBuf::from("no_extension")),
            None
        );
    }

    // ============================================================================
    // LINE COUNTING TESTS
    // ============================================================================

    #[test]
    fn test_count_lines_simple() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        fs::write(&file_path, "line1\nline2\nline3").unwrap();

        assert_eq!(count_lines(&file_path).unwrap(), 3);
    }

    #[test]
    fn test_count_lines_with_empty_lines() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        fs::write(&file_path, "line1\n\nline2\n  \nline3").unwrap();

        // Should only count non-empty lines (3 lines)
        assert_eq!(count_lines(&file_path).unwrap(), 3);
    }

    #[test]
    fn test_count_lines_empty_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("empty.txt");

        fs::write(&file_path, "").unwrap();

        assert_eq!(count_lines(&file_path).unwrap(), 0);
    }

    #[test]
    fn test_count_lines_whitespace_only() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("whitespace.txt");

        fs::write(&file_path, "   \n\t\n  \t  \n").unwrap();

        assert_eq!(count_lines(&file_path).unwrap(), 0);
    }

    #[test]
    fn test_count_lines_nonexistent_file() {
        let result = count_lines(&PathBuf::from("/nonexistent/file.txt"));
        assert!(result.is_err());
    }

    // ============================================================================
    // PATH IGNORE TESTS
    // ============================================================================

    #[test]
    fn test_is_ignored_git() {
        assert!(is_ignored(&PathBuf::from("/project/.git/config")));
        assert!(is_ignored(&PathBuf::from("/project/src/.git/HEAD")));
    }

    #[test]
    fn test_is_ignored_node_modules() {
        assert!(is_ignored(&PathBuf::from("/project/node_modules/package")));
        assert!(is_ignored(&PathBuf::from("node_modules/lib/index.js")));
    }

    #[test]
    fn test_is_ignored_target() {
        assert!(is_ignored(&PathBuf::from("/project/target/debug")));
        assert!(is_ignored(&PathBuf::from("target/release/app")));
    }

    #[test]
    fn test_is_ignored_pycache() {
        assert!(is_ignored(&PathBuf::from(
            "/project/__pycache__/module.pyc"
        )));
        assert!(is_ignored(&PathBuf::from(
            "src/__pycache__/test.cpython-39.pyc"
        )));
    }

    #[test]
    fn test_is_ignored_venv() {
        assert!(is_ignored(&PathBuf::from("/project/venv/lib")));
        assert!(is_ignored(&PathBuf::from("/project/.venv/bin/python")));
    }

    #[test]
    fn test_is_ignored_ide_folders() {
        assert!(is_ignored(&PathBuf::from("/project/.idea/workspace.xml")));
        assert!(is_ignored(&PathBuf::from("/project/.vscode/settings.json")));
    }

    #[test]
    fn test_is_not_ignored() {
        assert!(!is_ignored(&PathBuf::from("/project/src/main.rs")));
        assert!(!is_ignored(&PathBuf::from("/project/README.md")));
        assert!(!is_ignored(&PathBuf::from("Cargo.toml")));
    }

    // ============================================================================
    // DEPENDENCY DETECTION TESTS
    // ============================================================================

    #[test]
    fn test_detect_dependencies_python_requirements() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("requirements.txt"), "numpy\npandas\n").unwrap();

        let deps = detect_dependencies(temp_dir.path()).unwrap();

        assert_eq!(deps.len(), 1);
        assert!(matches!(deps[0].manager, DependencyManager::Pip));
        assert_eq!(deps[0].count, Some(2));
    }

    #[test]
    fn test_detect_dependencies_python_pipfile() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("Pipfile"), "[packages]\n").unwrap();

        let deps = detect_dependencies(temp_dir.path()).unwrap();

        assert_eq!(deps.len(), 1);
        assert!(matches!(deps[0].manager, DependencyManager::Pipenv));
    }

    #[test]
    fn test_detect_dependencies_python_poetry() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(
            temp_dir.path().join("pyproject.toml"),
            "[tool.poetry]\nname = \"test\"\n",
        )
        .unwrap();

        let deps = detect_dependencies(temp_dir.path()).unwrap();

        assert_eq!(deps.len(), 1);
        assert!(matches!(deps[0].manager, DependencyManager::Poetry));
    }

    #[test]
    fn test_detect_dependencies_rust_cargo() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(
            temp_dir.path().join("Cargo.toml"),
            "[package]\nname = \"test\"\n[dependencies]\nserde = \"1.0\"\n",
        )
        .unwrap();

        let deps = detect_dependencies(temp_dir.path()).unwrap();

        assert_eq!(deps.len(), 1);
        assert!(matches!(deps[0].manager, DependencyManager::Cargo));
        assert_eq!(deps[0].count, Some(1));
    }

    #[test]
    fn test_detect_dependencies_javascript_npm() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(
            temp_dir.path().join("package.json"),
            r#"{"dependencies": {"react": "^18.0.0", "lodash": "^4.0.0"}}"#,
        )
        .unwrap();

        let deps = detect_dependencies(temp_dir.path()).unwrap();

        assert_eq!(deps.len(), 1);
        assert!(matches!(deps[0].manager, DependencyManager::Npm));
        assert_eq!(deps[0].count, Some(2));
    }

    #[test]
    fn test_detect_dependencies_go_mod() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("go.mod"), "module test\n").unwrap();

        let deps = detect_dependencies(temp_dir.path()).unwrap();

        assert_eq!(deps.len(), 1);
        assert!(matches!(deps[0].manager, DependencyManager::GoMod));
    }

    #[test]
    fn test_detect_dependencies_multiple() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("requirements.txt"), "numpy\n").unwrap();
        fs::write(
            temp_dir.path().join("package.json"),
            r#"{"dependencies": {}}"#,
        )
        .unwrap();

        let deps = detect_dependencies(temp_dir.path()).unwrap();

        assert_eq!(deps.len(), 2);
    }

    #[test]
    fn test_detect_dependencies_none() {
        let temp_dir = TempDir::new().unwrap();

        let deps = detect_dependencies(temp_dir.path()).unwrap();

        assert_eq!(deps.len(), 0);
    }

    // ============================================================================
    // DEPENDENCY COUNTING TESTS
    // ============================================================================

    #[test]
    fn test_count_dependencies_pip() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("requirements.txt");

        fs::write(
            &file_path,
            "numpy>=1.20.0\npandas\n# comment\nscikit-learn\n\n",
        )
        .unwrap();

        let count = count_dependencies(&file_path, &DependencyManager::Pip);

        assert_eq!(count, Some(3)); // Excludes comment and empty line
    }

    #[test]
    fn test_count_dependencies_cargo() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("Cargo.toml");

        fs::write(
            &file_path,
            r#"[package]
name = "test"

[dependencies]
serde = "1.0"
tokio = "1.0"

[dev-dependencies]
criterion = "0.5"
"#,
        )
        .unwrap();

        let count = count_dependencies(&file_path, &DependencyManager::Cargo);

        assert_eq!(count, Some(3)); // 2 dependencies + 1 dev-dependency
    }

    #[test]
    fn test_count_dependencies_npm_with_dev() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("package.json");

        fs::write(
            &file_path,
            r#"{
  "dependencies": {
    "react": "^18.0.0",
    "lodash": "^4.0.0"
  },
  "devDependencies": {
    "jest": "^29.0.0"
  }
}"#,
        )
        .unwrap();

        let count = count_dependencies(&file_path, &DependencyManager::Npm);

        assert_eq!(count, Some(3)); // 2 regular + 1 dev
    }

    #[test]
    fn test_count_dependencies_npm_no_dev() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("package.json");

        fs::write(&file_path, r#"{"dependencies": {"react": "^18.0.0"}}"#).unwrap();

        let count = count_dependencies(&file_path, &DependencyManager::Npm);

        assert_eq!(count, Some(1));
    }

    #[test]
    fn test_count_dependencies_unsupported_manager() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("go.mod");

        fs::write(&file_path, "module test\n").unwrap();

        let count = count_dependencies(&file_path, &DependencyManager::GoMod);

        assert_eq!(count, None); // GoMod counting not implemented
    }

    // ============================================================================
    // ANALYZE_PROJECT INTEGRATION TESTS
    // ============================================================================

    #[test]
    #[cfg(feature = "native")]
    fn test_analyze_project_basic() {
        let temp_dir = TempDir::new().unwrap();

        // Create test files
        fs::write(temp_dir.path().join("main.rs"), "fn main() {}\n").unwrap();
        fs::write(temp_dir.path().join("lib.rs"), "pub fn test() {}\n").unwrap();

        let analysis = analyze_project(temp_dir.path(), false, true, false).unwrap();

        assert_eq!(analysis.total_files, 2);
        assert!(analysis.total_lines > 0);
        assert_eq!(analysis.primary_language, Some(Language::Rust));
        assert_eq!(analysis.languages.len(), 1);
        assert!(matches!(analysis.languages[0].language, Language::Rust));
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_analyze_project_with_dependencies() {
        let temp_dir = TempDir::new().unwrap();

        fs::write(temp_dir.path().join("main.py"), "print('hello')\n").unwrap();
        fs::write(temp_dir.path().join("requirements.txt"), "numpy\npandas\n").unwrap();

        let analysis = analyze_project(temp_dir.path(), false, true, true).unwrap();

        assert!(!analysis.dependencies.is_empty());
        assert!(matches!(
            analysis.dependencies[0].manager,
            DependencyManager::Pip
        ));
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_analyze_project_mixed_languages() {
        let temp_dir = TempDir::new().unwrap();

        fs::write(
            temp_dir.path().join("main.py"),
            "# Python\nprint('hello')\n",
        )
        .unwrap();
        fs::write(temp_dir.path().join("util.rs"), "// Rust\nfn main() {}\n").unwrap();
        fs::write(
            temp_dir.path().join("script.sh"),
            "#!/bin/bash\necho test\n",
        )
        .unwrap();

        let analysis = analyze_project(temp_dir.path(), false, true, false).unwrap();

        assert_eq!(analysis.languages.len(), 3);
        assert!(analysis.total_files >= 3);
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_analyze_project_ignores_directories() {
        let temp_dir = TempDir::new().unwrap();

        // Create source file
        fs::write(temp_dir.path().join("main.rs"), "fn main() {}\n").unwrap();

        // Create ignored directory with file
        fs::create_dir(temp_dir.path().join("node_modules")).unwrap();
        fs::write(
            temp_dir.path().join("node_modules/test.js"),
            "console.log('test');\n",
        )
        .unwrap();

        let analysis = analyze_project(temp_dir.path(), false, true, false).unwrap();

        // Should only count main.rs, not node_modules/test.js
        assert_eq!(analysis.total_files, 1);
        assert_eq!(analysis.primary_language, Some(Language::Rust));
    }

    #[test]
    fn test_analyze_project_empty_directory() {
        let temp_dir = TempDir::new().unwrap();

        let result = analyze_project(temp_dir.path(), false, true, false);

        // Should succeed but have no languages detected
        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert_eq!(analysis.total_files, 0);
        assert_eq!(analysis.total_lines, 0);
        assert!(analysis.primary_language.is_none());
    }

    // ============================================================================
    // TDG FALLBACK TESTS (Coverage for calculate_tdg_fallback)
    // ============================================================================

    #[test]
    #[cfg(feature = "native")]
    fn test_tdg_cov_001_fallback_no_tests() {
        let temp_dir = TempDir::new().unwrap();
        // Project with no tests directory and no #[test] attributes
        fs::write(temp_dir.path().join("main.rs"), "fn main() {}\n").unwrap();

        let score = calculate_tdg_fallback(temp_dir.path());
        // Score should be reduced for no tests
        assert!(score.is_some());
        assert!(score.unwrap() < 100.0);
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_tdg_cov_002_fallback_with_tests_dir() {
        let temp_dir = TempDir::new().unwrap();
        // Create tests directory
        fs::create_dir(temp_dir.path().join("tests")).unwrap();
        fs::write(
            temp_dir.path().join("tests/test.rs"),
            "#[test]\nfn test() {}\n",
        )
        .unwrap();
        // Add README
        fs::write(temp_dir.path().join("README.md"), "# Test\n").unwrap();
        // Add CI
        fs::create_dir_all(temp_dir.path().join(".github/workflows")).unwrap();
        // Add LICENSE
        fs::write(temp_dir.path().join("LICENSE"), "MIT\n").unwrap();

        let score = calculate_tdg_fallback(temp_dir.path());
        assert!(score.is_some());
        // Should have full score with all components
        assert_eq!(score.unwrap(), 100.0);
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_tdg_cov_003_fallback_with_test_attribute() {
        let temp_dir = TempDir::new().unwrap();
        // File with #[test] attribute but no tests directory
        fs::write(
            temp_dir.path().join("lib.rs"),
            "fn add(a: i32, b: i32) -> i32 { a + b }\n#[test]\nfn test_add() { assert_eq!(add(1, 2), 3); }\n",
        )
        .unwrap();
        fs::write(temp_dir.path().join("README.md"), "# Test\n").unwrap();
        fs::create_dir_all(temp_dir.path().join(".github/workflows")).unwrap();
        fs::write(temp_dir.path().join("LICENSE"), "MIT\n").unwrap();

        let score = calculate_tdg_fallback(temp_dir.path());
        assert!(score.is_some());
        assert_eq!(score.unwrap(), 100.0);
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_tdg_cov_004_fallback_no_readme() {
        let temp_dir = TempDir::new().unwrap();
        fs::create_dir(temp_dir.path().join("tests")).unwrap();
        fs::create_dir_all(temp_dir.path().join(".github/workflows")).unwrap();
        fs::write(temp_dir.path().join("LICENSE"), "MIT\n").unwrap();
        // No README

        let score = calculate_tdg_fallback(temp_dir.path());
        assert!(score.is_some());
        // Should have 95.0 (100 - 5 for no README)
        assert_eq!(score.unwrap(), 95.0);
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_tdg_cov_005_fallback_no_ci() {
        let temp_dir = TempDir::new().unwrap();
        fs::create_dir(temp_dir.path().join("tests")).unwrap();
        fs::write(temp_dir.path().join("README.md"), "# Test\n").unwrap();
        fs::write(temp_dir.path().join("LICENSE"), "MIT\n").unwrap();
        // No CI

        let score = calculate_tdg_fallback(temp_dir.path());
        assert!(score.is_some());
        // Should have 95.0 (100 - 5 for no CI)
        assert_eq!(score.unwrap(), 95.0);
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_tdg_cov_006_fallback_no_license() {
        let temp_dir = TempDir::new().unwrap();
        fs::create_dir(temp_dir.path().join("tests")).unwrap();
        fs::write(temp_dir.path().join("README.md"), "# Test\n").unwrap();
        fs::create_dir_all(temp_dir.path().join(".github/workflows")).unwrap();
        // No LICENSE

        let score = calculate_tdg_fallback(temp_dir.path());
        assert!(score.is_some());
        // Should have 95.0 (100 - 5 for no LICENSE)
        assert_eq!(score.unwrap(), 95.0);
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_tdg_cov_007_fallback_gitlab_ci() {
        let temp_dir = TempDir::new().unwrap();
        fs::create_dir(temp_dir.path().join("tests")).unwrap();
        fs::write(temp_dir.path().join("README.md"), "# Test\n").unwrap();
        fs::write(temp_dir.path().join(".gitlab-ci.yml"), "stages:\n").unwrap();
        fs::write(temp_dir.path().join("LICENSE"), "MIT\n").unwrap();

        let score = calculate_tdg_fallback(temp_dir.path());
        assert!(score.is_some());
        assert_eq!(score.unwrap(), 100.0);
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_tdg_cov_008_fallback_circleci() {
        let temp_dir = TempDir::new().unwrap();
        fs::create_dir(temp_dir.path().join("tests")).unwrap();
        fs::write(temp_dir.path().join("README.md"), "# Test\n").unwrap();
        fs::create_dir_all(temp_dir.path().join(".circleci")).unwrap();
        fs::write(temp_dir.path().join("LICENSE"), "MIT\n").unwrap();

        let score = calculate_tdg_fallback(temp_dir.path());
        assert!(score.is_some());
        assert_eq!(score.unwrap(), 100.0);
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_tdg_cov_009_fallback_license_md() {
        let temp_dir = TempDir::new().unwrap();
        fs::create_dir(temp_dir.path().join("tests")).unwrap();
        fs::write(temp_dir.path().join("README.md"), "# Test\n").unwrap();
        fs::create_dir_all(temp_dir.path().join(".github/workflows")).unwrap();
        fs::write(temp_dir.path().join("LICENSE.md"), "# MIT License\n").unwrap();

        let score = calculate_tdg_fallback(temp_dir.path());
        assert!(score.is_some());
        assert_eq!(score.unwrap(), 100.0);
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_tdg_cov_010_fallback_license_txt() {
        let temp_dir = TempDir::new().unwrap();
        fs::create_dir(temp_dir.path().join("tests")).unwrap();
        fs::write(temp_dir.path().join("README.md"), "# Test\n").unwrap();
        fs::create_dir_all(temp_dir.path().join(".github/workflows")).unwrap();
        fs::write(temp_dir.path().join("LICENSE.txt"), "MIT License\n").unwrap();

        let score = calculate_tdg_fallback(temp_dir.path());
        assert!(score.is_some());
        assert_eq!(score.unwrap(), 100.0);
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_tdg_cov_011_fallback_readme_no_ext() {
        let temp_dir = TempDir::new().unwrap();
        fs::create_dir(temp_dir.path().join("tests")).unwrap();
        fs::write(temp_dir.path().join("README"), "Test\n").unwrap();
        fs::create_dir_all(temp_dir.path().join(".github/workflows")).unwrap();
        fs::write(temp_dir.path().join("LICENSE"), "MIT\n").unwrap();

        let score = calculate_tdg_fallback(temp_dir.path());
        assert!(score.is_some());
        assert_eq!(score.unwrap(), 100.0);
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_tdg_cov_012_fallback_all_missing() {
        let temp_dir = TempDir::new().unwrap();
        // Empty directory - nothing present
        fs::write(temp_dir.path().join("main.txt"), "not code\n").unwrap();

        let score = calculate_tdg_fallback(temp_dir.path());
        assert!(score.is_some());
        // 100 - 10 (no tests) - 5 (no README) - 5 (no CI) - 5 (no LICENSE) = 75
        assert_eq!(score.unwrap(), 75.0);
    }

    // ============================================================================
    // POETRY DEPENDENCY DETECTION TESTS
    // ============================================================================

    #[test]
    fn test_poetry_cov_001_pyproject_without_poetry() {
        let temp_dir = TempDir::new().unwrap();
        // pyproject.toml without [tool.poetry]
        fs::write(
            temp_dir.path().join("pyproject.toml"),
            "[project]\nname = \"test\"\n",
        )
        .unwrap();

        let result = check_poetry_deps(temp_dir.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_poetry_cov_002_pyproject_with_poetry() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(
            temp_dir.path().join("pyproject.toml"),
            "[tool.poetry]\nname = \"test\"\n[tool.poetry.dependencies]\npython = \"^3.9\"\n",
        )
        .unwrap();

        let result = check_poetry_deps(temp_dir.path());
        assert!(result.is_some());
        let info = result.unwrap();
        assert!(matches!(info.manager, DependencyManager::Poetry));
    }

    #[test]
    fn test_poetry_cov_003_no_pyproject() {
        let temp_dir = TempDir::new().unwrap();
        // No pyproject.toml at all

        let result = check_poetry_deps(temp_dir.path());
        assert!(result.is_none());
    }

    // ============================================================================
    // ADDITIONAL DEPENDENCY MANAGER TESTS
    // ============================================================================

    #[test]
    fn test_dep_cov_001_environment_yml() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("environment.yml"), "name: env\n").unwrap();

        let deps = detect_dependencies(temp_dir.path()).unwrap();
        assert_eq!(deps.len(), 1);
        assert!(matches!(deps[0].manager, DependencyManager::Conda));
    }

    #[test]
    fn test_dep_cov_002_yarn_lock() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("yarn.lock"), "# yarn lockfile\n").unwrap();

        let deps = detect_dependencies(temp_dir.path()).unwrap();
        assert_eq!(deps.len(), 1);
        assert!(matches!(deps[0].manager, DependencyManager::Yarn));
    }

    #[test]
    fn test_dep_cov_003_maven_pom() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("pom.xml"), "<project></project>\n").unwrap();

        let deps = detect_dependencies(temp_dir.path()).unwrap();
        assert_eq!(deps.len(), 1);
        assert!(matches!(deps[0].manager, DependencyManager::Maven));
    }

    #[test]
    fn test_dep_cov_004_gradle_build() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("build.gradle"), "plugins {}\n").unwrap();

        let deps = detect_dependencies(temp_dir.path()).unwrap();
        assert_eq!(deps.len(), 1);
        assert!(matches!(deps[0].manager, DependencyManager::Gradle));
    }

    #[test]
    fn test_dep_cov_005_makefile() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("Makefile"), "all:\n\techo test\n").unwrap();

        let deps = detect_dependencies(temp_dir.path()).unwrap();
        assert_eq!(deps.len(), 1);
        assert!(matches!(deps[0].manager, DependencyManager::Make));
    }

    // ============================================================================
    // COUNT DEPENDENCIES EDGE CASES
    // ============================================================================

    #[test]
    fn test_count_cov_001_cargo_with_comments() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("Cargo.toml");

        fs::write(
            &file_path,
            r#"[package]
name = "test"

[dependencies]
# This is a comment
serde = "1.0"  # inline comment doesn't affect count

[dev-dependencies]
# Another comment
tokio = "1.0"
"#,
        )
        .unwrap();

        let count = count_dependencies(&file_path, &DependencyManager::Cargo);
        assert_eq!(count, Some(2)); // serde + tokio (comments excluded)
    }

    #[test]
    fn test_count_cov_002_npm_invalid_json() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("package.json");

        fs::write(&file_path, "{ invalid json }").unwrap();

        let count = count_dependencies(&file_path, &DependencyManager::Npm);
        assert_eq!(count, None);
    }

    #[test]
    fn test_count_cov_003_npm_no_dependencies() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("package.json");

        fs::write(&file_path, r#"{"name": "test", "version": "1.0.0"}"#).unwrap();

        let count = count_dependencies(&file_path, &DependencyManager::Npm);
        assert_eq!(count, Some(0));
    }

    #[test]
    fn test_count_cov_004_pip_all_comments() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("requirements.txt");

        fs::write(&file_path, "# comment 1\n# comment 2\n\n").unwrap();

        let count = count_dependencies(&file_path, &DependencyManager::Pip);
        assert_eq!(count, Some(0));
    }

    #[test]
    fn test_count_cov_005_nonexistent_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("nonexistent.txt");

        let count = count_dependencies(&file_path, &DependencyManager::Pip);
        assert_eq!(count, None);
    }

    // ============================================================================
    // IS_IGNORED ADDITIONAL TESTS
    // ============================================================================

    #[test]
    fn test_ignore_cov_001_svn() {
        assert!(is_ignored(&PathBuf::from("/project/.svn/entries")));
    }

    #[test]
    fn test_ignore_cov_002_hg() {
        assert!(is_ignored(&PathBuf::from("/project/.hg/store")));
    }

    #[test]
    fn test_ignore_cov_003_build() {
        assert!(is_ignored(&PathBuf::from("/project/build/output")));
    }

    #[test]
    fn test_ignore_cov_004_dist() {
        assert!(is_ignored(&PathBuf::from("/project/dist/bundle.js")));
    }

    #[test]
    fn test_ignore_cov_005_pytest_cache() {
        assert!(is_ignored(&PathBuf::from("/project/.pytest_cache/v")));
    }

    #[test]
    fn test_ignore_cov_006_nested_ignored() {
        // Deeply nested ignored directory
        assert!(is_ignored(&PathBuf::from(
            "/project/src/pkg/node_modules/dep/index.js"
        )));
    }

    // ============================================================================
    // DETECT LANGUAGE EDGE CASES
    // ============================================================================

    #[test]
    fn test_lang_cov_001_no_extension() {
        assert_eq!(
            detect_language_from_path(&PathBuf::from("/project/Dockerfile")),
            None
        );
    }

    #[test]
    fn test_lang_cov_002_hidden_file() {
        assert_eq!(
            detect_language_from_path(&PathBuf::from("/project/.gitignore")),
            None
        );
    }

    #[test]
    fn test_lang_cov_003_double_extension() {
        // Should detect based on final extension
        assert_eq!(
            detect_language_from_path(&PathBuf::from("file.test.py")),
            Some(Language::Python)
        );
    }

    // ============================================================================
    // ANALYZE_PROJECT ADDITIONAL TESTS
    // ============================================================================

    #[test]
    fn test_analyze_cov_001_no_flags() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("main.rs"), "fn main() {}\n").unwrap();

        let analysis = analyze_project(temp_dir.path(), false, false, false).unwrap();

        // Should return empty analysis when all flags are false
        assert!(analysis.languages.is_empty());
        assert!(analysis.dependencies.is_empty());
        assert!(analysis.tdg_score.is_none());
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_analyze_cov_002_with_tdg() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("main.rs"), "fn main() {}\n").unwrap();
        fs::create_dir(temp_dir.path().join("tests")).unwrap();
        fs::write(temp_dir.path().join("README.md"), "# Test\n").unwrap();

        let analysis = analyze_project(temp_dir.path(), true, false, false).unwrap();

        // TDG should be calculated
        assert!(analysis.tdg_score.is_some());
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_analyze_cov_003_zero_percentage() {
        let temp_dir = TempDir::new().unwrap();
        // Empty directory should result in 0% for any language

        let analysis = analyze_project(temp_dir.path(), false, true, false).unwrap();
        assert!(analysis.languages.is_empty());
    }
}
