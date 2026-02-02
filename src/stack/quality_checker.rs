//! Quality Checker
//!
//! Runs quality assessments on PAIML stack components using PMAT tools.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

use super::hero_image::HeroImageResult;
use super::quality::{ComponentQuality, QualityGrade, Score, StackQualityReport};

/// README sections to check and their point values.
const SECTION_CHECKS: &[(&str, u32)] = &[
    ("installation", 3),
    ("usage", 3),
    ("license", 3),
    ("contributing", 3),
];

/// Award `points` if `path.join(file)` exists, otherwise 0.
fn score_if_exists(path: &Path, file: &str, points: u32) -> u32 {
    if path.join(file).exists() {
        points
    } else {
        0
    }
}

/// Check whether a README section header exists (e.g. `## installation` or `# installation`).
fn check_section_exists(content_lower: &str, section: &str) -> bool {
    content_lower.contains(&format!("## {}", section))
        || content_lower.contains(&format!("# {}", section))
}

/// Extract an `f64` from a [`serde_json::Value`], returning `default` on failure.
fn extract_json_f64(value: &serde_json::Value, default: f64) -> f64 {
    value.as_f64().unwrap_or(default)
}

/// Run `cargo <subcommand>` in `dir` and return `points` if the command succeeds.
fn run_command_score(dir: &Path, args: &[&str], points: u32) -> u32 {
    use std::process::Command;

    let result = Command::new("cargo").args(args).current_dir(dir).output();
    if result.map(|o| o.status.success()).unwrap_or(false) {
        points
    } else {
        0
    }
}

/// Quality matrix checker for PAIML stack components
pub struct QualityChecker {
    /// Workspace root path
    workspace_root: PathBuf,
    /// Minimum required grade
    #[allow(dead_code)]
    min_grade: QualityGrade,
    /// Whether to require strict A+ for all
    #[allow(dead_code)]
    strict: bool,
}

impl QualityChecker {
    /// Create a new quality checker
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            min_grade: QualityGrade::AMinus,
            strict: false,
        }
    }

    /// Set minimum required grade
    #[allow(dead_code)]
    pub fn with_min_grade(mut self, grade: QualityGrade) -> Self {
        self.min_grade = grade;
        self
    }

    /// Enable strict A+ mode
    #[allow(dead_code)]
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Check quality for a single component
    pub async fn check_component(&self, name: &str) -> Result<ComponentQuality> {
        let path = self.find_component_path(name)?;

        // Run pmat rust-project-score
        let rust_score = self.run_rust_project_score(&path).await?;

        // Run pmat repo-score
        let (repo_score, readme_score) = self.run_repo_score(&path).await?;

        // Detect hero image
        let hero_image = HeroImageResult::detect(&path);

        Ok(ComponentQuality::new(
            name,
            path,
            rust_score,
            repo_score,
            readme_score,
            hero_image,
        ))
    }

    /// Check quality for all stack components
    #[allow(dead_code)]
    pub async fn check_all(&self) -> Result<StackQualityReport> {
        use crate::stack::PAIML_CRATES;

        let mut components = Vec::new();

        for crate_name in PAIML_CRATES {
            match self.check_component(crate_name).await {
                Ok(quality) => components.push(quality),
                Err(e) => {
                    // Log error but continue with other components
                    tracing::warn!("Failed to check {}: {}", crate_name, e);
                }
            }
        }

        Ok(StackQualityReport::from_components(components))
    }

    /// Find path to component repository
    fn find_component_path(&self, name: &str) -> Result<PathBuf> {
        // Check if it's the current workspace
        let cargo_toml = self.workspace_root.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                if content.contains(&format!("name = \"{}\"", name)) {
                    return Ok(self.workspace_root.clone());
                }
            }
        }

        // Check parent directory for sibling projects
        if let Some(parent) = self.workspace_root.parent() {
            let sibling = parent.join(name);
            if sibling.exists() && sibling.join("Cargo.toml").exists() {
                return Ok(sibling);
            }
        }

        Err(anyhow!("Could not find component: {}", name))
    }

    /// Run pmat rust-project-score on a path
    async fn run_rust_project_score(&self, path: &Path) -> Result<Score> {
        use std::process::Command;

        let output = Command::new("pmat")
            .args(["rust-project-score", "--path"])
            .arg(path)
            .args(["--format", "json"])
            .output();

        match output {
            Ok(output) if output.status.success() => {
                // Parse JSON output from pmat
                if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                    // pmat uses total_earned/total_possible (scale varies, normalize to 114)
                    let earned = extract_json_f64(&json["total_earned"], 0.0);
                    let possible = extract_json_f64(&json["total_possible"], 134.0);
                    let percentage = extract_json_f64(&json["percentage"], 0.0);

                    // Normalize to 0-114 scale for consistent grading
                    let normalized_score = ((percentage / 100.0) * 114.0).round() as u32;
                    let grade = QualityGrade::from_rust_project_score(normalized_score);

                    // Store actual earned/possible but use normalized for grade
                    return Ok(Score {
                        value: earned.round() as u32,
                        max: possible.round() as u32,
                        grade,
                    });
                }
            }
            Ok(output) => {
                // Log stderr for debugging
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stderr.is_empty() {
                    tracing::debug!("pmat stderr: {}", stderr);
                }
            }
            Err(e) => {
                tracing::debug!("pmat not available: {}", e);
            }
        }

        // Fallback: estimate score from cargo test and clippy
        self.estimate_rust_score(path).await
    }

    /// Estimate rust project score when pmat is not available
    async fn estimate_rust_score(&self, path: &Path) -> Result<Score> {
        let mut score = 50u32; // Base score

        // Check if tests pass (+20)
        score += run_command_score(path, &["test", "--quiet"], 20);

        // Check if clippy passes (+15)
        score += run_command_score(path, &["clippy", "--quiet", "--", "-D", "warnings"], 15);

        // Check for documentation (+10)
        score += score_if_exists(path, "README.md", 10);

        // Check for Cargo.toml metadata (+5)
        let cargo_toml = path.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                if content.contains("[package.metadata") || content.contains("documentation =") {
                    score += 5;
                }
            }
        }

        let grade = QualityGrade::from_rust_project_score(score);
        Ok(Score::new(score, 114, grade))
    }

    /// Run pmat repo-score on a path
    async fn run_repo_score(&self, path: &Path) -> Result<(Score, Score)> {
        use std::process::Command;

        let output = Command::new("pmat")
            .args(["repo-score", "--path"])
            .arg(path)
            .args(["--format", "json"])
            .output();

        match output {
            Ok(output) if output.status.success() => {
                if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                    // total_score is a float in pmat output
                    let total = extract_json_f64(&json["total_score"], 0.0).round() as u32;

                    // Extract documentation score from categories
                    let readme = extract_json_f64(
                        &json["categories"]["documentation"]["score"],
                        0.0,
                    )
                    .round() as u32;

                    let repo_grade = QualityGrade::from_repo_score(total);
                    let readme_grade = QualityGrade::from_readme_score(readme);

                    return Ok((
                        Score::new(total, 110, repo_grade),
                        Score::new(readme, 20, readme_grade),
                    ));
                }
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stderr.is_empty() {
                    tracing::debug!("pmat repo-score stderr: {}", stderr);
                }
            }
            Err(e) => {
                tracing::debug!("pmat repo-score not available: {}", e);
            }
        }

        // Fallback: estimate scores
        self.estimate_repo_scores(path).await
    }

    /// Estimate repo and readme scores when pmat is not available
    async fn estimate_repo_scores(&self, path: &Path) -> Result<(Score, Score)> {
        let mut repo_score = 40u32; // Base score
        let mut readme_score = 0u32;

        // Check README.md (+10 base, +points per section)
        let readme_path = path.join("README.md");
        if readme_path.exists() {
            repo_score += 10;
            readme_score += 5; // Base for existing

            if let Ok(content) = std::fs::read_to_string(&readme_path) {
                let content_lower = content.to_lowercase();

                // Check for required sections via table-driven lookup
                for &(section, points) in SECTION_CHECKS {
                    if check_section_exists(&content_lower, section) {
                        readme_score += points;
                    }
                }
                if content.len() > 500 {
                    readme_score += 3; // Substantial content
                }
            }
        }

        // Check for Makefile (+15)
        repo_score += score_if_exists(path, "Makefile", 15);

        // Check for CI (+15)
        repo_score += score_if_exists(path, ".github/workflows", 15);

        // Check for pre-commit hooks (+10)
        if path.join(".pre-commit-config.yaml").exists()
            || path.join(".git/hooks/pre-commit").exists()
        {
            repo_score += 10;
        }

        readme_score = readme_score.min(20);
        let repo_grade = QualityGrade::from_repo_score(repo_score);
        let readme_grade = QualityGrade::from_readme_score(readme_score);

        Ok((
            Score::new(repo_score, 110, repo_grade),
            Score::new(readme_score, 20, readme_grade),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Create a fresh temp directory, removing any stale leftover first.
    fn setup_test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Best-effort cleanup of a temp directory.
    fn cleanup_test_dir(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    // ===== Free function tests =====

    #[test]
    fn test_score_if_exists_present() {
        let dir = setup_test_dir("test_qc_score_exists");
        std::fs::write(dir.join("README.md"), "hello").unwrap();
        assert_eq!(score_if_exists(&dir, "README.md", 10), 10);
        cleanup_test_dir(&dir);
    }

    #[test]
    fn test_score_if_exists_absent() {
        let dir = setup_test_dir("test_qc_score_absent");
        assert_eq!(score_if_exists(&dir, "README.md", 10), 0);
        cleanup_test_dir(&dir);
    }

    #[test]
    fn test_check_section_exists_h2() {
        assert!(check_section_exists("## installation\nsome text", "installation"));
    }

    #[test]
    fn test_check_section_exists_h1() {
        assert!(check_section_exists("# usage\nsome text", "usage"));
    }

    #[test]
    fn test_check_section_exists_missing() {
        assert!(!check_section_exists("some random text", "installation"));
    }

    #[test]
    fn test_extract_json_f64_present() {
        let val = serde_json::json!(42.5);
        assert_eq!(extract_json_f64(&val, 0.0), 42.5);
    }

    #[test]
    fn test_extract_json_f64_null() {
        let val = serde_json::json!(null);
        assert_eq!(extract_json_f64(&val, 99.0), 99.0);
    }

    #[test]
    fn test_extract_json_f64_string() {
        let val = serde_json::json!("not a number");
        assert_eq!(extract_json_f64(&val, 7.0), 7.0);
    }

    #[test]
    fn test_extract_json_f64_integer() {
        let val = serde_json::json!(100);
        assert_eq!(extract_json_f64(&val, 0.0), 100.0);
    }

    #[test]
    fn test_run_command_score_success() {
        // run_command_score runs `cargo <args>`, so use a cargo subcommand
        let dir = setup_test_dir("test_qc_cmd_score");
        assert_eq!(run_command_score(&dir, &["--version"], 20), 20);
        cleanup_test_dir(&dir);
    }

    #[test]
    fn test_run_command_score_failure() {
        let dir = setup_test_dir("test_qc_cmd_fail");
        assert_eq!(run_command_score(&dir, &["false"], 20), 0);
        cleanup_test_dir(&dir);
    }

    #[test]
    fn test_run_command_score_not_found() {
        let dir = setup_test_dir("test_qc_cmd_notfound");
        assert_eq!(
            run_command_score(&dir, &["nonexistent_tool_xyz_abc"], 20),
            0
        );
        cleanup_test_dir(&dir);
    }

    // ===== QualityChecker construction =====

    #[test]
    fn test_quality_checker_creation() {
        let checker = QualityChecker::new(PathBuf::from("/tmp"));
        assert_eq!(checker.min_grade, QualityGrade::AMinus);
        assert!(!checker.strict);
    }

    #[test]
    fn test_quality_checker_with_min_grade() {
        let checker =
            QualityChecker::new(PathBuf::from("/tmp")).with_min_grade(QualityGrade::APlus);
        assert_eq!(checker.min_grade, QualityGrade::APlus);
    }

    #[test]
    fn test_quality_checker_strict() {
        let checker = QualityChecker::new(PathBuf::from("/tmp")).strict(true);
        assert!(checker.strict);
    }

    #[test]
    fn test_quality_checker_chaining() {
        let checker = QualityChecker::new(PathBuf::from("/tmp"))
            .with_min_grade(QualityGrade::AMinus)
            .strict(true);
        assert_eq!(checker.min_grade, QualityGrade::AMinus);
        assert!(checker.strict);
    }

    // ===== find_component_path =====

    #[test]
    fn test_find_component_path_current_workspace() {
        let temp_dir = setup_test_dir("test_quality_checker_workspace");
        std::fs::write(
            temp_dir.join("Cargo.toml"),
            "[package]\nname = \"test-crate\"\nversion = \"1.0.0\"\n",
        )
        .unwrap();
        let checker = QualityChecker::new(temp_dir.clone());
        let path = checker.find_component_path("test-crate").unwrap();
        assert_eq!(path, temp_dir);
        cleanup_test_dir(&temp_dir);
    }

    #[test]
    fn test_find_component_path_sibling() {
        let temp_dir = setup_test_dir("test_quality_siblings");
        let project_a = temp_dir.join("project-a");
        let project_b = temp_dir.join("project-b");
        std::fs::create_dir_all(&project_a).unwrap();
        std::fs::create_dir_all(&project_b).unwrap();
        std::fs::write(
            project_a.join("Cargo.toml"),
            "[package]\nname = \"project-a\"\nversion = \"1.0.0\"\n",
        )
        .unwrap();
        std::fs::write(
            project_b.join("Cargo.toml"),
            "[package]\nname = \"project-b\"\nversion = \"1.0.0\"\n",
        )
        .unwrap();
        let checker = QualityChecker::new(project_a.clone());
        let path = checker.find_component_path("project-b").unwrap();
        assert_eq!(path, project_b);
        cleanup_test_dir(&temp_dir);
    }

    #[test]
    fn test_find_component_path_not_found() {
        let temp_dir = setup_test_dir("test_quality_not_found");
        let checker = QualityChecker::new(temp_dir.clone());
        assert!(checker.find_component_path("nonexistent-crate").is_err());
        cleanup_test_dir(&temp_dir);
    }

    #[test]
    fn test_find_component_no_cargo_toml() {
        let temp_dir = setup_test_dir("test_quality_no_cargo");
        let checker = QualityChecker::new(temp_dir.clone());
        assert!(checker.find_component_path("any-crate").is_err());
        cleanup_test_dir(&temp_dir);
    }

    #[test]
    fn test_find_component_cargo_toml_no_match() {
        let temp_dir = setup_test_dir("test_quality_no_match");
        std::fs::write(
            temp_dir.join("Cargo.toml"),
            "[package]\nname = \"other-crate\"\nversion = \"1.0.0\"\n",
        )
        .unwrap();
        let checker = QualityChecker::new(temp_dir.clone());
        // Name doesn't match and no sibling → error
        assert!(checker.find_component_path("wanted-crate").is_err());
        cleanup_test_dir(&temp_dir);
    }

    // ===== estimate_repo_scores =====

    #[tokio::test]
    async fn test_estimate_repo_scores_empty_dir() {
        let dir = setup_test_dir("test_qc_repo_empty");
        let checker = QualityChecker::new(dir.clone());
        let (repo, readme) = checker.estimate_repo_scores(&dir).await.unwrap();
        assert_eq!(repo.value, 40); // base only
        assert_eq!(readme.value, 0);
        cleanup_test_dir(&dir);
    }

    #[tokio::test]
    async fn test_estimate_repo_scores_with_readme() {
        let dir = setup_test_dir("test_qc_repo_readme");
        // 5 (base) + 3 (installation) + 3 (usage) + 3 (license) + 3 (contributing) = 17
        std::fs::write(
            dir.join("README.md"),
            "# My Project\n\n## Installation\n\nRun cargo install.\n\n## Usage\n\nJust run it.\n\n## License\n\nMIT\n\n## Contributing\n\nPRs welcome.\n",
        )
        .unwrap();
        let checker = QualityChecker::new(dir.clone());
        let (repo, readme) = checker.estimate_repo_scores(&dir).await.unwrap();
        assert!(repo.value > 40); // base + README
        assert_eq!(readme.value, 17);
        cleanup_test_dir(&dir);
    }

    #[tokio::test]
    async fn test_estimate_repo_scores_with_makefile_and_ci() {
        let dir = setup_test_dir("test_qc_repo_mk_ci");
        std::fs::write(dir.join("Makefile"), "all:\n\ttrue\n").unwrap();
        std::fs::create_dir_all(dir.join(".github/workflows")).unwrap();
        let checker = QualityChecker::new(dir.clone());
        let (repo, _) = checker.estimate_repo_scores(&dir).await.unwrap();
        assert_eq!(repo.value, 40 + 15 + 15); // base + Makefile + CI
        cleanup_test_dir(&dir);
    }

    #[tokio::test]
    async fn test_estimate_repo_scores_with_precommit() {
        let dir = setup_test_dir("test_qc_repo_precommit");
        std::fs::write(dir.join(".pre-commit-config.yaml"), "repos: []\n").unwrap();
        let checker = QualityChecker::new(dir.clone());
        let (repo, _) = checker.estimate_repo_scores(&dir).await.unwrap();
        assert_eq!(repo.value, 40 + 10); // base + pre-commit
        cleanup_test_dir(&dir);
    }

    #[tokio::test]
    async fn test_estimate_repo_scores_readme_partial_sections() {
        let dir = setup_test_dir("test_qc_repo_partial");
        std::fs::write(dir.join("README.md"), "# Proj\n\n## Installation\nstuff\n").unwrap();
        let checker = QualityChecker::new(dir.clone());
        let (_, readme) = checker.estimate_repo_scores(&dir).await.unwrap();
        // 5 base + 3 installation = 8
        assert_eq!(readme.value, 8);
        cleanup_test_dir(&dir);
    }

    // ===== estimate_rust_score =====

    #[tokio::test]
    async fn test_estimate_rust_score_empty_dir() {
        let dir = setup_test_dir("test_qc_rust_empty");
        let checker = QualityChecker::new(dir.clone());
        let score = checker.estimate_rust_score(&dir).await.unwrap();
        // Base 50 only - cargo test/clippy will fail, no README, no metadata
        assert_eq!(score.value, 50);
        cleanup_test_dir(&dir);
    }

    #[tokio::test]
    async fn test_estimate_rust_score_with_readme() {
        let dir = setup_test_dir("test_qc_rust_readme");
        std::fs::write(dir.join("README.md"), "# Hello").unwrap();
        let checker = QualityChecker::new(dir.clone());
        let score = checker.estimate_rust_score(&dir).await.unwrap();
        assert_eq!(score.value, 60); // 50 base + 10 README
        cleanup_test_dir(&dir);
    }

    #[tokio::test]
    async fn test_estimate_rust_score_with_metadata() {
        let dir = setup_test_dir("test_qc_rust_meta");
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"x\"\ndocumentation = \"https://docs.rs/x\"\n",
        )
        .unwrap();
        let checker = QualityChecker::new(dir.clone());
        let score = checker.estimate_rust_score(&dir).await.unwrap();
        assert_eq!(score.value, 55); // 50 base + 5 metadata
        cleanup_test_dir(&dir);
    }

    #[tokio::test]
    async fn test_estimate_rust_score_with_package_metadata() {
        let dir = setup_test_dir("test_qc_rust_pkgmeta");
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"x\"\n[package.metadata.foo]\nbar = true\n",
        )
        .unwrap();
        let checker = QualityChecker::new(dir.clone());
        let score = checker.estimate_rust_score(&dir).await.unwrap();
        assert_eq!(score.value, 55); // 50 base + 5 metadata
        cleanup_test_dir(&dir);
    }

    #[tokio::test]
    async fn test_estimate_rust_score_no_cargo_toml() {
        let dir = setup_test_dir("test_qc_rust_nocargo");
        let checker = QualityChecker::new(dir.clone());
        let score = checker.estimate_rust_score(&dir).await.unwrap();
        assert_eq!(score.value, 50); // base only
        cleanup_test_dir(&dir);
    }

    #[tokio::test]
    async fn test_estimate_rust_score_with_readme_and_metadata() {
        let dir = setup_test_dir("test_qc_rust_both");
        std::fs::write(dir.join("README.md"), "# Project\n").unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"x\"\ndocumentation = \"y\"\n",
        )
        .unwrap();
        let checker = QualityChecker::new(dir.clone());
        let score = checker.estimate_rust_score(&dir).await.unwrap();
        assert_eq!(score.value, 65); // 50 + 10 README + 5 metadata
        cleanup_test_dir(&dir);
    }

    // ===== run_rust_project_score / run_repo_score (pmat fallback) =====

    #[tokio::test]
    async fn test_run_rust_project_score_returns_valid() {
        // pmat may or may not be installed; just verify we get a valid score
        let dir = setup_test_dir("test_qc_pmat_fallback");
        std::fs::write(dir.join("README.md"), "# Hi").unwrap();
        let checker = QualityChecker::new(dir.clone());
        let score = checker.run_rust_project_score(&dir).await.unwrap();
        assert!(score.value > 0);
        assert!(score.max > 0);
        cleanup_test_dir(&dir);
    }

    #[tokio::test]
    async fn test_run_repo_score_returns_valid() {
        // pmat may or may not be installed; just verify we get valid scores
        let dir = setup_test_dir("test_qc_repo_fallback");
        std::fs::write(dir.join("README.md"), "# Project\n## Installation\nstuff\n").unwrap();
        std::fs::write(dir.join("Makefile"), "all:\n").unwrap();
        let checker = QualityChecker::new(dir.clone());
        let (repo, readme) = checker.run_repo_score(&dir).await.unwrap();
        assert!(repo.value > 0);
        assert!(repo.max > 0);
        assert!(readme.max > 0);
        cleanup_test_dir(&dir);
    }

    // ===== check_component =====

    #[tokio::test]
    async fn test_check_component_self() {
        let dir = setup_test_dir("test_qc_check_self");
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"my-crate\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        std::fs::write(dir.join("README.md"), "# My Crate\n## Usage\nstuff\n").unwrap();
        let checker = QualityChecker::new(dir.clone());
        let result = checker.check_component("my-crate").await.unwrap();
        assert_eq!(result.name, "my-crate");
        cleanup_test_dir(&dir);
    }

    #[tokio::test]
    async fn test_check_component_not_found() {
        let dir = setup_test_dir("test_qc_check_notfound");
        let checker = QualityChecker::new(dir.clone());
        let result = checker.check_component("nonexistent-xyz").await;
        assert!(result.is_err());
        cleanup_test_dir(&dir);
    }

    // ===== readme length bonus =====

    #[tokio::test]
    async fn test_estimate_repo_scores_readme_length_bonus() {
        let dir = setup_test_dir("test_qc_readme_len");
        let long_content = format!(
            "# Project\n\n## Installation\n\n{}\n",
            "x".repeat(600)
        );
        std::fs::write(dir.join("README.md"), &long_content).unwrap();
        let checker = QualityChecker::new(dir.clone());
        let (_, readme) = checker.estimate_repo_scores(&dir).await.unwrap();
        // 5 base + 3 installation + 3 length bonus = 11
        assert_eq!(readme.value, 11);
        cleanup_test_dir(&dir);
    }

    // ===== section checks edge cases =====

    #[test]
    fn test_check_section_exists_lowered_input() {
        // The function expects content already lowercased by the caller
        assert!(check_section_exists("## installation\n", "installation"));
    }

    #[test]
    fn test_check_section_exists_no_hash() {
        assert!(!check_section_exists("installation\n", "installation"));
    }

    // ===== pre-commit git hook detection =====

    #[tokio::test]
    async fn test_estimate_repo_scores_git_hook() {
        let dir = setup_test_dir("test_qc_githook");
        std::fs::create_dir_all(dir.join(".git/hooks")).unwrap();
        std::fs::write(dir.join(".git/hooks/pre-commit"), "#!/bin/sh\n").unwrap();
        let checker = QualityChecker::new(dir.clone());
        let (repo, _) = checker.estimate_repo_scores(&dir).await.unwrap();
        assert_eq!(repo.value, 40 + 10); // base + pre-commit
        cleanup_test_dir(&dir);
    }
}
