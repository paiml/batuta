//! Quality Checker
//!
//! Runs quality assessments on PAIML stack components using PMAT tools.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

use super::hero_image::HeroImageResult;
use super::quality::{ComponentQuality, QualityGrade, Score, StackQualityReport};

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
                    let earned = json["total_earned"].as_f64().unwrap_or(0.0);
                    let possible = json["total_possible"].as_f64().unwrap_or(134.0);
                    let percentage = json["percentage"].as_f64().unwrap_or(0.0);

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
        use std::process::Command;

        let mut score = 50u32; // Base score

        // Check if tests pass (+20)
        let test_result = Command::new("cargo")
            .args(["test", "--quiet"])
            .current_dir(path)
            .output();

        if test_result.map(|o| o.status.success()).unwrap_or(false) {
            score += 20;
        }

        // Check if clippy passes (+15)
        let clippy_result = Command::new("cargo")
            .args(["clippy", "--quiet", "--", "-D", "warnings"])
            .current_dir(path)
            .output();

        if clippy_result.map(|o| o.status.success()).unwrap_or(false) {
            score += 15;
        }

        // Check for documentation (+10)
        let readme = path.join("README.md");
        if readme.exists() {
            score += 10;
        }

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
                    let total = json["total_score"].as_f64().unwrap_or(0.0).round() as u32;

                    // Extract documentation score from categories
                    let readme = json["categories"]["documentation"]["score"]
                        .as_f64()
                        .unwrap_or(0.0)
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

        // Check README.md (+10 base, +2 per section)
        let readme_path = path.join("README.md");
        if readme_path.exists() {
            repo_score += 10;
            readme_score += 5; // Base for existing

            if let Ok(content) = std::fs::read_to_string(&readme_path) {
                let content_lower = content.to_lowercase();

                // Check for required sections
                if content_lower.contains("## installation")
                    || content_lower.contains("# installation")
                {
                    readme_score += 3;
                }
                if content_lower.contains("## usage") || content_lower.contains("# usage") {
                    readme_score += 3;
                }
                if content_lower.contains("## license") || content_lower.contains("# license") {
                    readme_score += 3;
                }
                if content_lower.contains("## contributing")
                    || content_lower.contains("# contributing")
                {
                    readme_score += 3;
                }
                if content.len() > 500 {
                    readme_score += 3; // Substantial content
                }
            }
        }

        // Check for Makefile (+15)
        if path.join("Makefile").exists() {
            repo_score += 15;
        }

        // Check for CI (+15)
        if path.join(".github/workflows").exists() {
            repo_score += 15;
        }

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

    #[test]
    fn test_find_component_path_current_workspace() {
        let temp_dir = std::env::temp_dir().join("test_quality_checker_workspace");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        std::fs::write(
            temp_dir.join("Cargo.toml"),
            r#"[package]
name = "test-crate"
version = "1.0.0"
"#,
        )
        .unwrap();

        let checker = QualityChecker::new(temp_dir.clone());
        let path = checker.find_component_path("test-crate").unwrap();
        assert_eq!(path, temp_dir);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_find_component_path_sibling() {
        let temp_dir = std::env::temp_dir().join("test_quality_siblings");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let project_a = temp_dir.join("project-a");
        let project_b = temp_dir.join("project-b");
        std::fs::create_dir_all(&project_a).unwrap();
        std::fs::create_dir_all(&project_b).unwrap();

        std::fs::write(
            project_a.join("Cargo.toml"),
            r#"[package]
name = "project-a"
version = "1.0.0"
"#,
        )
        .unwrap();

        std::fs::write(
            project_b.join("Cargo.toml"),
            r#"[package]
name = "project-b"
version = "1.0.0"
"#,
        )
        .unwrap();

        let checker = QualityChecker::new(project_a.clone());
        let path = checker.find_component_path("project-b").unwrap();
        assert_eq!(path, project_b);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_find_component_path_not_found() {
        let temp_dir = std::env::temp_dir().join("test_quality_not_found");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let checker = QualityChecker::new(temp_dir.clone());
        let result = checker.find_component_path("nonexistent-crate");
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_find_component_no_cargo_toml() {
        let temp_dir = std::env::temp_dir().join("test_quality_no_cargo");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let checker = QualityChecker::new(temp_dir.clone());
        let result = checker.find_component_path("any-crate");
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
