//! Stack Quality Matrix
//!
//! Implements the quality enforcement system for PAIML stack components.
//! Ensures all components meet A+ quality standards before release.
//!
//! ## Quality Dimensions
//!
//! - **Rust Project Score** (pmat rust-project-score): 105-114 for A+
//! - **Repository Score** (pmat repo-score): 95-110 for A+
//! - **README Score**: 18-20 for A+
//! - **Hero Image**: Present and valid

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ============================================================================
// Quality Grade
// ============================================================================

/// Quality grade levels based on score percentages
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QualityGrade {
    /// A+ (95-100% of max score)
    APlus,
    /// A (90-94%)
    A,
    /// A- (85-89%) - PMAT minimum for production
    AMinus,
    /// B+ (80-84%)
    BPlus,
    /// B (70-79%)
    B,
    /// C (60-69%)
    C,
    /// D (50-59%)
    D,
    /// F (0-49%)
    F,
}

impl QualityGrade {
    /// Calculate grade from Rust project score (max 114)
    pub fn from_rust_project_score(score: u32) -> Self {
        match score {
            105..=114 => Self::APlus,
            95..=104 => Self::A,
            85..=94 => Self::AMinus,
            80..=84 => Self::BPlus,
            70..=79 => Self::B,
            60..=69 => Self::C,
            50..=59 => Self::D,
            _ => Self::F,
        }
    }

    /// Calculate grade from repository score (max 110)
    pub fn from_repo_score(score: u32) -> Self {
        match score {
            95..=110 => Self::APlus,
            90..=94 => Self::A,
            85..=89 => Self::AMinus,
            80..=84 => Self::BPlus,
            70..=79 => Self::B,
            60..=69 => Self::C,
            50..=59 => Self::D,
            _ => Self::F,
        }
    }

    /// Calculate grade from README score (max 20)
    pub fn from_readme_score(score: u32) -> Self {
        match score {
            18..=20 => Self::APlus,
            16..=17 => Self::A,
            14..=15 => Self::AMinus,
            12..=13 => Self::BPlus,
            10..=11 => Self::B,
            8..=9 => Self::C,
            6..=7 => Self::D,
            _ => Self::F,
        }
    }

    /// Calculate grade from Stack Quality Index (0-100)
    pub fn from_sqi(sqi: f64) -> Self {
        match sqi as u32 {
            95..=100 => Self::APlus,
            90..=94 => Self::A,
            85..=89 => Self::AMinus,
            80..=84 => Self::BPlus,
            70..=79 => Self::B,
            60..=69 => Self::C,
            50..=59 => Self::D,
            _ => Self::F,
        }
    }

    /// Check if grade is release-ready (A- or better)
    pub fn is_release_ready(&self) -> bool {
        matches!(self, Self::APlus | Self::A | Self::AMinus)
    }

    /// Check if grade is A+
    pub fn is_a_plus(&self) -> bool {
        matches!(self, Self::APlus)
    }

    /// Get display symbol for grade
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::APlus => "A+",
            Self::A => "A",
            Self::AMinus => "A-",
            Self::BPlus => "B+",
            Self::B => "B",
            Self::C => "C",
            Self::D => "D",
            Self::F => "F",
        }
    }

    /// Get status icon for grade
    pub fn icon(&self) -> &'static str {
        match self {
            Self::APlus => "✅",
            Self::A | Self::AMinus => "⚠️",
            _ => "❌",
        }
    }
}

impl std::fmt::Display for QualityGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

// ============================================================================
// Score Types
// ============================================================================

/// Score with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    /// Actual score value
    pub value: u32,
    /// Maximum possible score
    pub max: u32,
    /// Calculated grade
    pub grade: QualityGrade,
}

impl Score {
    /// Create a new score
    pub fn new(value: u32, max: u32, grade: QualityGrade) -> Self {
        Self { value, max, grade }
    }

    /// Calculate percentage
    pub fn percentage(&self) -> f64 {
        if self.max == 0 {
            0.0
        } else {
            (self.value as f64 / self.max as f64) * 100.0
        }
    }

    /// Normalize to 0-100 scale
    pub fn normalized(&self) -> f64 {
        self.percentage()
    }
}

// ============================================================================
// Hero Image
// ============================================================================

/// Supported image formats for hero images
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageFormat {
    Png,
    Jpg,
    WebP,
    Svg,
}

impl ImageFormat {
    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpg),
            "webp" => Some(Self::WebP),
            "svg" => Some(Self::Svg),
            _ => None,
        }
    }

    /// Get file extension for format
    #[allow(dead_code)]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpg => "jpg",
            Self::WebP => "webp",
            Self::Svg => "svg",
        }
    }
}

/// Hero image detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroImageResult {
    /// Whether a hero image was found
    pub present: bool,
    /// Path to the hero image (if found)
    pub path: Option<PathBuf>,
    /// Detected format
    pub format: Option<ImageFormat>,
    /// Image dimensions (width, height) if detectable
    pub dimensions: Option<(u32, u32)>,
    /// File size in bytes
    pub file_size: Option<u64>,
    /// Whether the image passes validation
    pub valid: bool,
    /// Issues found during validation
    pub issues: Vec<String>,
}

impl HeroImageResult {
    /// Create result for missing hero image
    pub fn missing() -> Self {
        Self {
            present: false,
            path: None,
            format: None,
            dimensions: None,
            file_size: None,
            valid: false,
            issues: vec!["No hero image found".to_string()],
        }
    }

    /// Create result for found hero image
    pub fn found(path: PathBuf, format: ImageFormat) -> Self {
        Self {
            present: true,
            path: Some(path),
            format: Some(format),
            dimensions: None,
            file_size: None,
            valid: true,
            issues: vec![],
        }
    }

    /// Detect hero image in repository
    pub fn detect(repo_path: &Path) -> Self {
        // Priority 1: Check docs/hero.* files (SVG preferred)
        for ext in &["svg", "png", "jpg", "jpeg", "webp"] {
            let hero_path = repo_path.join(format!("docs/hero.{}", ext));
            if hero_path.exists() {
                if let Some(format) = ImageFormat::from_extension(ext) {
                    return Self::validate_at_path(&hero_path, format);
                }
            }
        }

        // Priority 2: Check assets/hero.* files (SVG preferred)
        for ext in &["svg", "png", "jpg", "jpeg", "webp"] {
            let hero_path = repo_path.join(format!("assets/hero.{}", ext));
            if hero_path.exists() {
                if let Some(format) = ImageFormat::from_extension(ext) {
                    return Self::validate_at_path(&hero_path, format);
                }
            }
        }

        // Priority 3: Parse README.md for first image
        let readme_path = repo_path.join("README.md");
        if readme_path.exists() {
            if let Some(img_ref) = Self::extract_first_image_from_readme(&readme_path) {
                let full_path = repo_path.join(&img_ref);
                if full_path.exists() {
                    if let Some(ext) = full_path.extension().and_then(|e| e.to_str()) {
                        if let Some(format) = ImageFormat::from_extension(ext) {
                            return Self::validate_at_path(&full_path, format);
                        }
                    }
                }
            }
        }

        Self::missing()
    }

    /// Validate image at path
    fn validate_at_path(path: &Path, format: ImageFormat) -> Self {
        let mut result = Self::found(path.to_path_buf(), format);
        let mut issues = Vec::new();

        // Check file size
        if let Ok(metadata) = std::fs::metadata(path) {
            let size = metadata.len();
            result.file_size = Some(size);

            // Max 2MB
            if size > 2 * 1024 * 1024 {
                issues.push(format!("Image too large: {} bytes (max 2MB)", size));
            }
        }

        // Note: Dimension checking would require image crate
        // For now, we skip dimension validation

        if !issues.is_empty() {
            result.valid = false;
            result.issues = issues;
        }

        result
    }

    /// Extract first image reference from README.md
    fn extract_first_image_from_readme(readme_path: &Path) -> Option<String> {
        let content = std::fs::read_to_string(readme_path).ok()?;

        // Match markdown image syntax: ![alt](path)
        let re = regex_lite::Regex::new(r"!\[.*?\]\(([^)]+)\)").ok()?;
        if let Some(caps) = re.captures(&content) {
            let img_path = caps.get(1)?.as_str();
            // Skip external URLs
            if !img_path.starts_with("http://") && !img_path.starts_with("https://") {
                return Some(img_path.to_string());
            }
        }

        // Match HTML img syntax: <img src="path"
        let re_html = regex_lite::Regex::new(r#"<img[^>]+src=["']([^"']+)["']"#).ok()?;
        if let Some(caps) = re_html.captures(&content) {
            let img_path = caps.get(1)?.as_str();
            if !img_path.starts_with("http://") && !img_path.starts_with("https://") {
                return Some(img_path.to_string());
            }
        }

        None
    }
}

// ============================================================================
// Quality Issue
// ============================================================================

/// Severity of a quality issue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Blocks release
    Error,
    /// Should be addressed
    Warning,
    /// Informational
    Info,
}

/// A quality issue found during analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue type identifier
    pub issue_type: String,
    /// Human-readable message
    pub message: String,
    /// Severity level
    pub severity: IssueSeverity,
    /// Recommended action
    pub recommendation: Option<String>,
}

impl QualityIssue {
    /// Create a new quality issue
    pub fn new(
        issue_type: impl Into<String>,
        message: impl Into<String>,
        severity: IssueSeverity,
    ) -> Self {
        Self {
            issue_type: issue_type.into(),
            message: message.into(),
            severity,
            recommendation: None,
        }
    }

    /// Add recommendation
    pub fn with_recommendation(mut self, rec: impl Into<String>) -> Self {
        self.recommendation = Some(rec.into());
        self
    }

    /// Create error for score below threshold
    pub fn score_below_threshold(metric: &str, score: u32, threshold: u32) -> Self {
        Self::new(
            format!("{}_below_threshold", metric),
            format!(
                "{} score {} below A- threshold ({})",
                metric, score, threshold
            ),
            IssueSeverity::Error,
        )
        .with_recommendation(format!("Improve {} to at least {}", metric, threshold))
    }

    /// Create error for missing hero image
    pub fn missing_hero_image() -> Self {
        Self::new(
            "missing_hero_image",
            "No hero image found",
            IssueSeverity::Error,
        )
        .with_recommendation("Add hero.png to docs/ or include image at top of README.md")
    }
}

// ============================================================================
// Stack Layer
// ============================================================================

/// Layer in the PAIML stack architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StackLayer {
    /// Layer 0: Compute primitives (trueno, etc.)
    Compute,
    /// Layer 1: ML algorithms (aprender, etc.)
    Ml,
    /// Layer 2: Training & inference
    Training,
    /// Layer 3: Transpilers
    Transpilers,
    /// Layer 4: Orchestration
    Orchestration,
    /// Layer 5: Quality tools
    Quality,
    /// Layer 6: Data & MLOps
    DataMlops,
    /// Layer 7: Presentation
    Presentation,
}

impl StackLayer {
    /// Determine layer from component name
    pub fn from_component(name: &str) -> Self {
        match name {
            "trueno" | "trueno-viz" | "trueno-db" | "trueno-graph" | "trueno-rag" => Self::Compute,
            "aprender" | "aprender-shell" | "aprender-tsp" => Self::Ml,
            "entrenar" | "realizar" => Self::Training,
            "depyler" | "decy" | "ruchy" => Self::Transpilers,
            "batuta" | "repartir" | "pforge" => Self::Orchestration,
            "certeza" | "renacer" | "pmat" => Self::Quality,
            "alimentar" | "pacha" => Self::DataMlops,
            "presentar"
            | "sovereign-ai-stack-book"
            | "apr-cookbook"
            | "alm-cookbook"
            | "pres-cookbook" => Self::Presentation,
            _ => Self::Orchestration, // Default
        }
    }

    /// Get display name for layer
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Compute => "COMPUTE PRIMITIVES",
            Self::Ml => "ML ALGORITHMS",
            Self::Training => "TRAINING & INFERENCE",
            Self::Transpilers => "TRANSPILERS",
            Self::Orchestration => "ORCHESTRATION",
            Self::Quality => "QUALITY",
            Self::DataMlops => "DATA & MLOPS",
            Self::Presentation => "PRESENTATION",
        }
    }
}

// ============================================================================
// Component Quality
// ============================================================================

/// Quality assessment for a single component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentQuality {
    /// Component name
    pub name: String,
    /// Layer in the stack
    pub layer: StackLayer,
    /// Repository path
    pub path: PathBuf,
    /// Rust project score
    pub rust_score: Score,
    /// Repository score
    pub repo_score: Score,
    /// README score
    pub readme_score: Score,
    /// Hero image result
    pub hero_image: HeroImageResult,
    /// Stack Quality Index (0-100)
    pub sqi: f64,
    /// Overall grade
    pub grade: QualityGrade,
    /// Whether release is allowed
    pub release_ready: bool,
    /// Issues found
    pub issues: Vec<QualityIssue>,
}

impl ComponentQuality {
    /// Create a new component quality assessment
    pub fn new(
        name: impl Into<String>,
        path: PathBuf,
        rust_score: Score,
        repo_score: Score,
        readme_score: Score,
        hero_image: HeroImageResult,
    ) -> Self {
        let name = name.into();
        let layer = StackLayer::from_component(&name);
        let sqi = Self::calculate_sqi(&rust_score, &repo_score, &readme_score, &hero_image);
        let grade = QualityGrade::from_sqi(sqi);
        let issues = Self::collect_issues(&rust_score, &repo_score, &readme_score, &hero_image);
        let release_ready = grade.is_release_ready() && hero_image.valid;

        Self {
            name,
            layer,
            path,
            rust_score,
            repo_score,
            readme_score,
            hero_image,
            sqi,
            grade,
            release_ready,
            issues,
        }
    }

    /// Calculate Stack Quality Index
    /// SQI = (0.40 × Rust) + (0.30 × Repo) + (0.20 × README) + (0.10 × Hero)
    pub fn calculate_sqi(
        rust: &Score,
        repo: &Score,
        readme: &Score,
        hero: &HeroImageResult,
    ) -> f64 {
        let rust_normalized = rust.normalized();
        let repo_normalized = repo.normalized();
        let readme_normalized = readme.normalized();
        let hero_normalized = if hero.valid { 100.0 } else { 0.0 };

        (0.40 * rust_normalized)
            + (0.30 * repo_normalized)
            + (0.20 * readme_normalized)
            + (0.10 * hero_normalized)
    }

    /// Collect issues based on scores
    fn collect_issues(
        rust: &Score,
        repo: &Score,
        readme: &Score,
        hero: &HeroImageResult,
    ) -> Vec<QualityIssue> {
        let mut issues = Vec::new();

        // Check Rust score (A- threshold = 85)
        if rust.value < 85 {
            issues.push(QualityIssue::score_below_threshold(
                "rust_project",
                rust.value,
                85,
            ));
        }

        // Check Repo score (A- threshold = 85)
        if repo.value < 85 {
            issues.push(QualityIssue::score_below_threshold("repo", repo.value, 85));
        }

        // Check README score (A- threshold = 14)
        if readme.value < 14 {
            issues.push(QualityIssue::score_below_threshold(
                "readme",
                readme.value,
                14,
            ));
        }

        // Check hero image
        if !hero.valid {
            issues.push(QualityIssue::missing_hero_image());
        }

        issues
    }
}

// ============================================================================
// Quality Summary
// ============================================================================

/// Summary statistics for stack quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySummary {
    /// Total components checked
    pub total_components: usize,
    /// Components with A+ grade
    pub a_plus_count: usize,
    /// Components with A grade
    pub a_count: usize,
    /// Components with A- grade
    pub a_minus_count: usize,
    /// Components below A- threshold
    pub below_threshold_count: usize,
    /// Components missing hero image
    pub missing_hero_count: usize,
    /// Average rust score
    pub avg_rust_score: f64,
    /// Average repo score
    pub avg_repo_score: f64,
    /// Average readme score
    pub avg_readme_score: f64,
}

impl QualitySummary {
    /// Calculate summary from component list
    pub fn from_components(components: &[ComponentQuality]) -> Self {
        let total = components.len();
        if total == 0 {
            return Self {
                total_components: 0,
                a_plus_count: 0,
                a_count: 0,
                a_minus_count: 0,
                below_threshold_count: 0,
                missing_hero_count: 0,
                avg_rust_score: 0.0,
                avg_repo_score: 0.0,
                avg_readme_score: 0.0,
            };
        }

        let a_plus_count = components
            .iter()
            .filter(|c| c.grade == QualityGrade::APlus)
            .count();
        let a_count = components
            .iter()
            .filter(|c| c.grade == QualityGrade::A)
            .count();
        let a_minus_count = components
            .iter()
            .filter(|c| c.grade == QualityGrade::AMinus)
            .count();
        let below_threshold_count = components
            .iter()
            .filter(|c| !c.grade.is_release_ready())
            .count();
        let missing_hero_count = components.iter().filter(|c| !c.hero_image.valid).count();

        let avg_rust_score = components
            .iter()
            .map(|c| c.rust_score.value as f64)
            .sum::<f64>()
            / total as f64;
        let avg_repo_score = components
            .iter()
            .map(|c| c.repo_score.value as f64)
            .sum::<f64>()
            / total as f64;
        let avg_readme_score = components
            .iter()
            .map(|c| c.readme_score.value as f64)
            .sum::<f64>()
            / total as f64;

        Self {
            total_components: total,
            a_plus_count,
            a_count,
            a_minus_count,
            below_threshold_count,
            missing_hero_count,
            avg_rust_score,
            avg_repo_score,
            avg_readme_score,
        }
    }
}

// ============================================================================
// Stack Quality Report
// ============================================================================

/// Complete quality report for the stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackQualityReport {
    /// Timestamp of the report
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Individual component assessments
    pub components: Vec<ComponentQuality>,
    /// Summary statistics
    pub summary: QualitySummary,
    /// Overall Stack Quality Index
    pub stack_quality_index: f64,
    /// Overall grade
    pub overall_grade: QualityGrade,
    /// Whether all components are release-ready
    pub release_ready: bool,
    /// Components that block release
    pub blocked_components: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

impl StackQualityReport {
    /// Create report from component list
    pub fn from_components(components: Vec<ComponentQuality>) -> Self {
        let summary = QualitySummary::from_components(&components);

        // Calculate overall SQI as average of component SQIs
        let sqi = if components.is_empty() {
            0.0
        } else {
            components.iter().map(|c| c.sqi).sum::<f64>() / components.len() as f64
        };

        let grade = QualityGrade::from_sqi(sqi);
        let blocked: Vec<String> = components
            .iter()
            .filter(|c| !c.release_ready)
            .map(|c| c.name.clone())
            .collect();

        let release_ready = blocked.is_empty();
        let recommendations = Self::generate_recommendations(&components, &summary);

        Self {
            timestamp: chrono::Utc::now(),
            components,
            summary,
            stack_quality_index: sqi,
            overall_grade: grade,
            release_ready,
            blocked_components: blocked,
            recommendations,
        }
    }

    /// Generate recommendations based on analysis
    fn generate_recommendations(
        components: &[ComponentQuality],
        summary: &QualitySummary,
    ) -> Vec<String> {
        let mut recs = Vec::new();

        if summary.missing_hero_count > 0 {
            recs.push(format!(
                "Add hero images to {} components",
                summary.missing_hero_count
            ));
        }

        if summary.below_threshold_count > 0 {
            recs.push(format!(
                "Improve quality scores for {} components below A- threshold",
                summary.below_threshold_count
            ));
        }

        // Specific recommendations for blocked components
        for comp in components.iter().filter(|c| !c.release_ready) {
            if comp.rust_score.value < 85 {
                recs.push(format!(
                    "{}: Improve test coverage and documentation",
                    comp.name
                ));
            }
            if !comp.hero_image.valid {
                recs.push(format!("{}: Add hero.png to docs/ directory", comp.name));
            }
        }

        recs
    }

    /// Check if all components meet strict A+ requirement
    pub fn is_all_a_plus(&self) -> bool {
        self.components.iter().all(|c| c.grade.is_a_plus())
    }
}

// ============================================================================
// Quality Checker
// ============================================================================

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

// ============================================================================
// Report Formatting
// ============================================================================

/// Format quality report as text
pub fn format_report_text(report: &StackQualityReport) -> String {
    let mut output = String::new();

    output.push_str("PAIML Stack Quality Matrix\n");
    output.push_str(&"═".repeat(78));
    output.push_str("\n\n");

    // Group by layer
    let mut by_layer: HashMap<StackLayer, Vec<&ComponentQuality>> = HashMap::new();
    for comp in &report.components {
        by_layer.entry(comp.layer).or_default().push(comp);
    }

    // Sort layers
    let layer_order = [
        StackLayer::Compute,
        StackLayer::Ml,
        StackLayer::Training,
        StackLayer::Transpilers,
        StackLayer::Orchestration,
        StackLayer::Quality,
        StackLayer::DataMlops,
        StackLayer::Presentation,
    ];

    for layer in layer_order {
        if let Some(components) = by_layer.get(&layer) {
            output.push_str(&format!("{}\n", layer.display_name()));
            output.push_str(&"─".repeat(78));
            output.push('\n');

            output.push_str(&format!(
                "  {:20} {:8} {:8} {:8} {:6} {:7} {:6}\n",
                "Component", "Rust", "Repo", "README", "Hero", "SQI", "Grade"
            ));
            output.push_str(&format!(
                "  {:20} {:8} {:8} {:8} {:6} {:7} {:6}\n",
                "─".repeat(20),
                "─".repeat(8),
                "─".repeat(8),
                "─".repeat(8),
                "─".repeat(6),
                "─".repeat(7),
                "─".repeat(6)
            ));

            for comp in components {
                let hero_status = if comp.hero_image.valid { "✓" } else { "✗" };
                output.push_str(&format!(
                    "  {:20} {:>3}/{:<4} {:>3}/{:<4} {:>2}/{:<4} {:^6} {:>6.1} {} {}\n",
                    comp.name,
                    comp.rust_score.value,
                    comp.rust_score.max,
                    comp.repo_score.value,
                    comp.repo_score.max,
                    comp.readme_score.value,
                    comp.readme_score.max,
                    hero_status,
                    comp.sqi,
                    comp.grade.symbol(),
                    comp.grade.icon(),
                ));

                // Show issues
                for issue in &comp.issues {
                    let icon = match issue.severity {
                        IssueSeverity::Error => "└── ❌",
                        IssueSeverity::Warning => "└── ⚠️",
                        IssueSeverity::Info => "└── ℹ️",
                    };
                    output.push_str(&format!("    {} {}\n", icon, issue.message));
                }
            }
            output.push('\n');
        }
    }

    // Summary
    output.push_str(&"═".repeat(78));
    output.push_str("\nSUMMARY\n");
    output.push_str(&"═".repeat(78));
    output.push_str("\n\n");

    output.push_str(&format!(
        "Quality Distribution:\n  A+  {:3} components ({:.0}%)\n  A   {:3} components ({:.0}%)\n  A-  {:3} components ({:.0}%)\n  <A- {:3} components ({:.0}%)\n\n",
        report.summary.a_plus_count,
        (report.summary.a_plus_count as f64 / report.summary.total_components as f64) * 100.0,
        report.summary.a_count,
        (report.summary.a_count as f64 / report.summary.total_components as f64) * 100.0,
        report.summary.a_minus_count,
        (report.summary.a_minus_count as f64 / report.summary.total_components as f64) * 100.0,
        report.summary.below_threshold_count,
        (report.summary.below_threshold_count as f64 / report.summary.total_components as f64) * 100.0,
    ));

    output.push_str(&format!(
        "Stack Quality Index: {:.1} ({})\n\n",
        report.stack_quality_index, report.overall_grade
    ));

    if report.release_ready {
        output.push_str("Release Status: ✅ READY\n");
    } else {
        output.push_str("Release Status: ❌ BLOCKED\n");
        output.push_str(&format!(
            "  Blocked components: {}\n",
            report.blocked_components.join(", ")
        ));
    }

    if !report.recommendations.is_empty() {
        output.push_str("\nRecommended Actions:\n");
        for (i, rec) in report.recommendations.iter().enumerate() {
            output.push_str(&format!("  {}. {}\n", i + 1, rec));
        }
    }

    output
}

/// Format quality report as JSON
pub fn format_report_json(report: &StackQualityReport) -> Result<String> {
    serde_json::to_string_pretty(report).map_err(|e| anyhow!("JSON serialization error: {}", e))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // QualityGrade Tests - RED PHASE (Issue: STACK-QUALITY-MATRIX)
    // ========================================================================

    #[test]
    fn test_quality_grade_from_rust_project_score_a_plus() {
        // A+ range: 105-114
        assert_eq!(
            QualityGrade::from_rust_project_score(114),
            QualityGrade::APlus
        );
        assert_eq!(
            QualityGrade::from_rust_project_score(110),
            QualityGrade::APlus
        );
        assert_eq!(
            QualityGrade::from_rust_project_score(105),
            QualityGrade::APlus
        );
    }

    #[test]
    fn test_quality_grade_from_rust_project_score_a() {
        // A range: 95-104
        assert_eq!(QualityGrade::from_rust_project_score(104), QualityGrade::A);
        assert_eq!(QualityGrade::from_rust_project_score(100), QualityGrade::A);
        assert_eq!(QualityGrade::from_rust_project_score(95), QualityGrade::A);
    }

    #[test]
    fn test_quality_grade_from_rust_project_score_a_minus() {
        // A- range: 85-94 (PMAT minimum)
        assert_eq!(
            QualityGrade::from_rust_project_score(94),
            QualityGrade::AMinus
        );
        assert_eq!(
            QualityGrade::from_rust_project_score(90),
            QualityGrade::AMinus
        );
        assert_eq!(
            QualityGrade::from_rust_project_score(85),
            QualityGrade::AMinus
        );
    }

    #[test]
    fn test_quality_grade_from_rust_project_score_below() {
        assert_eq!(
            QualityGrade::from_rust_project_score(84),
            QualityGrade::BPlus
        );
        assert_eq!(QualityGrade::from_rust_project_score(75), QualityGrade::B);
        assert_eq!(QualityGrade::from_rust_project_score(65), QualityGrade::C);
        assert_eq!(QualityGrade::from_rust_project_score(55), QualityGrade::D);
        assert_eq!(QualityGrade::from_rust_project_score(40), QualityGrade::F);
    }

    #[test]
    fn test_quality_grade_from_repo_score_a_plus() {
        // A+ range: 95-110
        assert_eq!(QualityGrade::from_repo_score(110), QualityGrade::APlus);
        assert_eq!(QualityGrade::from_repo_score(100), QualityGrade::APlus);
        assert_eq!(QualityGrade::from_repo_score(95), QualityGrade::APlus);
    }

    #[test]
    fn test_quality_grade_from_readme_score() {
        assert_eq!(QualityGrade::from_readme_score(20), QualityGrade::APlus);
        assert_eq!(QualityGrade::from_readme_score(18), QualityGrade::APlus);
        assert_eq!(QualityGrade::from_readme_score(17), QualityGrade::A);
        assert_eq!(QualityGrade::from_readme_score(15), QualityGrade::AMinus);
        assert_eq!(QualityGrade::from_readme_score(12), QualityGrade::BPlus);
    }

    #[test]
    fn test_quality_grade_from_sqi() {
        assert_eq!(QualityGrade::from_sqi(98.5), QualityGrade::APlus);
        assert_eq!(QualityGrade::from_sqi(92.0), QualityGrade::A);
        assert_eq!(QualityGrade::from_sqi(87.0), QualityGrade::AMinus);
        assert_eq!(QualityGrade::from_sqi(75.0), QualityGrade::B);
    }

    #[test]
    fn test_quality_grade_is_release_ready() {
        assert!(QualityGrade::APlus.is_release_ready());
        assert!(QualityGrade::A.is_release_ready());
        assert!(QualityGrade::AMinus.is_release_ready());
        assert!(!QualityGrade::BPlus.is_release_ready());
        assert!(!QualityGrade::B.is_release_ready());
        assert!(!QualityGrade::F.is_release_ready());
    }

    #[test]
    fn test_quality_grade_is_a_plus() {
        assert!(QualityGrade::APlus.is_a_plus());
        assert!(!QualityGrade::A.is_a_plus());
        assert!(!QualityGrade::AMinus.is_a_plus());
    }

    #[test]
    fn test_quality_grade_symbol() {
        assert_eq!(QualityGrade::APlus.symbol(), "A+");
        assert_eq!(QualityGrade::A.symbol(), "A");
        assert_eq!(QualityGrade::AMinus.symbol(), "A-");
        assert_eq!(QualityGrade::BPlus.symbol(), "B+");
        assert_eq!(QualityGrade::F.symbol(), "F");
    }

    #[test]
    fn test_quality_grade_icon() {
        assert_eq!(QualityGrade::APlus.icon(), "✅");
        assert_eq!(QualityGrade::A.icon(), "⚠️");
        assert_eq!(QualityGrade::AMinus.icon(), "⚠️");
        assert_eq!(QualityGrade::BPlus.icon(), "❌");
    }

    #[test]
    fn test_quality_grade_display() {
        assert_eq!(format!("{}", QualityGrade::APlus), "A+");
        assert_eq!(format!("{}", QualityGrade::AMinus), "A-");
    }

    // ========================================================================
    // Score Tests
    // ========================================================================

    #[test]
    fn test_score_percentage() {
        let score = Score::new(85, 100, QualityGrade::AMinus);
        assert!((score.percentage() - 85.0).abs() < f64::EPSILON);

        let score_zero_max = Score::new(0, 0, QualityGrade::F);
        assert!((score_zero_max.percentage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_normalized() {
        let score = Score::new(105, 114, QualityGrade::APlus);
        let normalized = score.normalized();
        assert!(normalized > 90.0 && normalized < 93.0);
    }

    // ========================================================================
    // ImageFormat Tests
    // ========================================================================

    #[test]
    fn test_image_format_from_extension() {
        assert_eq!(ImageFormat::from_extension("png"), Some(ImageFormat::Png));
        assert_eq!(ImageFormat::from_extension("PNG"), Some(ImageFormat::Png));
        assert_eq!(ImageFormat::from_extension("jpg"), Some(ImageFormat::Jpg));
        assert_eq!(ImageFormat::from_extension("jpeg"), Some(ImageFormat::Jpg));
        assert_eq!(ImageFormat::from_extension("webp"), Some(ImageFormat::WebP));
        assert_eq!(ImageFormat::from_extension("svg"), Some(ImageFormat::Svg));
        assert_eq!(ImageFormat::from_extension("gif"), None);
    }

    #[test]
    fn test_image_format_extension() {
        assert_eq!(ImageFormat::Png.extension(), "png");
        assert_eq!(ImageFormat::Jpg.extension(), "jpg");
        assert_eq!(ImageFormat::WebP.extension(), "webp");
        assert_eq!(ImageFormat::Svg.extension(), "svg");
    }

    // ========================================================================
    // HeroImageResult Tests
    // ========================================================================

    #[test]
    fn test_hero_image_missing() {
        let result = HeroImageResult::missing();
        assert!(!result.present);
        assert!(!result.valid);
        assert!(result.path.is_none());
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_hero_image_found() {
        let result = HeroImageResult::found(PathBuf::from("docs/hero.png"), ImageFormat::Png);
        assert!(result.present);
        assert!(result.valid);
        assert_eq!(result.path, Some(PathBuf::from("docs/hero.png")));
        assert_eq!(result.format, Some(ImageFormat::Png));
    }

    #[test]
    fn test_hero_image_detect_missing() {
        let temp_dir = std::env::temp_dir().join("test_hero_missing");
        let _ = std::fs::create_dir_all(&temp_dir);

        let result = HeroImageResult::detect(&temp_dir);
        assert!(!result.present);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // ========================================================================
    // QualityIssue Tests
    // ========================================================================

    #[test]
    fn test_quality_issue_creation() {
        let issue = QualityIssue::new("test_issue", "Test message", IssueSeverity::Error);
        assert_eq!(issue.issue_type, "test_issue");
        assert_eq!(issue.message, "Test message");
        assert_eq!(issue.severity, IssueSeverity::Error);
        assert!(issue.recommendation.is_none());
    }

    #[test]
    fn test_quality_issue_with_recommendation() {
        let issue =
            QualityIssue::new("test", "msg", IssueSeverity::Warning).with_recommendation("Fix it");
        assert_eq!(issue.recommendation, Some("Fix it".to_string()));
    }

    #[test]
    fn test_quality_issue_score_below_threshold() {
        let issue = QualityIssue::score_below_threshold("rust_project", 80, 85);
        assert!(issue.message.contains("80"));
        assert!(issue.message.contains("85"));
        assert_eq!(issue.severity, IssueSeverity::Error);
    }

    #[test]
    fn test_quality_issue_missing_hero() {
        let issue = QualityIssue::missing_hero_image();
        assert!(issue.message.contains("hero"));
        assert!(issue.recommendation.is_some());
    }

    // ========================================================================
    // StackLayer Tests
    // ========================================================================

    #[test]
    fn test_stack_layer_from_component() {
        assert_eq!(StackLayer::from_component("trueno"), StackLayer::Compute);
        assert_eq!(
            StackLayer::from_component("trueno-viz"),
            StackLayer::Compute
        );
        assert_eq!(StackLayer::from_component("aprender"), StackLayer::Ml);
        assert_eq!(
            StackLayer::from_component("depyler"),
            StackLayer::Transpilers
        );
        assert_eq!(
            StackLayer::from_component("batuta"),
            StackLayer::Orchestration
        );
        assert_eq!(StackLayer::from_component("certeza"), StackLayer::Quality);
        assert_eq!(
            StackLayer::from_component("alimentar"),
            StackLayer::DataMlops
        );
        assert_eq!(
            StackLayer::from_component("presentar"),
            StackLayer::Presentation
        );
    }

    #[test]
    fn test_stack_layer_display_name() {
        assert_eq!(StackLayer::Compute.display_name(), "COMPUTE PRIMITIVES");
        assert_eq!(StackLayer::Ml.display_name(), "ML ALGORITHMS");
    }

    // ========================================================================
    // ComponentQuality Tests
    // ========================================================================

    #[test]
    fn test_component_quality_sqi_calculation() {
        // Perfect scores
        let rust = Score::new(114, 114, QualityGrade::APlus);
        let repo = Score::new(110, 110, QualityGrade::APlus);
        let readme = Score::new(20, 20, QualityGrade::APlus);
        let hero = HeroImageResult::found(PathBuf::from("hero.png"), ImageFormat::Png);

        let sqi = ComponentQuality::calculate_sqi(&rust, &repo, &readme, &hero);
        assert!((sqi - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_component_quality_sqi_with_missing_hero() {
        let rust = Score::new(114, 114, QualityGrade::APlus);
        let repo = Score::new(110, 110, QualityGrade::APlus);
        let readme = Score::new(20, 20, QualityGrade::APlus);
        let hero = HeroImageResult::missing();

        let sqi = ComponentQuality::calculate_sqi(&rust, &repo, &readme, &hero);
        // Should be 90% of perfect (missing 10% for hero)
        assert!((sqi - 90.0).abs() < 0.1);
    }

    #[test]
    fn test_component_quality_creation() {
        let rust = Score::new(107, 114, QualityGrade::APlus);
        let repo = Score::new(98, 110, QualityGrade::APlus);
        let readme = Score::new(20, 20, QualityGrade::APlus);
        let hero = HeroImageResult::found(PathBuf::from("hero.png"), ImageFormat::Png);

        let quality = ComponentQuality::new(
            "trueno",
            PathBuf::from("/path/to/trueno"),
            rust,
            repo,
            readme,
            hero,
        );

        assert_eq!(quality.name, "trueno");
        assert_eq!(quality.layer, StackLayer::Compute);
        assert!(quality.sqi > 90.0);
        assert!(quality.grade.is_release_ready());
        assert!(quality.release_ready);
    }

    #[test]
    fn test_component_quality_not_release_ready() {
        let rust = Score::new(70, 114, QualityGrade::B);
        let repo = Score::new(75, 110, QualityGrade::B);
        let readme = Score::new(12, 20, QualityGrade::BPlus);
        let hero = HeroImageResult::missing();

        let quality = ComponentQuality::new(
            "weak-crate",
            PathBuf::from("/path"),
            rust,
            repo,
            readme,
            hero,
        );

        assert!(!quality.release_ready);
        assert!(!quality.grade.is_release_ready());
        assert!(!quality.issues.is_empty());
    }

    // ========================================================================
    // QualitySummary Tests
    // ========================================================================

    #[test]
    fn test_quality_summary_empty() {
        let summary = QualitySummary::from_components(&[]);
        assert_eq!(summary.total_components, 0);
        assert_eq!(summary.a_plus_count, 0);
    }

    #[test]
    fn test_quality_summary_calculation() {
        // A+ component: SQI > 95
        // rust=107/114=93.9%, repo=98/110=89.1%, readme=20/20=100%, hero=100%
        // SQI = 0.4*93.9 + 0.3*89.1 + 0.2*100 + 0.1*100 = 94.3 (A)
        // Need higher scores for A+:
        // rust=112/114=98.2%, repo=105/110=95.5%, readme=20/20=100%, hero=100%
        // SQI = 0.4*98.2 + 0.3*95.5 + 0.2*100 + 0.1*100 = 97.9 (A+)
        let components = vec![
            create_test_component("a", 112, 105, 20, true), // A+ (SQI ~97.9)
            create_test_component("b", 100, 92, 17, true),  // A- (SQI ~87.2)
            create_test_component("c", 80, 75, 12, false),  // B (SQI ~68.5)
        ];

        let summary = QualitySummary::from_components(&components);

        assert_eq!(summary.total_components, 3);
        assert_eq!(summary.a_plus_count, 1);
        assert_eq!(summary.a_minus_count, 1);
        assert_eq!(summary.below_threshold_count, 1);
        assert_eq!(summary.missing_hero_count, 1);
    }

    // ========================================================================
    // StackQualityReport Tests
    // ========================================================================

    #[test]
    fn test_stack_quality_report_creation() {
        let components = vec![
            create_test_component("trueno", 107, 98, 20, true),
            create_test_component("aprender", 105, 95, 18, true),
        ];

        let report = StackQualityReport::from_components(components);

        assert_eq!(report.summary.total_components, 2);
        assert!(report.release_ready);
        assert!(report.blocked_components.is_empty());
        assert!(report.stack_quality_index > 90.0);
    }

    #[test]
    fn test_stack_quality_report_with_blocked() {
        let components = vec![
            create_test_component("trueno", 107, 98, 20, true),
            create_test_component("weak", 70, 70, 10, false),
        ];

        let report = StackQualityReport::from_components(components);

        assert!(!report.release_ready);
        assert_eq!(report.blocked_components, vec!["weak".to_string()]);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_stack_quality_report_is_all_a_plus() {
        // Both components need SQI >= 95 for A+
        // rust=114/114=100%, repo=110/110=100%, readme=20/20=100%, hero=100%
        // SQI = 0.4*100 + 0.3*100 + 0.2*100 + 0.1*100 = 100 (A+)
        let components = vec![
            create_test_component("a", 114, 110, 20, true), // Perfect A+
            create_test_component("b", 112, 105, 20, true), // A+ (SQI ~97.9)
        ];

        let report = StackQualityReport::from_components(components);
        assert!(report.is_all_a_plus());
    }

    #[test]
    fn test_stack_quality_report_not_all_a_plus() {
        let components = vec![
            create_test_component("a", 110, 100, 20, true),
            create_test_component("b", 95, 90, 16, true), // A, not A+
        ];

        let report = StackQualityReport::from_components(components);
        assert!(!report.is_all_a_plus());
    }

    // ========================================================================
    // Helper Functions
    // ========================================================================

    fn create_test_component(
        name: &str,
        rust: u32,
        repo: u32,
        readme: u32,
        has_hero: bool,
    ) -> ComponentQuality {
        let rust_score = Score::new(rust, 114, QualityGrade::from_rust_project_score(rust));
        let repo_score = Score::new(repo, 110, QualityGrade::from_repo_score(repo));
        let readme_score = Score::new(readme, 20, QualityGrade::from_readme_score(readme));
        let hero = if has_hero {
            HeroImageResult::found(PathBuf::from("hero.png"), ImageFormat::Png)
        } else {
            HeroImageResult::missing()
        };

        ComponentQuality::new(
            name,
            PathBuf::from("/test"),
            rust_score,
            repo_score,
            readme_score,
            hero,
        )
    }
}
