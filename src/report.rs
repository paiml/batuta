/// Report generation for migration analysis
use crate::types::{ProjectAnalysis, WorkflowState};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Migration report data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationReport {
    pub project_name: String,
    pub analysis: ProjectAnalysis,
    pub workflow: WorkflowState,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl MigrationReport {
    pub fn new(project_name: String, analysis: ProjectAnalysis, workflow: WorkflowState) -> Self {
        Self {
            project_name,
            analysis,
            workflow,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Generate HTML report
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        // HTML header
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"en\">\n");
        html.push_str("<head>\n");
        html.push_str("  <meta charset=\"UTF-8\">\n");
        html.push_str("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        html.push_str(&format!("  <title>Migration Report - {}</title>\n", self.project_name));
        html.push_str("  <style>\n");
        html.push_str(include_str!("report_style.css"));
        html.push_str("  </style>\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");

        // Header
        html.push_str(&format!(
            "<header><h1>Migration Report: {}</h1><p>Generated: {}</p></header>\n",
            self.project_name,
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        // Summary section
        html.push_str("<section class=\"summary\">\n");
        html.push_str("<h2>Summary</h2>\n");
        html.push_str("<div class=\"stats\">\n");
        html.push_str(&format!(
            "<div class=\"stat\"><span class=\"label\">Total Files</span><span class=\"value\">{}</span></div>\n",
            self.analysis.total_files
        ));
        html.push_str(&format!(
            "<div class=\"stat\"><span class=\"label\">Total Lines</span><span class=\"value\">{}</span></div>\n",
            self.analysis.total_lines
        ));
        if let Some(lang) = &self.analysis.primary_language {
            html.push_str(&format!(
                "<div class=\"stat\"><span class=\"label\">Primary Language</span><span class=\"value\">{}</span></div>\n",
                lang
            ));
        }
        if let Some(score) = self.analysis.tdg_score {
            let grade = get_tdg_grade(score);
            html.push_str(&format!(
                "<div class=\"stat\"><span class=\"label\">TDG Score</span><span class=\"value\">{:.1}/100 ({})</span></div>\n",
                score, grade
            ));
        }
        html.push_str("</div>\n");
        html.push_str("</section>\n");

        // Languages section
        if !self.analysis.languages.is_empty() {
            html.push_str("<section class=\"languages\">\n");
            html.push_str("<h2>Languages</h2>\n");
            html.push_str("<table>\n");
            html.push_str("<thead><tr><th>Language</th><th>Files</th><th>Lines</th><th>Percentage</th></tr></thead>\n");
            html.push_str("<tbody>\n");
            for lang_stat in &self.analysis.languages {
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.1}%</td></tr>\n",
                    lang_stat.language,
                    lang_stat.file_count,
                    lang_stat.line_count,
                    lang_stat.percentage
                ));
            }
            html.push_str("</tbody>\n");
            html.push_str("</table>\n");
            html.push_str("</section>\n");
        }

        // Dependencies section
        if !self.analysis.dependencies.is_empty() {
            html.push_str("<section class=\"dependencies\">\n");
            html.push_str("<h2>Dependencies</h2>\n");
            html.push_str("<ul>\n");
            for dep in &self.analysis.dependencies {
                let count_str = if let Some(count) = dep.count {
                    format!(" ({} packages)", count)
                } else {
                    String::new()
                };
                html.push_str(&format!(
                    "<li><strong>{}</strong>{} - {:?}</li>\n",
                    dep.manager, count_str, dep.file_path
                ));
            }
            html.push_str("</ul>\n");
            html.push_str("</section>\n");
        }

        // Workflow progress section
        html.push_str("<section class=\"workflow\">\n");
        html.push_str("<h2>Workflow Progress</h2>\n");
        html.push_str(&format!("<p class=\"progress\">{:.0}% complete</p>\n", self.workflow.progress_percentage()));
        html.push_str("<table>\n");
        html.push_str("<thead><tr><th>Phase</th><th>Status</th><th>Started</th><th>Completed</th></tr></thead>\n");
        html.push_str("<tbody>\n");
        for phase in crate::types::WorkflowPhase::all() {
            if let Some(info) = self.workflow.phases.get(&phase) {
                let status_class = match info.status {
                    crate::types::PhaseStatus::Completed => "completed",
                    crate::types::PhaseStatus::InProgress => "in-progress",
                    crate::types::PhaseStatus::Failed => "failed",
                    crate::types::PhaseStatus::NotStarted => "not-started",
                };
                html.push_str(&format!("<tr class=\"{}\">\n", status_class));
                html.push_str(&format!("<td>{}</td>\n", phase));
                html.push_str(&format!("<td>{}</td>\n", info.status));
                html.push_str(&format!(
                    "<td>{}</td>\n",
                    info.started_at
                        .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
                        .unwrap_or_else(|| "-".to_string())
                ));
                html.push_str(&format!(
                    "<td>{}</td>\n",
                    info.completed_at
                        .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
                        .unwrap_or_else(|| "-".to_string())
                ));
                html.push_str("</tr>\n");
            }
        }
        html.push_str("</tbody>\n");
        html.push_str("</table>\n");
        html.push_str("</section>\n");

        // Recommendations section
        html.push_str("<section class=\"recommendations\">\n");
        html.push_str("<h2>Recommendations</h2>\n");
        html.push_str("<ul>\n");
        if let Some(transpiler) = self.analysis.recommend_transpiler() {
            html.push_str(&format!("<li>Use <strong>{}</strong> for transpilation</li>\n", transpiler));
        }
        if self.analysis.has_ml_dependencies() {
            html.push_str("<li>Consider <strong>Aprender</strong> for ML algorithms and <strong>Realizar</strong> for inference</li>\n");
        }
        if let Some(score) = self.analysis.tdg_score {
            if score < 85.0 {
                html.push_str("<li>TDG score below 85 - consider refactoring before migration</li>\n");
            }
        }
        html.push_str("</ul>\n");
        html.push_str("</section>\n");

        // Footer
        html.push_str("<footer>\n");
        html.push_str("<p>Generated by Batuta - Sovereign AI Stack</p>\n");
        html.push_str("<p><a href=\"https://github.com/paiml/Batuta\">github.com/paiml/Batuta</a></p>\n");
        html.push_str("</footer>\n");

        html.push_str("</body>\n");
        html.push_str("</html>\n");

        html
    }

    /// Generate Markdown report
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        // Header
        md.push_str(&format!("# Migration Report: {}\n\n", self.project_name));
        md.push_str(&format!(
            "**Generated:** {}\n\n",
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        // Summary
        md.push_str("## Summary\n\n");
        md.push_str(&format!("- **Total Files:** {}\n", self.analysis.total_files));
        md.push_str(&format!("- **Total Lines:** {}\n", self.analysis.total_lines));
        if let Some(lang) = &self.analysis.primary_language {
            md.push_str(&format!("- **Primary Language:** {}\n", lang));
        }
        if let Some(score) = self.analysis.tdg_score {
            let grade = get_tdg_grade(score);
            md.push_str(&format!("- **TDG Score:** {:.1}/100 ({})\n", score, grade));
        }
        md.push('\n');

        // Languages
        if !self.analysis.languages.is_empty() {
            md.push_str("## Languages\n\n");
            md.push_str("| Language | Files | Lines | Percentage |\n");
            md.push_str("|----------|-------|-------|------------|\n");
            for lang_stat in &self.analysis.languages {
                md.push_str(&format!(
                    "| {} | {} | {} | {:.1}% |\n",
                    lang_stat.language,
                    lang_stat.file_count,
                    lang_stat.line_count,
                    lang_stat.percentage
                ));
            }
            md.push('\n');
        }

        // Dependencies
        if !self.analysis.dependencies.is_empty() {
            md.push_str("## Dependencies\n\n");
            for dep in &self.analysis.dependencies {
                let count_str = if let Some(count) = dep.count {
                    format!(" ({} packages)", count)
                } else {
                    String::new()
                };
                md.push_str(&format!("- **{}**{} - `{:?}`\n", dep.manager, count_str, dep.file_path));
            }
            md.push('\n');
        }

        // Workflow
        md.push_str("## Workflow Progress\n\n");
        md.push_str(&format!("**Overall:** {:.0}% complete\n\n", self.workflow.progress_percentage()));
        md.push_str("| Phase | Status | Started | Completed |\n");
        md.push_str("|-------|--------|---------|----------|\n");
        for phase in crate::types::WorkflowPhase::all() {
            if let Some(info) = self.workflow.phases.get(&phase) {
                md.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    phase,
                    info.status,
                    info.started_at
                        .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_else(|| "-".to_string()),
                    info.completed_at
                        .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_else(|| "-".to_string())
                ));
            }
        }
        md.push('\n');

        // Recommendations
        md.push_str("## Recommendations\n\n");
        if let Some(transpiler) = self.analysis.recommend_transpiler() {
            md.push_str(&format!("- Use **{}** for transpilation\n", transpiler));
        }
        if self.analysis.has_ml_dependencies() {
            md.push_str("- Consider **Aprender** for ML algorithms and **Realizar** for inference\n");
        }
        if let Some(score) = self.analysis.tdg_score {
            if score < 85.0 {
                md.push_str("- TDG score below 85 - consider refactoring before migration\n");
            }
        }
        md.push('\n');

        // Footer
        md.push_str("---\n\n");
        md.push_str("*Generated by Batuta - Sovereign AI Stack*  \n");
        md.push_str("https://github.com/paiml/Batuta\n");

        md
    }

    /// Generate JSON report
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Generate plain text report
    pub fn to_text(&self) -> String {
        let mut text = String::new();

        // Header
        text.push_str(&format!("MIGRATION REPORT: {}\n", self.project_name));
        text.push_str(&format!(
            "Generated: {}\n",
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        text.push_str(&"=".repeat(80));
        text.push_str("\n\n");

        // Summary
        text.push_str("SUMMARY\n");
        text.push_str(&"-".repeat(80));
        text.push('\n');
        text.push_str(&format!("Total Files: {}\n", self.analysis.total_files));
        text.push_str(&format!("Total Lines: {}\n", self.analysis.total_lines));
        if let Some(lang) = &self.analysis.primary_language {
            text.push_str(&format!("Primary Language: {}\n", lang));
        }
        if let Some(score) = self.analysis.tdg_score {
            let grade = get_tdg_grade(score);
            text.push_str(&format!("TDG Score: {:.1}/100 ({})\n", score, grade));
        }
        text.push('\n');

        // Languages
        if !self.analysis.languages.is_empty() {
            text.push_str("LANGUAGES\n");
            text.push_str(&"-".repeat(80));
            text.push('\n');
            for lang_stat in &self.analysis.languages {
                text.push_str(&format!(
                    "{:15} {:8} files  {:10} lines  {:5.1}%\n",
                    format!("{}", lang_stat.language),
                    lang_stat.file_count,
                    lang_stat.line_count,
                    lang_stat.percentage
                ));
            }
            text.push('\n');
        }

        // Dependencies
        if !self.analysis.dependencies.is_empty() {
            text.push_str("DEPENDENCIES\n");
            text.push_str(&"-".repeat(80));
            text.push('\n');
            for dep in &self.analysis.dependencies {
                let count_str = if let Some(count) = dep.count {
                    format!(" ({} packages)", count)
                } else {
                    String::new()
                };
                text.push_str(&format!("{}{}\n", dep.manager, count_str));
                text.push_str(&format!("  File: {:?}\n", dep.file_path));
            }
            text.push('\n');
        }

        // Workflow
        text.push_str("WORKFLOW PROGRESS\n");
        text.push_str(&"-".repeat(80));
        text.push('\n');
        text.push_str(&format!("Overall: {:.0}% complete\n\n", self.workflow.progress_percentage()));
        for phase in crate::types::WorkflowPhase::all() {
            if let Some(info) = self.workflow.phases.get(&phase) {
                text.push_str(&format!("{:15} {:12}", format!("{}", phase), format!("{}", info.status)));
                if let Some(started) = info.started_at {
                    text.push_str(&format!("  Started: {}", started.format("%Y-%m-%d %H:%M")));
                }
                if let Some(completed) = info.completed_at {
                    text.push_str(&format!("  Completed: {}", completed.format("%Y-%m-%d %H:%M")));
                }
                text.push('\n');
            }
        }
        text.push('\n');

        // Recommendations
        text.push_str("RECOMMENDATIONS\n");
        text.push_str(&"-".repeat(80));
        text.push('\n');
        if let Some(transpiler) = self.analysis.recommend_transpiler() {
            text.push_str(&format!("• Use {} for transpilation\n", transpiler));
        }
        if self.analysis.has_ml_dependencies() {
            text.push_str("• Consider Aprender for ML algorithms and Realizar for inference\n");
        }
        if let Some(score) = self.analysis.tdg_score {
            if score < 85.0 {
                text.push_str("• TDG score below 85 - consider refactoring before migration\n");
            }
        }
        text.push('\n');

        text
    }

    /// Save report to file
    pub fn save(&self, path: &Path, format: ReportFormat) -> Result<()> {
        let content = match format {
            ReportFormat::Html => self.to_html(),
            ReportFormat::Markdown => self.to_markdown(),
            ReportFormat::Json => self.to_json()?,
            ReportFormat::Text => self.to_text(),
        };

        std::fs::write(path, content)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    Html,
    Markdown,
    Json,
    Text,
}

fn get_tdg_grade(score: f64) -> &'static str {
    if score >= 95.0 {
        "A+"
    } else if score >= 90.0 {
        "A"
    } else if score >= 85.0 {
        "B+"
    } else if score >= 80.0 {
        "B"
    } else if score >= 70.0 {
        "C"
    } else {
        "D"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DependencyInfo, DependencyManager, Language, LanguageStats};
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_analysis() -> ProjectAnalysis {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test/project"));
        analysis.total_files = 50;
        analysis.total_lines = 5000;
        analysis.primary_language = Some(Language::Python);
        analysis.tdg_score = Some(87.5);

        analysis.languages.push(LanguageStats {
            language: Language::Python,
            file_count: 40,
            line_count: 4000,
            percentage: 80.0,
        });
        analysis.languages.push(LanguageStats {
            language: Language::Rust,
            file_count: 10,
            line_count: 1000,
            percentage: 20.0,
        });

        analysis.dependencies.push(DependencyInfo {
            manager: DependencyManager::Pip,
            file_path: PathBuf::from("requirements.txt"),
            count: Some(25),
        });

        analysis
    }

    fn create_test_workflow() -> WorkflowState {
        let mut workflow = WorkflowState::new();
        workflow.start_phase(crate::types::WorkflowPhase::Analysis);
        workflow.complete_phase(crate::types::WorkflowPhase::Analysis);
        workflow.start_phase(crate::types::WorkflowPhase::Transpilation);
        workflow
    }

    // ============================================================================
    // MIGRATION REPORT TESTS
    // ============================================================================

    #[test]
    fn test_migration_report_new() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();

        let report = MigrationReport::new("TestProject".to_string(), analysis.clone(), workflow.clone());

        assert_eq!(report.project_name, "TestProject");
        assert_eq!(report.analysis.total_files, 50);
        assert_eq!(report.workflow.progress_percentage(), 20.0);
    }

    #[test]
    fn test_migration_report_clone() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report1 = MigrationReport::new("Test".to_string(), analysis, workflow);

        let report2 = report1.clone();
        assert_eq!(report1.project_name, report2.project_name);
        assert_eq!(report1.analysis.total_files, report2.analysis.total_files);
    }

    // ============================================================================
    // HTML REPORT TESTS
    // ============================================================================

    #[test]
    fn test_to_html_contains_header() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let html = report.to_html();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<html lang=\"en\">"));
        assert!(html.contains("TestProject"));
    }

    #[test]
    fn test_to_html_contains_summary() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let html = report.to_html();

        assert!(html.contains("Summary"));
        assert!(html.contains("50")); // total_files
        assert!(html.contains("5000")); // total_lines
        assert!(html.contains("Python"));
    }

    #[test]
    fn test_to_html_contains_tdg_score() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let html = report.to_html();

        assert!(html.contains("TDG Score"));
        assert!(html.contains("87.5"));
        assert!(html.contains("B+"));
    }

    #[test]
    fn test_to_html_contains_languages_table() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let html = report.to_html();

        assert!(html.contains("<table>"));
        assert!(html.contains("Languages"));
        assert!(html.contains("Python"));
        assert!(html.contains("Rust"));
        assert!(html.contains("80.0%"));
    }

    #[test]
    fn test_to_html_contains_dependencies() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let html = report.to_html();

        assert!(html.contains("Dependencies"));
        assert!(html.contains("pip"));
        assert!(html.contains("25 packages"));
    }

    #[test]
    fn test_to_html_contains_workflow() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let html = report.to_html();

        assert!(html.contains("Workflow Progress"));
        assert!(html.contains("20%")); // progress_percentage
        assert!(html.contains("Analysis"));
        assert!(html.contains("Transpilation"));
    }

    #[test]
    fn test_to_html_contains_recommendations() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let html = report.to_html();

        assert!(html.contains("Recommendations"));
        assert!(html.contains("Depyler"));
        assert!(html.contains("Aprender"));
    }

    #[test]
    fn test_to_html_contains_footer() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let html = report.to_html();

        assert!(html.contains("</html>"));
        assert!(html.contains("Batuta"));
        assert!(html.contains("github.com/paiml/Batuta"));
    }

    // ============================================================================
    // MARKDOWN REPORT TESTS
    // ============================================================================

    #[test]
    fn test_to_markdown_contains_header() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let md = report.to_markdown();

        assert!(md.contains("# Migration Report: TestProject"));
        assert!(md.contains("**Generated:**"));
    }

    #[test]
    fn test_to_markdown_contains_summary() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let md = report.to_markdown();

        assert!(md.contains("## Summary"));
        assert!(md.contains("**Total Files:** 50"));
        assert!(md.contains("**Total Lines:** 5000"));
        assert!(md.contains("**Primary Language:** Python"));
    }

    #[test]
    fn test_to_markdown_contains_languages_table() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let md = report.to_markdown();

        assert!(md.contains("## Languages"));
        assert!(md.contains("| Language | Files | Lines | Percentage |"));
        assert!(md.contains("| Python | 40 | 4000 | 80.0% |"));
    }

    #[test]
    fn test_to_markdown_contains_dependencies() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let md = report.to_markdown();

        assert!(md.contains("## Dependencies"));
        assert!(md.contains("**pip (requirements.txt)** (25 packages)"));
    }

    #[test]
    fn test_to_markdown_contains_workflow() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let md = report.to_markdown();

        assert!(md.contains("## Workflow Progress"));
        assert!(md.contains("**Overall:** 20% complete"));
        assert!(md.contains("| Phase | Status | Started | Completed |"));
    }

    #[test]
    fn test_to_markdown_contains_footer() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let md = report.to_markdown();

        assert!(md.contains("---"));
        assert!(md.contains("*Generated by Batuta - Sovereign AI Stack*"));
        assert!(md.contains("https://github.com/paiml/Batuta"));
    }

    // ============================================================================
    // JSON REPORT TESTS
    // ============================================================================

    #[test]
    fn test_to_json_valid() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let json = report.to_json().unwrap();

        assert!(json.contains("\"project_name\""));
        assert!(json.contains("\"TestProject\""));
        assert!(json.contains("\"analysis\""));
        assert!(json.contains("\"workflow\""));
    }

    #[test]
    fn test_to_json_deserialize() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let json = report.to_json().unwrap();
        let deserialized: MigrationReport = serde_json::from_str(&json).unwrap();

        assert_eq!(report.project_name, deserialized.project_name);
        assert_eq!(report.analysis.total_files, deserialized.analysis.total_files);
    }

    // ============================================================================
    // TEXT REPORT TESTS
    // ============================================================================

    #[test]
    fn test_to_text_contains_header() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let text = report.to_text();

        assert!(text.contains("MIGRATION REPORT: TestProject"));
        assert!(text.contains("Generated:"));
        assert!(text.contains("========"));
    }

    #[test]
    fn test_to_text_contains_summary() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let text = report.to_text();

        assert!(text.contains("SUMMARY"));
        assert!(text.contains("Total Files: 50"));
        assert!(text.contains("Total Lines: 5000"));
        assert!(text.contains("Primary Language: Python"));
    }

    #[test]
    fn test_to_text_contains_languages() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let text = report.to_text();

        assert!(text.contains("LANGUAGES"));
        assert!(text.contains("Python"));
        assert!(text.contains("Rust"));
        assert!(text.contains("80.0%"));
    }

    #[test]
    fn test_to_text_contains_workflow() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        let text = report.to_text();

        assert!(text.contains("WORKFLOW PROGRESS"));
        assert!(text.contains("Overall: 20% complete"));
        assert!(text.contains("Analysis"));
    }

    // ============================================================================
    // FILE SAVE TESTS
    // ============================================================================

    #[test]
    fn test_save_html() {
        let temp_dir = TempDir::new().unwrap();
        let report_path = temp_dir.path().join("report.html");

        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        report.save(&report_path, ReportFormat::Html).unwrap();

        assert!(report_path.exists());
        let content = std::fs::read_to_string(&report_path).unwrap();
        assert!(content.contains("<!DOCTYPE html>"));
    }

    #[test]
    fn test_save_markdown() {
        let temp_dir = TempDir::new().unwrap();
        let report_path = temp_dir.path().join("report.md");

        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        report.save(&report_path, ReportFormat::Markdown).unwrap();

        assert!(report_path.exists());
        let content = std::fs::read_to_string(&report_path).unwrap();
        assert!(content.contains("# Migration Report"));
    }

    #[test]
    fn test_save_json() {
        let temp_dir = TempDir::new().unwrap();
        let report_path = temp_dir.path().join("report.json");

        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        report.save(&report_path, ReportFormat::Json).unwrap();

        assert!(report_path.exists());
        let content = std::fs::read_to_string(&report_path).unwrap();
        assert!(content.contains("\"project_name\""));
    }

    #[test]
    fn test_save_text() {
        let temp_dir = TempDir::new().unwrap();
        let report_path = temp_dir.path().join("report.txt");

        let analysis = create_test_analysis();
        let workflow = create_test_workflow();
        let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

        report.save(&report_path, ReportFormat::Text).unwrap();

        assert!(report_path.exists());
        let content = std::fs::read_to_string(&report_path).unwrap();
        assert!(content.contains("MIGRATION REPORT"));
    }

    // ============================================================================
    // HELPER FUNCTION TESTS
    // ============================================================================

    #[test]
    fn test_get_tdg_grade_a_plus() {
        assert_eq!(get_tdg_grade(100.0), "A+");
        assert_eq!(get_tdg_grade(95.0), "A+");
    }

    #[test]
    fn test_get_tdg_grade_a() {
        assert_eq!(get_tdg_grade(94.9), "A");
        assert_eq!(get_tdg_grade(90.0), "A");
    }

    #[test]
    fn test_get_tdg_grade_b_plus() {
        assert_eq!(get_tdg_grade(89.9), "B+");
        assert_eq!(get_tdg_grade(85.0), "B+");
    }

    #[test]
    fn test_get_tdg_grade_b() {
        assert_eq!(get_tdg_grade(84.9), "B");
        assert_eq!(get_tdg_grade(80.0), "B");
    }

    #[test]
    fn test_get_tdg_grade_c() {
        assert_eq!(get_tdg_grade(79.9), "C");
        assert_eq!(get_tdg_grade(70.0), "C");
    }

    #[test]
    fn test_get_tdg_grade_d() {
        assert_eq!(get_tdg_grade(69.9), "D");
        assert_eq!(get_tdg_grade(50.0), "D");
        assert_eq!(get_tdg_grade(0.0), "D");
    }

    // ============================================================================
    // EDGE CASE TESTS
    // ============================================================================

    #[test]
    fn test_report_with_no_languages() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.total_files = 0;
        analysis.total_lines = 0;
        let workflow = WorkflowState::new();

        let report = MigrationReport::new("Empty".to_string(), analysis, workflow);

        let html = report.to_html();
        assert!(html.contains("Empty"));

        let md = report.to_markdown();
        assert!(md.contains("Empty"));

        let text = report.to_text();
        assert!(text.contains("Empty"));
    }

    #[test]
    fn test_report_with_no_dependencies() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.total_files = 10;
        analysis.total_lines = 100;
        let workflow = WorkflowState::new();

        let report = MigrationReport::new("NoDeps".to_string(), analysis, workflow);

        let html = report.to_html();
        let md = report.to_markdown();
        let text = report.to_text();

        // Should not crash and should contain project name
        assert!(html.contains("NoDeps"));
        assert!(md.contains("NoDeps"));
        assert!(text.contains("NoDeps"));
    }

    #[test]
    fn test_report_with_no_tdg_score() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.total_files = 10;
        analysis.tdg_score = None;
        let workflow = WorkflowState::new();

        let report = MigrationReport::new("NoTDG".to_string(), analysis, workflow);

        let html = report.to_html();
        // Should not contain TDG score section
        assert!(!html.contains("TDG Score"));
    }

    #[test]
    fn test_report_format_clone_copy() {
        let format1 = ReportFormat::Html;
        let format2 = format1; // Copy

        // Both should be usable
        let _ = format!("{:?}", format1);
        let _ = format!("{:?}", format2);
    }

    #[test]
    fn test_report_with_high_tdg_score() {
        let mut analysis = create_test_analysis();
        analysis.tdg_score = Some(96.0);
        let workflow = create_test_workflow();

        let report = MigrationReport::new("HighTDG".to_string(), analysis, workflow);

        let html = report.to_html();
        assert!(html.contains("96.0"));
        assert!(html.contains("A+"));

        // Recommendations should not suggest refactoring for high scores
        assert!(!html.contains("consider refactoring before migration"));
    }

    #[test]
    fn test_report_with_low_tdg_score() {
        let mut analysis = create_test_analysis();
        analysis.tdg_score = Some(70.0);
        let workflow = create_test_workflow();

        let report = MigrationReport::new("LowTDG".to_string(), analysis, workflow);

        let html = report.to_html();
        assert!(html.contains("70.0"));
        assert!(html.contains("C"));

        // Recommendations should suggest refactoring for low scores
        assert!(html.contains("consider refactoring before migration"));
    }

    #[test]
    fn test_report_with_ml_dependencies() {
        let analysis = create_test_analysis();
        let workflow = create_test_workflow();

        let report = MigrationReport::new("MLProject".to_string(), analysis, workflow);

        let html = report.to_html();
        // Should recommend ML tools for projects with Pip dependencies
        assert!(html.contains("Aprender"));
        assert!(html.contains("Realizar"));
    }
}
