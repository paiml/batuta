use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Workflow phase in the 5-phase Batuta pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkflowPhase {
    Analysis,
    Transpilation,
    Optimization,
    Validation,
    Deployment,
}

impl WorkflowPhase {
    /// Get the next phase in the workflow
    pub fn next(&self) -> Option<WorkflowPhase> {
        match self {
            WorkflowPhase::Analysis => Some(WorkflowPhase::Transpilation),
            WorkflowPhase::Transpilation => Some(WorkflowPhase::Optimization),
            WorkflowPhase::Optimization => Some(WorkflowPhase::Validation),
            WorkflowPhase::Validation => Some(WorkflowPhase::Deployment),
            WorkflowPhase::Deployment => None,
        }
    }

    /// Get all phases in order
    pub fn all() -> Vec<WorkflowPhase> {
        vec![
            WorkflowPhase::Analysis,
            WorkflowPhase::Transpilation,
            WorkflowPhase::Optimization,
            WorkflowPhase::Validation,
            WorkflowPhase::Deployment,
        ]
    }
}

impl std::fmt::Display for WorkflowPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkflowPhase::Analysis => write!(f, "Analysis"),
            WorkflowPhase::Transpilation => write!(f, "Transpilation"),
            WorkflowPhase::Optimization => write!(f, "Optimization"),
            WorkflowPhase::Validation => write!(f, "Validation"),
            WorkflowPhase::Deployment => write!(f, "Deployment"),
        }
    }
}

/// Status of a workflow phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhaseStatus {
    NotStarted,
    InProgress,
    Completed,
    Failed,
}

impl std::fmt::Display for PhaseStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhaseStatus::NotStarted => write!(f, "Not Started"),
            PhaseStatus::InProgress => write!(f, "In Progress"),
            PhaseStatus::Completed => write!(f, "Completed"),
            PhaseStatus::Failed => write!(f, "Failed"),
        }
    }
}

/// Information about a single phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseInfo {
    pub phase: WorkflowPhase,
    pub status: PhaseStatus,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub error: Option<String>,
}

impl PhaseInfo {
    pub fn new(phase: WorkflowPhase) -> Self {
        Self {
            phase,
            status: PhaseStatus::NotStarted,
            started_at: None,
            completed_at: None,
            error: None,
        }
    }

    pub fn start(&mut self) {
        self.status = PhaseStatus::InProgress;
        self.started_at = Some(chrono::Utc::now());
    }

    pub fn complete(&mut self) {
        self.status = PhaseStatus::Completed;
        self.completed_at = Some(chrono::Utc::now());
    }

    pub fn fail(&mut self, error: String) {
        self.status = PhaseStatus::Failed;
        self.error = Some(error);
        self.completed_at = Some(chrono::Utc::now());
    }
}

/// Complete workflow state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowState {
    pub current_phase: Option<WorkflowPhase>,
    pub phases: std::collections::HashMap<WorkflowPhase, PhaseInfo>,
}

impl WorkflowState {
    pub fn new() -> Self {
        let mut phases = std::collections::HashMap::new();
        for phase in WorkflowPhase::all() {
            phases.insert(phase, PhaseInfo::new(phase));
        }

        Self {
            current_phase: None,
            phases,
        }
    }

    /// Load workflow state from file
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }

        let content = std::fs::read_to_string(path)?;
        let state = serde_json::from_str(&content)?;
        Ok(state)
    }

    /// Save workflow state to file
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Start a phase
    pub fn start_phase(&mut self, phase: WorkflowPhase) {
        self.current_phase = Some(phase);
        if let Some(info) = self.phases.get_mut(&phase) {
            info.start();
        }
    }

    /// Complete the current phase
    pub fn complete_phase(&mut self, phase: WorkflowPhase) {
        if let Some(info) = self.phases.get_mut(&phase) {
            info.complete();
        }

        // Move to next phase if available
        if let Some(next) = phase.next() {
            self.current_phase = Some(next);
        } else {
            self.current_phase = None;
        }
    }

    /// Fail the current phase
    pub fn fail_phase(&mut self, phase: WorkflowPhase, error: String) {
        if let Some(info) = self.phases.get_mut(&phase) {
            info.fail(error);
        }
        self.current_phase = Some(phase);
    }

    /// Get status of a specific phase
    pub fn get_phase_status(&self, phase: WorkflowPhase) -> PhaseStatus {
        self.phases
            .get(&phase)
            .map(|info| info.status)
            .unwrap_or(PhaseStatus::NotStarted)
    }

    /// Check if a phase is completed
    pub fn is_phase_completed(&self, phase: WorkflowPhase) -> bool {
        self.get_phase_status(phase) == PhaseStatus::Completed
    }

    /// Get overall progress percentage
    pub fn progress_percentage(&self) -> f64 {
        let total = WorkflowPhase::all().len() as f64;
        let completed = self
            .phases
            .values()
            .filter(|info| info.status == PhaseStatus::Completed)
            .count() as f64;

        (completed / total) * 100.0
    }
}

/// Programming language detected in the project
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    Python,
    C,
    Cpp,
    Rust,
    Shell,
    JavaScript,
    TypeScript,
    Go,
    Java,
    Other(String),
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::Python => write!(f, "Python"),
            Language::C => write!(f, "C"),
            Language::Cpp => write!(f, "C++"),
            Language::Rust => write!(f, "Rust"),
            Language::Shell => write!(f, "Shell"),
            Language::JavaScript => write!(f, "JavaScript"),
            Language::TypeScript => write!(f, "TypeScript"),
            Language::Go => write!(f, "Go"),
            Language::Java => write!(f, "Java"),
            Language::Other(name) => write!(f, "{}", name),
        }
    }
}

/// Dependency manager type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyManager {
    Pip,           // requirements.txt
    Pipenv,        // Pipfile
    Poetry,        // pyproject.toml
    Conda,         // environment.yml
    Cargo,         // Cargo.toml
    Npm,           // package.json
    Yarn,          // yarn.lock
    GoMod,         // go.mod
    Maven,         // pom.xml
    Gradle,        // build.gradle
    Make,          // Makefile
}

impl std::fmt::Display for DependencyManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DependencyManager::Pip => write!(f, "pip (requirements.txt)"),
            DependencyManager::Pipenv => write!(f, "Pipenv"),
            DependencyManager::Poetry => write!(f, "Poetry"),
            DependencyManager::Conda => write!(f, "Conda"),
            DependencyManager::Cargo => write!(f, "Cargo"),
            DependencyManager::Npm => write!(f, "npm"),
            DependencyManager::Yarn => write!(f, "Yarn"),
            DependencyManager::GoMod => write!(f, "Go modules"),
            DependencyManager::Maven => write!(f, "Maven"),
            DependencyManager::Gradle => write!(f, "Gradle"),
            DependencyManager::Make => write!(f, "Make"),
        }
    }
}

/// Information about detected dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    pub manager: DependencyManager,
    pub file_path: PathBuf,
    pub count: Option<usize>,
}

/// Language statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageStats {
    pub language: Language,
    pub file_count: usize,
    pub line_count: usize,
    pub percentage: f64,
}

/// Complete project analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectAnalysis {
    pub root_path: PathBuf,
    pub languages: Vec<LanguageStats>,
    pub primary_language: Option<Language>,
    pub dependencies: Vec<DependencyInfo>,
    pub total_files: usize,
    pub total_lines: usize,
    pub tdg_score: Option<f64>,
}

impl ProjectAnalysis {
    pub fn new(root_path: PathBuf) -> Self {
        Self {
            root_path,
            languages: Vec::new(),
            primary_language: None,
            dependencies: Vec::new(),
            total_files: 0,
            total_lines: 0,
            tdg_score: None,
        }
    }

    pub fn recommend_transpiler(&self) -> Option<&'static str> {
        match self.primary_language.as_ref()? {
            Language::Python => Some("Depyler (Python → Rust)"),
            Language::C | Language::Cpp => Some("Decy (C/C++ → Rust)"),
            Language::Shell => Some("Bashrs (Shell → Rust)"),
            Language::Rust => Some("Already Rust! Consider Ruchy for gradual typing."),
            _ => None,
        }
    }

    pub fn has_ml_dependencies(&self) -> bool {
        // Check if project uses ML frameworks
        self.dependencies.iter().any(|dep| {
            matches!(
                dep.manager,
                DependencyManager::Pip | DependencyManager::Conda | DependencyManager::Poetry
            )
        })
    }
}
