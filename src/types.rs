use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
