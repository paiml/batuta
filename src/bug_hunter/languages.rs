//! Multi-Language Pattern Support
//!
//! Provides language-specific bug patterns for Python, TypeScript, Go, and Rust.

use super::types::{DefectCategory, FindingSeverity};

/// Supported programming languages for bug hunting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    Go,
}

impl Language {
    /// Detect language from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Language::Rust),
            "py" => Some(Language::Python),
            "ts" | "tsx" | "js" | "jsx" => Some(Language::TypeScript),
            "go" => Some(Language::Go),
            _ => None,
        }
    }

    /// Get file extensions for this language.
    #[allow(dead_code)]
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Language::Rust => &["rs"],
            Language::Python => &["py"],
            Language::TypeScript => &["ts", "tsx", "js", "jsx"],
            Language::Go => &["go"],
        }
    }

    /// Get glob patterns for this language.
    #[allow(dead_code)]
    pub fn glob_patterns(&self) -> Vec<&'static str> {
        match self {
            Language::Rust => vec!["**/*.rs"],
            Language::Python => vec!["**/*.py"],
            Language::TypeScript => vec!["**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx"],
            Language::Go => vec!["**/*.go"],
        }
    }
}

/// A language-specific pattern.
#[allow(dead_code)]
pub struct LangPattern {
    pub pattern: &'static str,
    pub category: DefectCategory,
    pub severity: FindingSeverity,
    pub suspiciousness: f64,
    pub language: Option<Language>, // None = applies to all languages
}

/// Get patterns applicable to a specific language.
pub fn patterns_for_language(lang: Language) -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    let mut patterns = vec![];

    // Universal patterns (all languages)
    patterns.extend(universal_patterns());

    // Language-specific patterns
    match lang {
        Language::Rust => patterns.extend(rust_patterns()),
        Language::Python => patterns.extend(python_patterns()),
        Language::TypeScript => patterns.extend(typescript_patterns()),
        Language::Go => patterns.extend(go_patterns()),
    }

    patterns
}

/// Patterns that apply to all languages.
fn universal_patterns() -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    vec![
        // Universal debt markers
        ("TODO", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
        ("FIXME", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("HACK", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("XXX", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("BUG", DefectCategory::LogicErrors, FindingSeverity::High, 0.7),
        // Hidden debt euphemisms
        ("placeholder", DefectCategory::HiddenDebt, FindingSeverity::High, 0.75),
        ("stub", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("dummy", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("temporary", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.6),
        ("hardcoded", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.5),
        ("workaround", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.6),
        ("tech debt", DefectCategory::HiddenDebt, FindingSeverity::High, 0.8),
    ]
}

/// Rust-specific patterns.
fn rust_patterns() -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    vec![
        ("unwrap()", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.4),
        ("expect(", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
        ("unsafe {", DefectCategory::MemorySafety, FindingSeverity::High, 0.7),
        ("transmute", DefectCategory::MemorySafety, FindingSeverity::High, 0.8),
        ("panic!", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("unreachable!", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
        ("unimplemented!", DefectCategory::HiddenDebt, FindingSeverity::Critical, 0.9),
        ("todo!", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("#[ignore]", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        (".unwrap_or_else(|_|", DefectCategory::SilentDegradation, FindingSeverity::High, 0.7),
        ("Err(_) => {}", DefectCategory::SilentDegradation, FindingSeverity::High, 0.75),
    ]
}

/// Python-specific patterns.
fn python_patterns() -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    vec![
        // Exception handling
        ("except:", DefectCategory::SilentDegradation, FindingSeverity::High, 0.8),
        ("except Exception:", DefectCategory::SilentDegradation, FindingSeverity::Medium, 0.6),
        ("except BaseException:", DefectCategory::SilentDegradation, FindingSeverity::High, 0.8),
        ("pass  # TODO", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        // Security
        ("eval(", DefectCategory::SecurityVulnerabilities, FindingSeverity::Critical, 0.95),
        ("exec(", DefectCategory::SecurityVulnerabilities, FindingSeverity::Critical, 0.95),
        ("pickle.loads", DefectCategory::SecurityVulnerabilities, FindingSeverity::High, 0.8),
        ("shell=True", DefectCategory::SecurityVulnerabilities, FindingSeverity::High, 0.85),
        ("__import__", DefectCategory::SecurityVulnerabilities, FindingSeverity::Medium, 0.6),
        // Anti-patterns
        ("global ", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("import *", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("assert ", DefectCategory::TestDebt, FindingSeverity::Low, 0.3), // in production code
        ("# type: ignore", DefectCategory::TypeErrors, FindingSeverity::Medium, 0.5),
        // Test debt
        ("@pytest.mark.skip", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        ("@unittest.skip", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        ("raise NotImplementedError", DefectCategory::HiddenDebt, FindingSeverity::High, 0.8),
        // Threading issues
        ("threading.Thread(", DefectCategory::ConcurrencyBugs, FindingSeverity::Medium, 0.5),
    ]
}

/// TypeScript/JavaScript-specific patterns.
fn typescript_patterns() -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    vec![
        // Type safety
        ("any", DefectCategory::TypeErrors, FindingSeverity::Medium, 0.5),
        ("as any", DefectCategory::TypeErrors, FindingSeverity::High, 0.7),
        ("// @ts-ignore", DefectCategory::TypeErrors, FindingSeverity::High, 0.75),
        ("// @ts-nocheck", DefectCategory::TypeErrors, FindingSeverity::Critical, 0.9),
        ("@ts-expect-error", DefectCategory::TypeErrors, FindingSeverity::Medium, 0.5),
        // Security
        ("eval(", DefectCategory::SecurityVulnerabilities, FindingSeverity::Critical, 0.95),
        ("innerHTML", DefectCategory::SecurityVulnerabilities, FindingSeverity::High, 0.8),
        ("dangerouslySetInnerHTML", DefectCategory::SecurityVulnerabilities, FindingSeverity::High, 0.8),
        ("document.write", DefectCategory::SecurityVulnerabilities, FindingSeverity::High, 0.8),
        // Anti-patterns
        ("console.log", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
        ("debugger", DefectCategory::LogicErrors, FindingSeverity::High, 0.7),
        ("== null", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("!= null", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        // Test debt
        ("it.skip", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        ("describe.skip", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        ("test.skip", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        (".only(", DefectCategory::TestDebt, FindingSeverity::High, 0.8),
        // Promise anti-patterns
        (".catch(() => {", DefectCategory::SilentDegradation, FindingSeverity::High, 0.75),
        (".catch(e => {})", DefectCategory::SilentDegradation, FindingSeverity::High, 0.8),
    ]
}

/// Go-specific patterns.
fn go_patterns() -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    vec![
        // Error handling
        ("_ = err", DefectCategory::SilentDegradation, FindingSeverity::Critical, 0.9),
        ("err != nil { return", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3), // OK pattern, low priority
        ("panic(", DefectCategory::LogicErrors, FindingSeverity::High, 0.7),
        ("log.Fatal", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        // Concurrency
        ("go func()", DefectCategory::ConcurrencyBugs, FindingSeverity::Medium, 0.5),
        ("sync.Mutex", DefectCategory::ConcurrencyBugs, FindingSeverity::Low, 0.3),
        ("data race", DefectCategory::ConcurrencyBugs, FindingSeverity::Critical, 0.95),
        // Security
        ("sql.Query(", DefectCategory::SecurityVulnerabilities, FindingSeverity::High, 0.7), // potential SQL injection
        ("http.Get(", DefectCategory::SecurityVulnerabilities, FindingSeverity::Medium, 0.5), // SSRF potential
        ("exec.Command(", DefectCategory::SecurityVulnerabilities, FindingSeverity::High, 0.8),
        // Anti-patterns
        ("interface{}", DefectCategory::TypeErrors, FindingSeverity::Medium, 0.4),
        ("//nolint", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        // Test debt
        ("t.Skip", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        ("testing.Short()", DefectCategory::TestDebt, FindingSeverity::Low, 0.3),
    ]
}

/// Get all supported file extensions as glob patterns.
pub fn all_language_globs() -> Vec<String> {
    vec![
        "**/*.rs".to_string(),
        "**/*.py".to_string(),
        "**/*.ts".to_string(),
        "**/*.tsx".to_string(),
        "**/*.js".to_string(),
        "**/*.jsx".to_string(),
        "**/*.go".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_from_extension() {
        assert_eq!(Language::from_extension("rs"), Some(Language::Rust));
        assert_eq!(Language::from_extension("py"), Some(Language::Python));
        assert_eq!(Language::from_extension("ts"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("go"), Some(Language::Go));
        assert_eq!(Language::from_extension("txt"), None);
    }

    #[test]
    fn test_patterns_for_rust() {
        let patterns = patterns_for_language(Language::Rust);
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "unwrap()"));
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "TODO")); // universal
    }

    #[test]
    fn test_patterns_for_python() {
        let patterns = patterns_for_language(Language::Python);
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "eval("));
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "TODO")); // universal
    }

    #[test]
    fn test_patterns_for_typescript() {
        let patterns = patterns_for_language(Language::TypeScript);
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "as any"));
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "TODO")); // universal
    }

    #[test]
    fn test_patterns_for_go() {
        let patterns = patterns_for_language(Language::Go);
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "panic("));
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "TODO")); // universal
    }

    #[test]
    fn test_all_language_globs() {
        let globs = all_language_globs();
        assert!(globs.contains(&"**/*.rs".to_string()));
        assert!(globs.contains(&"**/*.py".to_string()));
        assert!(globs.contains(&"**/*.go".to_string()));
    }
}
