//! Stack Compliance Configuration
//!
//! Defines the schema for stack-comply.yaml configuration files.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Main compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplyConfig {
    /// Workspace root directory
    #[serde(default)]
    pub workspace: Option<PathBuf>,

    /// Rules to enable (if empty, all rules are enabled)
    #[serde(default)]
    pub enabled_rules: Vec<String>,

    /// Rules to disable
    #[serde(default)]
    pub disabled_rules: Vec<String>,

    /// Include non-PAIML crates in checks
    #[serde(default)]
    pub include_external: bool,

    /// Project-specific overrides
    #[serde(default)]
    pub project_overrides: HashMap<String, ProjectOverride>,

    /// Makefile configuration
    #[serde(default)]
    pub makefile: MakefileConfig,

    /// Cargo.toml configuration
    #[serde(default)]
    pub cargo_toml: CargoTomlConfig,

    /// CI workflow configuration
    #[serde(default)]
    pub ci_workflows: CiWorkflowConfig,

    /// Duplication detection configuration
    #[serde(default)]
    pub duplication: DuplicationConfig,
}

impl Default for ComplyConfig {
    fn default() -> Self {
        Self {
            workspace: None,
            enabled_rules: Vec::new(),
            disabled_rules: Vec::new(),
            include_external: false,
            project_overrides: HashMap::new(),
            makefile: MakefileConfig::default(),
            cargo_toml: CargoTomlConfig::default(),
            ci_workflows: CiWorkflowConfig::default(),
            duplication: DuplicationConfig::default(),
        }
    }
}

impl ComplyConfig {
    /// Create default configuration for a workspace
    pub fn default_for_workspace(workspace: &Path) -> Self {
        Self {
            workspace: Some(workspace.to_path_buf()),
            ..Default::default()
        }
    }

    /// Load configuration from a YAML file
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from workspace or use defaults
    pub fn load_or_default(workspace: &Path) -> Self {
        let config_path = workspace.join("stack-comply.yaml");
        if config_path.exists() {
            Self::load(&config_path).unwrap_or_else(|_| Self::default_for_workspace(workspace))
        } else {
            Self::default_for_workspace(workspace)
        }
    }

    /// Save configuration to a YAML file
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let content = serde_yaml::to_string(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// Project-specific override configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProjectOverride {
    /// Rules exempt from checking for this project
    #[serde(default)]
    pub exempt_rules: Vec<String>,

    /// Custom Makefile targets for this project
    #[serde(default)]
    pub custom_targets: Vec<String>,

    /// Justification for overrides
    pub justification: Option<String>,
}

/// Makefile target configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MakefileConfig {
    /// Required targets with expected command patterns
    pub required_targets: HashMap<String, TargetConfig>,

    /// Allowed variations for specific targets
    pub allowed_variations: HashMap<String, Vec<VariationConfig>>,

    /// Prohibited commands (e.g., cargo tarpaulin)
    pub prohibited_commands: Vec<String>,
}

impl Default for MakefileConfig {
    fn default() -> Self {
        let mut required_targets = HashMap::new();

        required_targets.insert(
            "test-fast".to_string(),
            TargetConfig {
                pattern: Some("cargo nextest run --lib".to_string()),
                description: "Fast unit tests".to_string(),
                required: true,
            },
        );

        required_targets.insert(
            "test".to_string(),
            TargetConfig {
                pattern: Some("cargo nextest run".to_string()),
                description: "Standard tests".to_string(),
                required: true,
            },
        );

        required_targets.insert(
            "lint".to_string(),
            TargetConfig {
                pattern: Some("cargo clippy".to_string()),
                description: "Clippy linting".to_string(),
                required: true,
            },
        );

        required_targets.insert(
            "fmt".to_string(),
            TargetConfig {
                pattern: Some("cargo fmt".to_string()),
                description: "Format code".to_string(),
                required: true,
            },
        );

        required_targets.insert(
            "coverage".to_string(),
            TargetConfig {
                pattern: Some("cargo llvm-cov".to_string()),
                description: "Coverage report".to_string(),
                required: true,
            },
        );

        let prohibited_commands = vec![
            "cargo tarpaulin".to_string(),
            "cargo-tarpaulin".to_string(),
        ];

        Self {
            required_targets,
            allowed_variations: HashMap::new(),
            prohibited_commands,
        }
    }
}

/// Configuration for a single Makefile target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetConfig {
    /// Expected command pattern (regex or contains)
    pub pattern: Option<String>,
    /// Description of what this target should do
    pub description: String,
    /// Whether this target is required
    #[serde(default = "default_true")]
    pub required: bool,
}

fn default_true() -> bool {
    true
}

/// Allowed variation for a target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationConfig {
    /// Pattern that's allowed
    pub pattern: String,
    /// Reason this variation is acceptable
    pub reason: String,
}

/// Cargo.toml consistency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoTomlConfig {
    /// Required dependencies with version constraints
    #[serde(default)]
    pub required_dependencies: HashMap<String, String>,

    /// Prohibited dependencies
    #[serde(default)]
    pub prohibited_dependencies: Vec<String>,

    /// Required metadata fields
    #[serde(default)]
    pub required_metadata: RequiredMetadata,

    /// Required features when applicable
    #[serde(default)]
    pub required_features: HashMap<String, FeatureRequirement>,
}

impl Default for CargoTomlConfig {
    fn default() -> Self {
        let mut required_dependencies = HashMap::new();
        required_dependencies.insert("trueno".to_string(), ">=0.14.0".to_string());

        Self {
            required_dependencies,
            prohibited_dependencies: vec!["cargo-tarpaulin".to_string()],
            required_metadata: RequiredMetadata::default(),
            required_features: HashMap::new(),
        }
    }
}

/// Required Cargo.toml metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredMetadata {
    /// Required license
    pub license: Option<String>,
    /// Required edition
    pub edition: Option<String>,
    /// Minimum rust-version
    pub rust_version: Option<String>,
}

impl Default for RequiredMetadata {
    fn default() -> Self {
        Self {
            license: Some("MIT OR Apache-2.0".to_string()),
            edition: Some("2024".to_string()),
            rust_version: Some("1.85".to_string()),
        }
    }
}

/// Feature requirement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureRequirement {
    /// Condition when this feature is required
    pub required_if: String,
    /// Dependencies this feature must include
    #[serde(default)]
    pub must_include: Vec<String>,
}

/// CI workflow parity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiWorkflowConfig {
    /// Required workflow files
    #[serde(default)]
    pub required_workflows: Vec<String>,

    /// Required jobs in CI workflow
    #[serde(default)]
    pub required_jobs: Vec<String>,

    /// Required matrix dimensions
    #[serde(default)]
    pub required_matrix: MatrixConfig,

    /// Required artifacts
    #[serde(default)]
    pub required_artifacts: Vec<String>,
}

impl Default for CiWorkflowConfig {
    fn default() -> Self {
        Self {
            required_workflows: vec!["ci.yml".to_string(), "ci.yaml".to_string()],
            required_jobs: vec![
                "fmt-check".to_string(),
                "clippy".to_string(),
                "test".to_string(),
            ],
            required_matrix: MatrixConfig::default(),
            required_artifacts: vec!["coverage-report".to_string()],
        }
    }
}

/// CI matrix configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixConfig {
    /// Required OS values
    #[serde(default)]
    pub os: Vec<String>,
    /// Required Rust toolchain values
    #[serde(default)]
    pub rust: Vec<String>,
}

impl Default for MatrixConfig {
    fn default() -> Self {
        Self {
            os: vec!["ubuntu-latest".to_string()],
            rust: vec!["stable".to_string()],
        }
    }
}

/// Code duplication detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicationConfig {
    /// Similarity threshold for duplicates (0.0-1.0)
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f64,

    /// Minimum fragment size in lines
    #[serde(default = "default_min_fragment_size")]
    pub min_fragment_size: usize,

    /// Number of MinHash permutations
    #[serde(default = "default_num_perm")]
    pub num_permutations: usize,

    /// File patterns to include
    #[serde(default)]
    pub include_patterns: Vec<String>,

    /// File patterns to exclude
    #[serde(default)]
    pub exclude_patterns: Vec<String>,

    /// Whether to only report cross-project duplicates
    #[serde(default = "default_true")]
    pub cross_project_only: bool,
}

fn default_similarity_threshold() -> f64 {
    0.85
}

fn default_min_fragment_size() -> usize {
    50
}

fn default_num_perm() -> usize {
    128
}

impl Default for DuplicationConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            min_fragment_size: 50,
            num_permutations: 128,
            include_patterns: vec!["**/*.rs".to_string()],
            exclude_patterns: vec![
                "**/target/**".to_string(),
                "**/tests/**".to_string(),
                "**/benches/**".to_string(),
            ],
            cross_project_only: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = ComplyConfig::default();
        assert!(!config.makefile.required_targets.is_empty());
        assert!(config.makefile.required_targets.contains_key("test-fast"));
        assert!(config.makefile.required_targets.contains_key("lint"));
    }

    #[test]
    fn test_config_serialization() {
        let config = ComplyConfig::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("makefile"));
        assert!(yaml.contains("cargo_toml"));
    }

    #[test]
    fn test_config_load() {
        let yaml = r#"
workspace: /tmp/test
enabled_rules:
  - makefile-targets
makefile:
  prohibited_commands:
    - cargo tarpaulin
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();

        let config = ComplyConfig::load(file.path()).unwrap();
        assert_eq!(config.enabled_rules, vec!["makefile-targets"]);
    }

    #[test]
    fn test_default_makefile_config() {
        let config = MakefileConfig::default();
        assert!(config.required_targets.contains_key("test-fast"));
        assert!(config.required_targets.contains_key("coverage"));
        assert!(config.prohibited_commands.contains(&"cargo tarpaulin".to_string()));
    }

    #[test]
    fn test_duplication_config_defaults() {
        let config = DuplicationConfig::default();
        assert!((config.similarity_threshold - 0.85).abs() < f64::EPSILON);
        assert_eq!(config.min_fragment_size, 50);
        assert!(config.cross_project_only);
    }
}
