//! Sovereign distribution manifest for air-gapped deployments
//!
//! This module contains sovereign distribution types extracted from experiment/mod.rs.

use super::ExperimentError;
use serde::{Deserialize, Serialize};

/// Sovereign distribution manifest for air-gapped deployments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignDistribution {
    /// Distribution name
    pub name: String,
    /// Version
    pub version: String,
    /// Target platforms
    pub platforms: Vec<String>,
    /// Required artifacts
    pub artifacts: Vec<SovereignArtifact>,
    /// Cryptographic signatures
    pub signatures: Vec<ArtifactSignature>,
    /// Offline registry configuration
    pub offline_registry: Option<OfflineRegistryConfig>,
    /// Nix flake hash for reproducibility
    pub nix_flake_hash: Option<String>,
}

/// Artifact in a sovereign distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignArtifact {
    /// Artifact name
    pub name: String,
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// SHA-256 hash
    pub sha256: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Download URL (for pre-staging)
    pub source_url: Option<String>,
}

/// Types of distributable artifacts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactType {
    Binary,
    Model,
    Dataset,
    Config,
    Documentation,
    Container,
    NixDerivation,
}

/// Cryptographic signature for artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactSignature {
    /// Artifact name this signature is for
    pub artifact_name: String,
    /// Signature algorithm
    pub algorithm: SignatureAlgorithm,
    /// Base64-encoded signature
    pub signature: String,
    /// Public key identifier
    pub key_id: String,
}

/// Supported signature algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignatureAlgorithm {
    Ed25519,
    RSA4096,
    EcdsaP256,
}

/// Offline model registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineRegistryConfig {
    /// Registry path
    pub path: String,
    /// Index file location
    pub index_path: String,
    /// Supported platforms
    pub platforms: Vec<String>,
    /// Last sync timestamp
    pub last_sync: Option<String>,
}

impl SovereignDistribution {
    /// Create a new sovereign distribution
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            platforms: Vec::new(),
            artifacts: Vec::new(),
            signatures: Vec::new(),
            offline_registry: None,
            nix_flake_hash: None,
        }
    }

    /// Add a platform target
    pub fn add_platform(&mut self, platform: impl Into<String>) {
        self.platforms.push(platform.into());
    }

    /// Add an artifact
    pub fn add_artifact(&mut self, artifact: SovereignArtifact) {
        self.artifacts.push(artifact);
    }

    /// Validate all artifacts have signatures
    pub fn validate_signatures(&self) -> Result<(), ExperimentError> {
        for artifact in &self.artifacts {
            let has_sig = self
                .signatures
                .iter()
                .any(|s| s.artifact_name == artifact.name);
            if !has_sig {
                return Err(ExperimentError::SovereignValidationFailed(format!(
                    "Missing signature for artifact: {}",
                    artifact.name
                )));
            }
        }
        Ok(())
    }

    /// Calculate total distribution size
    pub fn total_size_bytes(&self) -> u64 {
        self.artifacts.iter().map(|a| a.size_bytes).sum()
    }
}
