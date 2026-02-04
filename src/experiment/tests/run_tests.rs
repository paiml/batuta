//! Tests for experiment runs, storage, research artifacts, and sovereign distribution.

use crate::experiment::*;

// -------------------------------------------------------------------------
// SovereignDistribution Tests
// -------------------------------------------------------------------------

#[test]
fn test_sovereign_distribution_creation() {
    let dist = SovereignDistribution::new("my-model", "1.0.0");
    assert_eq!(dist.name, "my-model");
    assert_eq!(dist.version, "1.0.0");
    assert!(dist.platforms.is_empty());
    assert!(dist.artifacts.is_empty());
}

#[test]
fn test_sovereign_distribution_add_platform() {
    let mut dist = SovereignDistribution::new("my-model", "1.0.0");
    dist.add_platform("linux-x86_64");
    dist.add_platform("darwin-aarch64");

    assert_eq!(dist.platforms.len(), 2);
    assert!(dist.platforms.contains(&"linux-x86_64".to_string()));
}

#[test]
fn test_sovereign_distribution_add_artifact() {
    let mut dist = SovereignDistribution::new("my-model", "1.0.0");
    dist.add_artifact(SovereignArtifact {
        name: "model.onnx".to_string(),
        artifact_type: ArtifactType::Model,
        sha256: "abc123".to_string(),
        size_bytes: 1_000_000,
        source_url: None,
    });

    assert_eq!(dist.artifacts.len(), 1);
    assert_eq!(dist.total_size_bytes(), 1_000_000);
}

#[test]
fn test_sovereign_distribution_validate_signatures_missing() {
    let mut dist = SovereignDistribution::new("my-model", "1.0.0");
    dist.add_artifact(SovereignArtifact {
        name: "model.onnx".to_string(),
        artifact_type: ArtifactType::Model,
        sha256: "abc123".to_string(),
        size_bytes: 1_000_000,
        source_url: None,
    });

    let result = dist.validate_signatures();
    assert!(result.is_err());
    match result {
        Err(ExperimentError::SovereignValidationFailed(msg)) => {
            assert!(msg.contains("model.onnx"));
        }
        _ => panic!("Expected SovereignValidationFailed error"),
    }
}

#[test]
fn test_sovereign_distribution_validate_signatures_present() {
    let mut dist = SovereignDistribution::new("my-model", "1.0.0");
    dist.add_artifact(SovereignArtifact {
        name: "model.onnx".to_string(),
        artifact_type: ArtifactType::Model,
        sha256: "abc123".to_string(),
        size_bytes: 1_000_000,
        source_url: None,
    });
    dist.signatures.push(ArtifactSignature {
        artifact_name: "model.onnx".to_string(),
        algorithm: SignatureAlgorithm::Ed25519,
        signature: "sig123".to_string(),
        key_id: "key1".to_string(),
    });

    let result = dist.validate_signatures();
    assert!(result.is_ok());
}

// -------------------------------------------------------------------------
// ORCID Tests
// -------------------------------------------------------------------------

#[test]
fn test_orcid_valid() {
    let orcid = Orcid::new("0000-0002-1825-0097");
    assert!(orcid.is_ok());
    assert_eq!(orcid.unwrap().as_str(), "0000-0002-1825-0097");
}

#[test]
fn test_orcid_valid_with_x() {
    let orcid = Orcid::new("0000-0002-1825-009X");
    assert!(orcid.is_ok());
}

#[test]
fn test_orcid_invalid_format() {
    let orcid = Orcid::new("invalid-orcid");
    assert!(orcid.is_err());
    match orcid {
        Err(ExperimentError::InvalidOrcid(s)) => {
            assert_eq!(s, "invalid-orcid");
        }
        _ => panic!("Expected InvalidOrcid error"),
    }
}

#[test]
fn test_orcid_invalid_too_short() {
    let orcid = Orcid::new("0000-0002-1825");
    assert!(orcid.is_err());
}

// -------------------------------------------------------------------------
// CitationMetadata Tests
// -------------------------------------------------------------------------

#[test]
fn test_citation_metadata_to_bibtex() {
    let citation = CitationMetadata {
        citation_type: CitationType::Article,
        title: "Test Paper".to_string(),
        authors: vec!["Alice Smith".to_string(), "Bob Jones".to_string()],
        year: 2024,
        month: Some(6),
        doi: Some("10.1234/test".to_string()),
        url: None,
        venue: Some("Journal of Testing".to_string()),
        volume: Some("42".to_string()),
        pages: Some("1-10".to_string()),
        publisher: None,
        version: None,
    };

    let bibtex = citation.to_bibtex("smith2024test");
    assert!(bibtex.contains("@article{smith2024test,"));
    assert!(bibtex.contains("title = {Test Paper}"));
    assert!(bibtex.contains("author = {Alice Smith and Bob Jones}"));
    assert!(bibtex.contains("year = {2024}"));
    assert!(bibtex.contains("doi = {10.1234/test}"));
    assert!(bibtex.contains("journal = {Journal of Testing}"));
}

#[test]
fn test_citation_metadata_to_cff() {
    let citation = CitationMetadata {
        citation_type: CitationType::Software,
        title: "My Tool".to_string(),
        authors: vec!["Developer".to_string()],
        year: 2024,
        month: Some(11),
        doi: Some("10.5281/zenodo.123".to_string()),
        url: Some("https://github.com/example/tool".to_string()),
        venue: None,
        volume: None,
        pages: None,
        publisher: None,
        version: Some("1.0.0".to_string()),
    };

    let cff = citation.to_cff();
    assert!(cff.contains("cff-version: 1.2.0"));
    assert!(cff.contains("title: \"My Tool\""));
    assert!(cff.contains("name: \"Developer\""));
    assert!(cff.contains("version: \"1.0.0\""));
    assert!(cff.contains("doi: \"10.5281/zenodo.123\""));
}

// -------------------------------------------------------------------------
// ExperimentRun Tests
// -------------------------------------------------------------------------

#[test]
fn test_experiment_run_creation() {
    let device = ComputeDevice::Cpu {
        cores: 8,
        threads_per_core: 2,
        architecture: CpuArchitecture::X86_64,
    };

    let run = ExperimentRun::new(
        "run-001",
        "my-experiment",
        ModelParadigm::DeepLearning,
        device,
    );

    assert_eq!(run.run_id, "run-001");
    assert_eq!(run.experiment_name, "my-experiment");
    assert_eq!(run.paradigm, ModelParadigm::DeepLearning);
    assert_eq!(run.status, RunStatus::Running);
    assert!(run.ended_at.is_none());
}

#[test]
fn test_experiment_run_log_metric() {
    let device = ComputeDevice::Cpu {
        cores: 8,
        threads_per_core: 2,
        architecture: CpuArchitecture::X86_64,
    };

    let mut run = ExperimentRun::new(
        "run-001",
        "my-experiment",
        ModelParadigm::DeepLearning,
        device,
    );
    run.log_metric("accuracy", 0.95);
    run.log_metric("loss", 0.05);

    assert_eq!(run.metrics.get("accuracy"), Some(&0.95));
    assert_eq!(run.metrics.get("loss"), Some(&0.05));
}

#[test]
fn test_experiment_run_log_param() {
    let device = ComputeDevice::Cpu {
        cores: 8,
        threads_per_core: 2,
        architecture: CpuArchitecture::X86_64,
    };

    let mut run = ExperimentRun::new(
        "run-001",
        "my-experiment",
        ModelParadigm::DeepLearning,
        device,
    );
    run.log_param("learning_rate", serde_json::json!(0.001));
    run.log_param("batch_size", serde_json::json!(32));

    assert_eq!(
        run.hyperparameters.get("learning_rate"),
        Some(&serde_json::json!(0.001))
    );
}

#[test]
fn test_experiment_run_complete() {
    let device = ComputeDevice::Cpu {
        cores: 8,
        threads_per_core: 2,
        architecture: CpuArchitecture::X86_64,
    };

    let mut run = ExperimentRun::new(
        "run-001",
        "my-experiment",
        ModelParadigm::DeepLearning,
        device,
    );
    run.complete();

    assert_eq!(run.status, RunStatus::Completed);
    assert!(run.ended_at.is_some());
}

#[test]
fn test_experiment_run_fail() {
    let device = ComputeDevice::Cpu {
        cores: 8,
        threads_per_core: 2,
        architecture: CpuArchitecture::X86_64,
    };

    let mut run = ExperimentRun::new(
        "run-001",
        "my-experiment",
        ModelParadigm::DeepLearning,
        device,
    );
    run.fail();

    assert_eq!(run.status, RunStatus::Failed);
    assert!(run.ended_at.is_some());
}

#[test]
fn test_experiment_run_serialization() {
    let device = ComputeDevice::Cpu {
        cores: 8,
        threads_per_core: 2,
        architecture: CpuArchitecture::X86_64,
    };

    let mut run = ExperimentRun::new(
        "run-001",
        "my-experiment",
        ModelParadigm::DeepLearning,
        device,
    );
    run.log_metric("accuracy", 0.95);

    let json = serde_json::to_string(&run).unwrap();
    let deserialized: ExperimentRun = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.run_id, "run-001");
    assert_eq!(deserialized.metrics.get("accuracy"), Some(&0.95));
}

// -------------------------------------------------------------------------
// ExperimentStorage Tests
// -------------------------------------------------------------------------

#[test]
fn test_in_memory_storage_store_and_get() {
    let storage = InMemoryExperimentStorage::new();
    let device = ComputeDevice::Cpu {
        cores: 8,
        threads_per_core: 2,
        architecture: CpuArchitecture::X86_64,
    };

    let run = ExperimentRun::new(
        "run-001",
        "my-experiment",
        ModelParadigm::DeepLearning,
        device,
    );
    storage.store_run(&run).unwrap();

    let retrieved = storage.get_run("run-001").unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().run_id, "run-001");
}

#[test]
fn test_in_memory_storage_get_nonexistent() {
    let storage = InMemoryExperimentStorage::new();
    let retrieved = storage.get_run("nonexistent").unwrap();
    assert!(retrieved.is_none());
}

#[test]
fn test_in_memory_storage_list_runs() {
    let storage = InMemoryExperimentStorage::new();
    let device = ComputeDevice::Cpu {
        cores: 8,
        threads_per_core: 2,
        architecture: CpuArchitecture::X86_64,
    };

    let run1 = ExperimentRun::new(
        "run-001",
        "exp-a",
        ModelParadigm::DeepLearning,
        device.clone(),
    );
    let run2 = ExperimentRun::new(
        "run-002",
        "exp-a",
        ModelParadigm::FineTuning,
        device.clone(),
    );
    let run3 = ExperimentRun::new("run-003", "exp-b", ModelParadigm::TraditionalML, device);

    storage.store_run(&run1).unwrap();
    storage.store_run(&run2).unwrap();
    storage.store_run(&run3).unwrap();

    let runs = storage.list_runs("exp-a").unwrap();
    assert_eq!(runs.len(), 2);
}

#[test]
fn test_in_memory_storage_delete_run() {
    let storage = InMemoryExperimentStorage::new();
    let device = ComputeDevice::Cpu {
        cores: 8,
        threads_per_core: 2,
        architecture: CpuArchitecture::X86_64,
    };

    let run = ExperimentRun::new(
        "run-001",
        "my-experiment",
        ModelParadigm::DeepLearning,
        device,
    );
    storage.store_run(&run).unwrap();

    storage.delete_run("run-001").unwrap();
    let retrieved = storage.get_run("run-001").unwrap();
    assert!(retrieved.is_none());
}

// -------------------------------------------------------------------------
// CRediT Role Tests
// -------------------------------------------------------------------------

#[test]
fn test_credit_role_variants() {
    let roles = vec![
        CreditRole::Conceptualization,
        CreditRole::DataCuration,
        CreditRole::FormalAnalysis,
        CreditRole::FundingAcquisition,
        CreditRole::Investigation,
        CreditRole::Methodology,
        CreditRole::ProjectAdministration,
        CreditRole::Resources,
        CreditRole::Software,
        CreditRole::Supervision,
        CreditRole::Validation,
        CreditRole::Visualization,
        CreditRole::WritingOriginalDraft,
        CreditRole::WritingReviewEditing,
    ];
    assert_eq!(roles.len(), 14);
}

#[test]
fn test_research_contributor_creation() {
    let contributor = ResearchContributor {
        name: "Alice Researcher".to_string(),
        orcid: Orcid::new("0000-0002-1825-0097").ok(),
        affiliation: "MIT".to_string(),
        roles: vec![CreditRole::Conceptualization, CreditRole::Software],
        email: Some("alice@mit.edu".to_string()),
    };

    assert_eq!(contributor.name, "Alice Researcher");
    assert!(contributor.orcid.is_some());
    assert_eq!(contributor.roles.len(), 2);
}
