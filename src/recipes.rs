//! Orchestration Recipes Module
//!
//! Provides high-level workflows for integrating experiment tracking,
//! cost-performance benchmarking, and sovereign deployment pipelines
//! with the Batuta orchestration framework.

use crate::experiment::{
    CitationMetadata, CitationType, ComputeDevice, CostMetrics, CostPerformanceBenchmark,
    CostPerformancePoint, CreditRole, EnergyMetrics, ExperimentError, ExperimentRun,
    ExperimentStorage, ModelParadigm, PlatformEfficiency, ResearchArtifact,
    ResearchContributor, SovereignArtifact, SovereignDistribution,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Recipe execution result
#[derive(Debug, Clone)]
pub struct RecipeResult {
    /// Recipe name
    pub recipe_name: String,
    /// Whether the recipe succeeded
    pub success: bool,
    /// Output artifacts
    pub artifacts: Vec<String>,
    /// Metrics collected
    pub metrics: HashMap<String, f64>,
    /// Error message if failed
    pub error: Option<String>,
}

impl RecipeResult {
    /// Create a successful result
    pub fn success(name: impl Into<String>) -> Self {
        Self {
            recipe_name: name.into(),
            success: true,
            artifacts: Vec::new(),
            metrics: HashMap::new(),
            error: None,
        }
    }

    /// Create a failed result
    pub fn failure(name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            recipe_name: name.into(),
            success: false,
            artifacts: Vec::new(),
            metrics: HashMap::new(),
            error: Some(error.into()),
        }
    }

    /// Add an artifact
    pub fn with_artifact(mut self, artifact: impl Into<String>) -> Self {
        self.artifacts.push(artifact.into());
        self
    }

    /// Add a metric
    pub fn with_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }
}

/// Experiment tracking recipe configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentTrackingConfig {
    /// Experiment name
    pub experiment_name: String,
    /// Model paradigm
    pub paradigm: ModelParadigm,
    /// Compute device configuration
    pub device: ComputeDevice,
    /// Platform type
    pub platform: PlatformEfficiency,
    /// Enable energy tracking
    pub track_energy: bool,
    /// Enable cost tracking
    pub track_cost: bool,
    /// Carbon intensity for CO2 calculation (g/kWh)
    pub carbon_intensity: Option<f64>,
    /// Tags for organization
    pub tags: Vec<String>,
}

impl Default for ExperimentTrackingConfig {
    fn default() -> Self {
        Self {
            experiment_name: "default-experiment".to_string(),
            paradigm: ModelParadigm::DeepLearning,
            device: ComputeDevice::Cpu {
                cores: 8,
                threads_per_core: 2,
                architecture: crate::experiment::CpuArchitecture::X86_64,
            },
            platform: PlatformEfficiency::Server,
            track_energy: true,
            track_cost: true,
            carbon_intensity: Some(400.0), // Global average
            tags: Vec::new(),
        }
    }
}

/// Experiment tracking recipe
#[derive(Debug)]
pub struct ExperimentTrackingRecipe {
    config: ExperimentTrackingConfig,
    current_run: Option<ExperimentRun>,
    start_time: Option<std::time::Instant>,
}

impl ExperimentTrackingRecipe {
    /// Create a new experiment tracking recipe
    pub fn new(config: ExperimentTrackingConfig) -> Self {
        Self {
            config,
            current_run: None,
            start_time: None,
        }
    }

    /// Start a new experiment run
    pub fn start_run(&mut self, run_id: impl Into<String>) -> &mut ExperimentRun {
        let mut run = ExperimentRun::new(
            run_id,
            &self.config.experiment_name,
            self.config.paradigm,
            self.config.device.clone(),
        );
        run.platform = self.config.platform;
        run.tags = self.config.tags.clone();
        self.current_run = Some(run);
        self.start_time = Some(std::time::Instant::now());
        self.current_run.as_mut().unwrap()
    }

    /// Log a metric to the current run
    pub fn log_metric(&mut self, name: impl Into<String>, value: f64) -> Result<(), ExperimentError> {
        self.current_run
            .as_mut()
            .ok_or_else(|| ExperimentError::StorageError("No active run".to_string()))?
            .log_metric(name, value);
        Ok(())
    }

    /// Log a hyperparameter
    pub fn log_param(&mut self, name: impl Into<String>, value: serde_json::Value) -> Result<(), ExperimentError> {
        self.current_run
            .as_mut()
            .ok_or_else(|| ExperimentError::StorageError("No active run".to_string()))?
            .log_param(name, value);
        Ok(())
    }

    /// End the current run and calculate metrics
    pub fn end_run(&mut self, success: bool) -> Result<RecipeResult, ExperimentError> {
        let run = self.current_run.as_mut().ok_or_else(|| {
            ExperimentError::StorageError("No active run".to_string())
        })?;

        if success {
            run.complete();
        } else {
            run.fail();
        }

        let duration = self.start_time.take().map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);

        // Calculate energy metrics if enabled
        if self.config.track_energy {
            let power = self.config.device.estimated_power_watts() as f64;
            let energy_joules = power * duration;
            let mut energy = EnergyMetrics::new(energy_joules, power, power * 1.2, duration);

            if let Some(carbon_intensity) = self.config.carbon_intensity {
                energy = energy.with_carbon_intensity(carbon_intensity);
            }

            run.energy = Some(energy);
        }

        let mut result = RecipeResult::success("experiment-tracking");
        result = result.with_metric("duration_seconds", duration);

        if let Some(ref energy) = run.energy {
            result = result.with_metric("energy_joules", energy.total_joules);
            if let Some(co2) = energy.co2_grams {
                result = result.with_metric("co2_grams", co2);
            }
        }

        for (name, value) in &run.metrics {
            result = result.with_metric(format!("run_{}", name), *value);
        }

        Ok(result)
    }

    /// Get the current run
    pub fn current_run(&self) -> Option<&ExperimentRun> {
        self.current_run.as_ref()
    }

    /// Store the run to a backend
    pub fn store_run<S: ExperimentStorage>(&self, storage: &S) -> Result<(), ExperimentError> {
        if let Some(ref run) = self.current_run {
            storage.store_run(run)?;
        }
        Ok(())
    }
}

/// Cost-performance benchmarking recipe
#[derive(Debug)]
pub struct CostPerformanceBenchmarkRecipe {
    benchmark: CostPerformanceBenchmark,
    budget_constraint: Option<f64>,
    performance_target: Option<f64>,
}

impl CostPerformanceBenchmarkRecipe {
    /// Create a new benchmarking recipe
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            benchmark: CostPerformanceBenchmark::new(name),
            budget_constraint: None,
            performance_target: None,
        }
    }

    /// Set a budget constraint
    pub fn with_budget(mut self, max_cost: f64) -> Self {
        self.budget_constraint = Some(max_cost);
        self
    }

    /// Set a performance target
    pub fn with_performance_target(mut self, target: f64) -> Self {
        self.performance_target = Some(target);
        self
    }

    /// Add an experiment run as a data point
    pub fn add_run(&mut self, run: &ExperimentRun, performance_metric: &str) {
        let performance = run.metrics.get(performance_metric).copied().unwrap_or(0.0);
        let cost = run.cost.as_ref().map(|c| c.total_cost_usd).unwrap_or(0.0);
        let energy = run.energy.as_ref().map(|e| e.total_joules).unwrap_or(0.0);

        let mut metadata = HashMap::new();
        metadata.insert("paradigm".to_string(), format!("{:?}", run.paradigm));
        metadata.insert("device".to_string(), format!("{:?}", run.device));
        metadata.insert("platform".to_string(), format!("{:?}", run.platform));

        self.benchmark.add_point(CostPerformancePoint {
            id: run.run_id.clone(),
            performance,
            cost,
            energy_joules: energy,
            latency_ms: None,
            metadata,
        });
    }

    /// Run the benchmark analysis
    pub fn analyze(&mut self) -> RecipeResult {
        let mut result = RecipeResult::success("cost-performance-benchmark");

        // Compute Pareto frontier
        let frontier = self.benchmark.compute_pareto_frontier().to_vec();
        result = result.with_metric("pareto_optimal_count", frontier.len() as f64);
        result = result.with_metric("total_configurations", self.benchmark.points.len() as f64);

        // Find best within budget if constraint set
        if let Some(budget) = self.budget_constraint {
            if let Some(best) = self.benchmark.best_within_budget(budget) {
                result = result.with_metric("best_in_budget_performance", best.performance);
                result = result.with_metric("best_in_budget_cost", best.cost);
            }
        }

        // Check if any point meets performance target
        if let Some(target) = self.performance_target {
            let meets_target = self.benchmark.points.iter().any(|p| p.performance >= target);
            result = result.with_metric("meets_target", if meets_target { 1.0 } else { 0.0 });

            // Find cheapest that meets target
            let cheapest = self.benchmark.points.iter()
                .filter(|p| p.performance >= target)
                .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());

            if let Some(cheapest) = cheapest {
                result = result.with_metric("cheapest_meeting_target_cost", cheapest.cost);
            }
        }

        // Add efficiency scores
        let efficiency = self.benchmark.efficiency_scores();
        if !efficiency.is_empty() {
            let max_efficiency = efficiency.iter().map(|e| e.1).fold(f64::NEG_INFINITY, f64::max);
            result = result.with_metric("max_efficiency", max_efficiency);
        }

        result
    }

    /// Get the benchmark
    pub fn benchmark(&self) -> &CostPerformanceBenchmark {
        &self.benchmark
    }

    /// Get mutable benchmark
    pub fn benchmark_mut(&mut self) -> &mut CostPerformanceBenchmark {
        &mut self.benchmark
    }
}

/// Sovereign deployment recipe configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignDeploymentConfig {
    /// Distribution name
    pub name: String,
    /// Version
    pub version: String,
    /// Target platforms
    pub platforms: Vec<String>,
    /// Require signatures
    pub require_signatures: bool,
    /// Nix flake support
    pub enable_nix_flake: bool,
    /// Offline registry path
    pub offline_registry_path: Option<String>,
}

impl Default for SovereignDeploymentConfig {
    fn default() -> Self {
        Self {
            name: "sovereign-model".to_string(),
            version: "0.1.0".to_string(),
            platforms: vec!["linux-x86_64".to_string()],
            require_signatures: true,
            enable_nix_flake: true,
            offline_registry_path: None,
        }
    }
}

/// Sovereign deployment recipe
#[derive(Debug)]
pub struct SovereignDeploymentRecipe {
    config: SovereignDeploymentConfig,
    distribution: SovereignDistribution,
}

impl SovereignDeploymentRecipe {
    /// Create a new sovereign deployment recipe
    pub fn new(config: SovereignDeploymentConfig) -> Self {
        let mut distribution = SovereignDistribution::new(&config.name, &config.version);
        for platform in &config.platforms {
            distribution.add_platform(platform);
        }
        Self { config, distribution }
    }

    /// Add a model artifact
    pub fn add_model(&mut self, name: impl Into<String>, sha256: impl Into<String>, size_bytes: u64) {
        self.distribution.add_artifact(SovereignArtifact {
            name: name.into(),
            artifact_type: crate::experiment::ArtifactType::Model,
            sha256: sha256.into(),
            size_bytes,
            source_url: None,
        });
    }

    /// Add a binary artifact
    pub fn add_binary(&mut self, name: impl Into<String>, sha256: impl Into<String>, size_bytes: u64) {
        self.distribution.add_artifact(SovereignArtifact {
            name: name.into(),
            artifact_type: crate::experiment::ArtifactType::Binary,
            sha256: sha256.into(),
            size_bytes,
            source_url: None,
        });
    }

    /// Add a dataset artifact
    pub fn add_dataset(&mut self, name: impl Into<String>, sha256: impl Into<String>, size_bytes: u64) {
        self.distribution.add_artifact(SovereignArtifact {
            name: name.into(),
            artifact_type: crate::experiment::ArtifactType::Dataset,
            sha256: sha256.into(),
            size_bytes,
            source_url: None,
        });
    }

    /// Sign an artifact (placeholder - would use real crypto in production)
    pub fn sign_artifact(&mut self, artifact_name: impl Into<String>, key_id: impl Into<String>) {
        let name = artifact_name.into();
        // In production, this would actually compute the signature
        let signature = format!("sig_placeholder_{}", &name);
        self.distribution.signatures.push(crate::experiment::ArtifactSignature {
            artifact_name: name,
            algorithm: crate::experiment::SignatureAlgorithm::Ed25519,
            signature,
            key_id: key_id.into(),
        });
    }

    /// Validate and build the distribution
    pub fn build(&self) -> Result<RecipeResult, ExperimentError> {
        // Validate signatures if required
        if self.config.require_signatures {
            self.distribution.validate_signatures()?;
        }

        let mut result = RecipeResult::success("sovereign-deployment");
        result = result.with_metric("artifact_count", self.distribution.artifacts.len() as f64);
        result = result.with_metric("total_size_bytes", self.distribution.total_size_bytes() as f64);
        result = result.with_metric("platform_count", self.distribution.platforms.len() as f64);

        // Add artifacts to result
        for artifact in &self.distribution.artifacts {
            result = result.with_artifact(&artifact.name);
        }

        Ok(result)
    }

    /// Get the distribution
    pub fn distribution(&self) -> &SovereignDistribution {
        &self.distribution
    }

    /// Export distribution manifest as JSON
    pub fn export_manifest(&self) -> Result<String, ExperimentError> {
        serde_json::to_string_pretty(&self.distribution)
            .map_err(|e| ExperimentError::StorageError(e.to_string()))
    }
}

/// Academic research artifact recipe
#[derive(Debug)]
pub struct ResearchArtifactRecipe {
    artifact: ResearchArtifact,
}

impl ResearchArtifactRecipe {
    /// Create a new research artifact recipe
    pub fn new(title: impl Into<String>, abstract_text: impl Into<String>) -> Self {
        Self {
            artifact: ResearchArtifact {
                title: title.into(),
                abstract_text: abstract_text.into(),
                contributors: Vec::new(),
                keywords: Vec::new(),
                doi: None,
                arxiv_id: None,
                license: "MIT".to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                datasets: Vec::new(),
                code_repositories: Vec::new(),
                pre_registration: None,
            },
        }
    }

    /// Add a contributor
    pub fn add_contributor(
        &mut self,
        name: impl Into<String>,
        affiliation: impl Into<String>,
        roles: Vec<CreditRole>,
    ) {
        self.artifact.contributors.push(ResearchContributor {
            name: name.into(),
            orcid: None,
            affiliation: affiliation.into(),
            roles,
            email: None,
        });
    }

    /// Add keywords
    pub fn add_keywords(&mut self, keywords: Vec<String>) {
        self.artifact.keywords.extend(keywords);
    }

    /// Set DOI
    pub fn set_doi(&mut self, doi: impl Into<String>) {
        self.artifact.doi = Some(doi.into());
    }

    /// Add dataset reference
    pub fn add_dataset(&mut self, dataset: impl Into<String>) {
        self.artifact.datasets.push(dataset.into());
    }

    /// Add code repository
    pub fn add_repository(&mut self, repo: impl Into<String>) {
        self.artifact.code_repositories.push(repo.into());
    }

    /// Generate citation metadata
    pub fn generate_citation(&self) -> CitationMetadata {
        let authors: Vec<String> = self.artifact.contributors.iter()
            .map(|c| c.name.clone())
            .collect();

        let now = chrono::Utc::now();

        CitationMetadata {
            citation_type: CitationType::Software,
            title: self.artifact.title.clone(),
            authors,
            year: now.format("%Y").to_string().parse().unwrap_or(2024),
            month: Some(now.format("%m").to_string().parse().unwrap_or(1)),
            doi: self.artifact.doi.clone(),
            url: self.artifact.code_repositories.first().cloned(),
            venue: None,
            volume: None,
            pages: None,
            publisher: None,
            version: Some("1.0.0".to_string()),
        }
    }

    /// Build the research artifact
    pub fn build(&self) -> RecipeResult {
        let mut result = RecipeResult::success("research-artifact");
        result = result.with_metric("contributor_count", self.artifact.contributors.len() as f64);
        result = result.with_metric("keyword_count", self.artifact.keywords.len() as f64);
        result = result.with_metric("dataset_count", self.artifact.datasets.len() as f64);

        if self.artifact.doi.is_some() {
            result = result.with_artifact("DOI registered");
        }

        let citation = self.generate_citation();
        result = result.with_artifact(format!("BibTeX: {}", citation.to_bibtex("artifact")));

        result
    }

    /// Get the artifact
    pub fn artifact(&self) -> &ResearchArtifact {
        &self.artifact
    }
}

/// CI/CD integration recipe for cost-performance benchmarking
#[derive(Debug)]
pub struct CiCdBenchmarkRecipe {
    /// Benchmark name
    name: String,
    /// Threshold checks
    thresholds: HashMap<String, f64>,
    /// Results from runs
    results: Vec<CostPerformancePoint>,
}

impl CiCdBenchmarkRecipe {
    /// Create a new CI/CD benchmark recipe
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            thresholds: HashMap::new(),
            results: Vec::new(),
        }
    }

    /// Add a performance threshold (fails if below)
    pub fn add_min_performance_threshold(&mut self, metric: impl Into<String>, min_value: f64) {
        self.thresholds.insert(format!("min_{}", metric.into()), min_value);
    }

    /// Add a cost threshold (fails if above)
    pub fn add_max_cost_threshold(&mut self, max_cost: f64) {
        self.thresholds.insert("max_cost".to_string(), max_cost);
    }

    /// Add a result
    pub fn add_result(&mut self, point: CostPerformancePoint) {
        self.results.push(point);
    }

    /// Check thresholds and return CI/CD result
    pub fn check(&self) -> RecipeResult {
        let mut result = RecipeResult::success(&self.name);
        let mut all_passed = true;

        for point in &self.results {
            // Check cost threshold
            if let Some(&max_cost) = self.thresholds.get("max_cost") {
                if point.cost > max_cost {
                    all_passed = false;
                    result = result.with_metric(format!("{}_cost_exceeded", point.id), point.cost - max_cost);
                }
            }

            // Check performance thresholds
            for (key, &threshold) in &self.thresholds {
                if key.starts_with("min_") {
                    let metric_name = key.strip_prefix("min_").unwrap();
                    if metric_name == "performance" && point.performance < threshold {
                        all_passed = false;
                        result = result.with_metric(
                            format!("{}_performance_below_threshold", point.id),
                            threshold - point.performance,
                        );
                    }
                }
            }
        }

        result = result.with_metric("all_checks_passed", if all_passed { 1.0 } else { 0.0 });

        if !all_passed {
            result.success = false;
            result.error = Some("One or more threshold checks failed".to_string());
        }

        result
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{CpuArchitecture, InMemoryExperimentStorage};

    // -------------------------------------------------------------------------
    // RecipeResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_recipe_result_success() {
        let result = RecipeResult::success("test-recipe");
        assert!(result.success);
        assert_eq!(result.recipe_name, "test-recipe");
        assert!(result.error.is_none());
    }

    #[test]
    fn test_recipe_result_failure() {
        let result = RecipeResult::failure("test-recipe", "Something went wrong");
        assert!(!result.success);
        assert_eq!(result.error, Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_recipe_result_with_artifact() {
        let result = RecipeResult::success("test")
            .with_artifact("artifact1")
            .with_artifact("artifact2");
        assert_eq!(result.artifacts.len(), 2);
    }

    #[test]
    fn test_recipe_result_with_metric() {
        let result = RecipeResult::success("test")
            .with_metric("accuracy", 0.95)
            .with_metric("loss", 0.05);
        assert_eq!(result.metrics.get("accuracy"), Some(&0.95));
        assert_eq!(result.metrics.get("loss"), Some(&0.05));
    }

    // -------------------------------------------------------------------------
    // ExperimentTrackingRecipe Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_experiment_tracking_recipe_creation() {
        let config = ExperimentTrackingConfig::default();
        let recipe = ExperimentTrackingRecipe::new(config);
        assert!(recipe.current_run().is_none());
    }

    #[test]
    fn test_experiment_tracking_start_run() {
        let config = ExperimentTrackingConfig::default();
        let mut recipe = ExperimentTrackingRecipe::new(config);

        let run = recipe.start_run("run-001");
        assert_eq!(run.run_id, "run-001");
        assert!(recipe.current_run().is_some());
    }

    #[test]
    fn test_experiment_tracking_log_metric() {
        let config = ExperimentTrackingConfig::default();
        let mut recipe = ExperimentTrackingRecipe::new(config);

        recipe.start_run("run-001");
        recipe.log_metric("accuracy", 0.95).unwrap();

        let run = recipe.current_run().unwrap();
        assert_eq!(run.metrics.get("accuracy"), Some(&0.95));
    }

    #[test]
    fn test_experiment_tracking_log_metric_no_run() {
        let config = ExperimentTrackingConfig::default();
        let mut recipe = ExperimentTrackingRecipe::new(config);

        let result = recipe.log_metric("accuracy", 0.95);
        assert!(result.is_err());
    }

    #[test]
    fn test_experiment_tracking_end_run() {
        let config = ExperimentTrackingConfig::default();
        let mut recipe = ExperimentTrackingRecipe::new(config);

        recipe.start_run("run-001");
        recipe.log_metric("accuracy", 0.95).unwrap();

        let result = recipe.end_run(true).unwrap();
        assert!(result.success);
        assert!(result.metrics.contains_key("duration_seconds"));
    }

    #[test]
    fn test_experiment_tracking_store_run() {
        let config = ExperimentTrackingConfig::default();
        let mut recipe = ExperimentTrackingRecipe::new(config);
        let storage = InMemoryExperimentStorage::new();

        recipe.start_run("run-001");
        recipe.store_run(&storage).unwrap();

        let retrieved = storage.get_run("run-001").unwrap();
        assert!(retrieved.is_some());
    }

    // -------------------------------------------------------------------------
    // CostPerformanceBenchmarkRecipe Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_benchmark_recipe_creation() {
        let recipe = CostPerformanceBenchmarkRecipe::new("test-benchmark");
        assert_eq!(recipe.benchmark().name, "test-benchmark");
    }

    #[test]
    fn test_benchmark_recipe_with_budget() {
        let recipe = CostPerformanceBenchmarkRecipe::new("test").with_budget(100.0);
        assert_eq!(recipe.budget_constraint, Some(100.0));
    }

    #[test]
    fn test_benchmark_recipe_analyze() {
        let mut recipe = CostPerformanceBenchmarkRecipe::new("test")
            .with_budget(150.0)
            .with_performance_target(0.90);

        // Add some points directly
        recipe.benchmark_mut().add_point(CostPerformancePoint {
            id: "config1".to_string(),
            performance: 0.95,
            cost: 100.0,
            energy_joules: 1000.0,
            latency_ms: None,
            metadata: HashMap::new(),
        });

        recipe.benchmark_mut().add_point(CostPerformancePoint {
            id: "config2".to_string(),
            performance: 0.85,
            cost: 50.0,
            energy_joules: 500.0,
            latency_ms: None,
            metadata: HashMap::new(),
        });

        let result = recipe.analyze();
        assert!(result.success);
        assert!(result.metrics.contains_key("pareto_optimal_count"));
        assert!(result.metrics.contains_key("meets_target"));
    }

    // -------------------------------------------------------------------------
    // SovereignDeploymentRecipe Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sovereign_deployment_creation() {
        let config = SovereignDeploymentConfig::default();
        let recipe = SovereignDeploymentRecipe::new(config);
        assert_eq!(recipe.distribution().name, "sovereign-model");
    }

    #[test]
    fn test_sovereign_deployment_add_artifacts() {
        let config = SovereignDeploymentConfig {
            require_signatures: false,
            ..Default::default()
        };
        let mut recipe = SovereignDeploymentRecipe::new(config);

        recipe.add_model("model.onnx", "sha256_hash", 1000000);
        recipe.add_binary("inference", "sha256_hash2", 500000);

        assert_eq!(recipe.distribution().artifacts.len(), 2);
    }

    #[test]
    fn test_sovereign_deployment_build_without_signatures() {
        let config = SovereignDeploymentConfig {
            require_signatures: false,
            ..Default::default()
        };
        let mut recipe = SovereignDeploymentRecipe::new(config);
        recipe.add_model("model.onnx", "sha256_hash", 1000000);

        let result = recipe.build().unwrap();
        assert!(result.success);
        assert_eq!(result.metrics.get("artifact_count"), Some(&1.0));
    }

    #[test]
    fn test_sovereign_deployment_build_with_signatures() {
        let config = SovereignDeploymentConfig {
            require_signatures: true,
            ..Default::default()
        };
        let mut recipe = SovereignDeploymentRecipe::new(config);
        recipe.add_model("model.onnx", "sha256_hash", 1000000);
        recipe.sign_artifact("model.onnx", "key-001");

        let result = recipe.build().unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_sovereign_deployment_missing_signature() {
        let config = SovereignDeploymentConfig {
            require_signatures: true,
            ..Default::default()
        };
        let mut recipe = SovereignDeploymentRecipe::new(config);
        recipe.add_model("model.onnx", "sha256_hash", 1000000);

        let result = recipe.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_sovereign_deployment_export_manifest() {
        let config = SovereignDeploymentConfig {
            require_signatures: false,
            ..Default::default()
        };
        let mut recipe = SovereignDeploymentRecipe::new(config);
        recipe.add_model("model.onnx", "sha256_hash", 1000000);

        let manifest = recipe.export_manifest().unwrap();
        assert!(manifest.contains("sovereign-model"));
        assert!(manifest.contains("model.onnx"));
    }

    // -------------------------------------------------------------------------
    // ResearchArtifactRecipe Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_research_artifact_creation() {
        let recipe = ResearchArtifactRecipe::new(
            "Test Paper",
            "This is the abstract.",
        );
        assert_eq!(recipe.artifact().title, "Test Paper");
    }

    #[test]
    fn test_research_artifact_add_contributor() {
        let mut recipe = ResearchArtifactRecipe::new("Test", "Abstract");
        recipe.add_contributor(
            "Alice Smith",
            "MIT",
            vec![CreditRole::Conceptualization, CreditRole::Software],
        );

        assert_eq!(recipe.artifact().contributors.len(), 1);
        assert_eq!(recipe.artifact().contributors[0].name, "Alice Smith");
    }

    #[test]
    fn test_research_artifact_generate_citation() {
        let mut recipe = ResearchArtifactRecipe::new("Test Paper", "Abstract");
        recipe.add_contributor("Alice Smith", "MIT", vec![CreditRole::Software]);
        recipe.set_doi("10.1234/test");

        let citation = recipe.generate_citation();
        assert_eq!(citation.title, "Test Paper");
        assert_eq!(citation.authors, vec!["Alice Smith"]);
        assert_eq!(citation.doi, Some("10.1234/test".to_string()));
    }

    #[test]
    fn test_research_artifact_build() {
        let mut recipe = ResearchArtifactRecipe::new("Test Paper", "Abstract");
        recipe.add_contributor("Alice Smith", "MIT", vec![CreditRole::Software]);
        recipe.add_keywords(vec!["ML".to_string(), "Rust".to_string()]);
        recipe.set_doi("10.1234/test");

        let result = recipe.build();
        assert!(result.success);
        assert_eq!(result.metrics.get("contributor_count"), Some(&1.0));
        assert_eq!(result.metrics.get("keyword_count"), Some(&2.0));
    }

    // -------------------------------------------------------------------------
    // CiCdBenchmarkRecipe Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cicd_recipe_creation() {
        let recipe = CiCdBenchmarkRecipe::new("ci-benchmark");
        assert_eq!(recipe.name, "ci-benchmark");
    }

    #[test]
    fn test_cicd_recipe_thresholds() {
        let mut recipe = CiCdBenchmarkRecipe::new("ci-benchmark");
        recipe.add_min_performance_threshold("performance", 0.90);
        recipe.add_max_cost_threshold(100.0);

        assert_eq!(recipe.thresholds.get("min_performance"), Some(&0.90));
        assert_eq!(recipe.thresholds.get("max_cost"), Some(&100.0));
    }

    #[test]
    fn test_cicd_recipe_check_pass() {
        let mut recipe = CiCdBenchmarkRecipe::new("ci-benchmark");
        recipe.add_min_performance_threshold("performance", 0.90);
        recipe.add_max_cost_threshold(100.0);

        recipe.add_result(CostPerformancePoint {
            id: "test".to_string(),
            performance: 0.95,
            cost: 80.0,
            energy_joules: 1000.0,
            latency_ms: None,
            metadata: HashMap::new(),
        });

        let result = recipe.check();
        assert!(result.success);
        assert_eq!(result.metrics.get("all_checks_passed"), Some(&1.0));
    }

    #[test]
    fn test_cicd_recipe_check_fail_cost() {
        let mut recipe = CiCdBenchmarkRecipe::new("ci-benchmark");
        recipe.add_max_cost_threshold(100.0);

        recipe.add_result(CostPerformancePoint {
            id: "test".to_string(),
            performance: 0.95,
            cost: 150.0,
            energy_joules: 1000.0,
            latency_ms: None,
            metadata: HashMap::new(),
        });

        let result = recipe.check();
        assert!(!result.success);
        assert_eq!(result.metrics.get("all_checks_passed"), Some(&0.0));
    }

    #[test]
    fn test_cicd_recipe_check_fail_performance() {
        let mut recipe = CiCdBenchmarkRecipe::new("ci-benchmark");
        recipe.add_min_performance_threshold("performance", 0.90);

        recipe.add_result(CostPerformancePoint {
            id: "test".to_string(),
            performance: 0.85,
            cost: 80.0,
            energy_joules: 1000.0,
            latency_ms: None,
            metadata: HashMap::new(),
        });

        let result = recipe.check();
        assert!(!result.success);
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_experiment_workflow() {
        // 1. Create experiment tracking
        let config = ExperimentTrackingConfig {
            experiment_name: "integration-test".to_string(),
            paradigm: ModelParadigm::DeepLearning,
            device: ComputeDevice::Cpu {
                cores: 8,
                threads_per_core: 2,
                architecture: CpuArchitecture::X86_64,
            },
            platform: PlatformEfficiency::Server,
            track_energy: true,
            track_cost: false,
            carbon_intensity: Some(400.0),
            tags: vec!["test".to_string()],
        };
        let mut tracking = ExperimentTrackingRecipe::new(config);

        // 2. Run experiment
        tracking.start_run("run-001");
        tracking.log_metric("accuracy", 0.95).unwrap();
        tracking.log_metric("loss", 0.05).unwrap();
        tracking.log_param("learning_rate", serde_json::json!(0.001)).unwrap();

        let result = tracking.end_run(true).unwrap();
        assert!(result.success);

        // 3. Store in backend
        let storage = InMemoryExperimentStorage::new();
        tracking.store_run(&storage).unwrap();

        // 4. Verify stored
        let runs = storage.list_runs("integration-test").unwrap();
        assert_eq!(runs.len(), 1);
    }

    #[test]
    fn test_benchmark_to_cicd_workflow() {
        // 1. Create benchmark
        let mut benchmark = CostPerformanceBenchmarkRecipe::new("perf-test")
            .with_budget(200.0)
            .with_performance_target(0.85);

        // 2. Add configurations
        benchmark.benchmark_mut().add_point(CostPerformancePoint {
            id: "small-model".to_string(),
            performance: 0.88,
            cost: 50.0,
            energy_joules: 500.0,
            latency_ms: Some(10.0),
            metadata: HashMap::new(),
        });

        benchmark.benchmark_mut().add_point(CostPerformancePoint {
            id: "large-model".to_string(),
            performance: 0.95,
            cost: 150.0,
            energy_joules: 1500.0,
            latency_ms: Some(50.0),
            metadata: HashMap::new(),
        });

        // 3. Analyze
        let analysis = benchmark.analyze();
        assert!(analysis.success);

        // 4. Feed into CI/CD
        let mut cicd = CiCdBenchmarkRecipe::new("ci-gate");
        cicd.add_min_performance_threshold("performance", 0.85);
        cicd.add_max_cost_threshold(200.0);

        for point in benchmark.benchmark().points.iter() {
            cicd.add_result(point.clone());
        }

        let cicd_result = cicd.check();
        assert!(cicd_result.success);
    }
}
