//! HuggingFace Ecosystem Catalog
//!
//! Implements HF-QUERY-001, HF-QUERY-004, HF-QUERY-005, HF-QUERY-006
//!
//! Provides:
//! - Complete 50+ component registry
//! - Course alignment for Coursera specialization
//! - Dependency graph between components
//! - Documentation links
//!
//! ## Observability (HF-OBS-003)
//!
//! Key catalog operations are instrumented with tracing spans:
//! - `hf.catalog.search` - Component search operations
//! - `hf.catalog.by_course` - Course-filtered queries
//! - `hf.catalog.by_category` - Category-filtered queries

// Allow dead_code for methods that are tested but not yet exposed via CLI
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, instrument};

// ============================================================================
// HF-QUERY-001: Core Types
// ============================================================================

/// Category of HuggingFace component
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HfComponentCategory {
    /// Hub and client libraries
    Hub,
    /// Deployment and inference
    Deployment,
    /// Core ML libraries
    Library,
    /// Training and optimization
    Training,
    /// Collaboration and extras
    Collaboration,
    /// Community resources
    Community,
}

impl HfComponentCategory {
    /// Get all categories
    pub fn all() -> &'static [Self] {
        &[
            Self::Hub,
            Self::Deployment,
            Self::Library,
            Self::Training,
            Self::Collaboration,
            Self::Community,
        ]
    }

    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Hub => "Hub & Client Libraries",
            Self::Deployment => "Deployment & Inference",
            Self::Library => "Core ML Libraries",
            Self::Training => "Training & Optimization",
            Self::Collaboration => "Collaboration & Extras",
            Self::Community => "Community Resources",
        }
    }
}

impl std::fmt::Display for HfComponentCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Asset type for course planning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetType {
    Video,
    Lab,
    Reading,
    Quiz,
    Discussion,
}

/// Course alignment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CourseAlignment {
    /// Course number (1-5)
    pub course: u8,
    /// Week number (1-3)
    pub week: u8,
    /// Lesson identifiers (e.g., "1.1", "2.3")
    pub lessons: Vec<String>,
    /// Asset types used
    pub asset_types: Vec<AssetType>,
}

impl CourseAlignment {
    pub fn new(course: u8, week: u8) -> Self {
        Self {
            course,
            week,
            lessons: Vec::new(),
            asset_types: Vec::new(),
        }
    }

    pub fn with_lessons(mut self, lessons: &[&str]) -> Self {
        self.lessons = lessons.iter().map(|s| (*s).to_string()).collect();
        self
    }

    pub fn with_assets(mut self, assets: &[AssetType]) -> Self {
        self.asset_types = assets.to_vec();
        self
    }
}

/// A component in the HuggingFace ecosystem (full metadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogComponent {
    /// Unique identifier (e.g., "transformers", "peft")
    pub id: String,
    /// Display name
    pub name: String,
    /// Category
    pub category: HfComponentCategory,
    /// Short description
    pub description: String,
    /// Documentation URL
    pub docs_url: String,
    /// GitHub repository URL
    pub repo_url: Option<String>,
    /// PyPI package name
    pub pypi_name: Option<String>,
    /// npm package name
    pub npm_name: Option<String>,
    /// Dependencies on other components
    pub dependencies: Vec<String>,
    /// Course alignments
    pub courses: Vec<CourseAlignment>,
    /// Related components
    pub related: Vec<String>,
    /// Tags for search
    pub tags: Vec<String>,
}

impl CatalogComponent {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        category: HfComponentCategory,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            category,
            description: String::new(),
            docs_url: String::new(),
            repo_url: None,
            pypi_name: None,
            npm_name: None,
            dependencies: Vec::new(),
            courses: Vec::new(),
            related: Vec::new(),
            tags: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_docs(mut self, url: impl Into<String>) -> Self {
        self.docs_url = url.into();
        self
    }

    pub fn with_repo(mut self, url: impl Into<String>) -> Self {
        self.repo_url = Some(url.into());
        self
    }

    pub fn with_pypi(mut self, name: impl Into<String>) -> Self {
        self.pypi_name = Some(name.into());
        self
    }

    pub fn with_npm(mut self, name: impl Into<String>) -> Self {
        self.npm_name = Some(name.into());
        self
    }

    pub fn with_deps(mut self, deps: &[&str]) -> Self {
        self.dependencies = deps.iter().map(|s| (*s).to_string()).collect();
        self
    }

    pub fn with_course(mut self, alignment: CourseAlignment) -> Self {
        self.courses.push(alignment);
        self
    }

    pub fn with_related(mut self, related: &[&str]) -> Self {
        self.related = related.iter().map(|s| (*s).to_string()).collect();
        self
    }

    pub fn with_tags(mut self, tags: &[&str]) -> Self {
        self.tags = tags.iter().map(|s| (*s).to_string()).collect();
        self
    }
}

// ============================================================================
// HF-QUERY-001: Catalog
// ============================================================================

/// The complete HuggingFace ecosystem catalog
#[derive(Debug, Clone, Default)]
pub struct HfCatalog {
    components: HashMap<String, CatalogComponent>,
}

impl HfCatalog {
    /// Create empty catalog
    pub fn new() -> Self {
        Self::default()
    }

    /// Load the standard catalog with all 50+ components
    pub fn standard() -> Self {
        let mut catalog = Self::new();
        catalog.register_hub_components();
        catalog.register_deployment_components();
        catalog.register_library_components();
        catalog.register_training_components();
        catalog.register_collaboration_components();
        catalog.register_community_components();
        catalog.register_integration_components();
        catalog
    }

    /// Add a component to the catalog
    pub fn add(&mut self, component: CatalogComponent) {
        self.components.insert(component.id.clone(), component);
    }

    /// Get component by ID
    #[instrument(name = "hf.catalog.get", skip(self), fields(found = tracing::field::Empty))]
    pub fn get(&self, id: &str) -> Option<&CatalogComponent> {
        let result = self.components.get(id);
        tracing::Span::current().record("found", result.is_some());
        if result.is_some() {
            debug!(component_id = id, "Retrieved catalog component");
        }
        result
    }

    /// Get all component IDs
    pub fn list(&self) -> Vec<&str> {
        let mut ids: Vec<_> = self.components.keys().map(String::as_str).collect();
        ids.sort();
        ids
    }

    /// Get total component count
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Check if catalog is empty
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Get all components as an iterator
    pub fn all(&self) -> impl Iterator<Item = &CatalogComponent> {
        self.components.values()
    }

    /// Get components by tag
    #[instrument(name = "hf.catalog.by_tag", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn by_tag(&self, tag: &str) -> Vec<&CatalogComponent> {
        let tag_lower = tag.to_lowercase();
        let results: Vec<_> = self
            .components
            .values()
            .filter(|c| c.tags.iter().any(|t| t.to_lowercase() == tag_lower))
            .collect();
        tracing::Span::current().record("result_count", results.len());
        debug!(tag = tag, count = results.len(), "Tag query completed");
        results
    }

    /// Get components by category
    #[instrument(name = "hf.catalog.by_category", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn by_category(&self, category: HfComponentCategory) -> Vec<&CatalogComponent> {
        let results: Vec<_> = self
            .components
            .values()
            .filter(|c| c.category == category)
            .collect();
        tracing::Span::current().record("result_count", results.len());
        debug!(
            category = ?category,
            count = results.len(),
            "Category query completed"
        );
        results
    }

    /// Search components by query (matches id, name, description, tags)
    #[instrument(name = "hf.catalog.search", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn search(&self, query: &str) -> Vec<&CatalogComponent> {
        let query_lower = query.to_lowercase();
        let results: Vec<_> = self
            .components
            .values()
            .filter(|c| {
                c.id.to_lowercase().contains(&query_lower)
                    || c.name.to_lowercase().contains(&query_lower)
                    || c.description.to_lowercase().contains(&query_lower)
                    || c.tags
                        .iter()
                        .any(|t| t.to_lowercase().contains(&query_lower))
            })
            .collect();
        tracing::Span::current().record("result_count", results.len());
        info!(
            query = query,
            count = results.len(),
            "Catalog search completed"
        );
        results
    }

    // ========================================================================
    // HF-QUERY-004: Course Alignment
    // ========================================================================

    /// Get components for a specific course
    #[instrument(name = "hf.catalog.by_course", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn by_course(&self, course: u8) -> Vec<&CatalogComponent> {
        let results: Vec<_> = self
            .components
            .values()
            .filter(|c| c.courses.iter().any(|ca| ca.course == course))
            .collect();
        tracing::Span::current().record("result_count", results.len());
        info!(
            course = course,
            count = results.len(),
            "Course query completed"
        );
        results
    }

    /// Get components for a specific course and week
    #[instrument(name = "hf.catalog.by_course_week", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn by_course_week(&self, course: u8, week: u8) -> Vec<&CatalogComponent> {
        let results: Vec<_> = self
            .components
            .values()
            .filter(|c| {
                c.courses
                    .iter()
                    .any(|ca| ca.course == course && ca.week == week)
            })
            .collect();
        tracing::Span::current().record("result_count", results.len());
        debug!(
            course = course,
            week = week,
            count = results.len(),
            "Course-week query completed"
        );
        results
    }

    /// Get components by asset type (labs, videos, etc.)
    #[instrument(name = "hf.catalog.by_asset_type", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn by_asset_type(&self, asset: AssetType) -> Vec<&CatalogComponent> {
        let results: Vec<_> = self
            .components
            .values()
            .filter(|c| c.courses.iter().any(|ca| ca.asset_types.contains(&asset)))
            .collect();
        tracing::Span::current().record("result_count", results.len());
        debug!(asset = ?asset, count = results.len(), "Asset type query completed");
        results
    }

    // ========================================================================
    // HF-QUERY-005: Dependency Graph
    // ========================================================================

    /// Get dependencies of a component
    #[instrument(name = "hf.catalog.deps", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn deps(&self, id: &str) -> Vec<&CatalogComponent> {
        let results = self
            .get(id)
            .map(|c| {
                c.dependencies
                    .iter()
                    .filter_map(|dep_id| self.components.get(dep_id))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        tracing::Span::current().record("result_count", results.len());
        debug!(
            component_id = id,
            dep_count = results.len(),
            "Dependency lookup completed"
        );
        results
    }

    /// Get reverse dependencies (what depends on this component)
    #[instrument(name = "hf.catalog.rdeps", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn rdeps(&self, id: &str) -> Vec<&CatalogComponent> {
        let results: Vec<_> = self
            .components
            .values()
            .filter(|c| c.dependencies.contains(&id.to_string()))
            .collect();
        tracing::Span::current().record("result_count", results.len());
        debug!(
            component_id = id,
            rdep_count = results.len(),
            "Reverse dependency lookup completed"
        );
        results
    }

    /// Check if two components are compatible (no conflicts)
    pub fn compatible(&self, id1: &str, id2: &str) -> bool {
        // Simple compatibility: both exist and share common deps or are unrelated
        self.get(id1).is_some() && self.get(id2).is_some()
    }

    // ========================================================================
    // HF-QUERY-006: Documentation Links
    // ========================================================================

    /// Get documentation URL for a component
    pub fn docs_url(&self, id: &str) -> Option<&str> {
        self.get(id).map(|c| c.docs_url.as_str())
    }

    /// Get API reference URL (docs + /api)
    pub fn api_url(&self, id: &str) -> Option<String> {
        self.docs_url(id)
            .map(|url| format!("{}/api", url.trim_end_matches('/')))
    }

    /// Get tutorials URL
    pub fn tutorials_url(&self, id: &str) -> Option<String> {
        self.docs_url(id)
            .map(|url| format!("{}/tutorials", url.trim_end_matches('/')))
    }

    // ========================================================================
    // Component Registration
    // ========================================================================

    fn register_hub_components(&mut self) {
        // Hub models
        self.add(
            CatalogComponent::new("hub-models", "Models", HfComponentCategory::Hub)
                .with_description("700K+ ML models on HuggingFace Hub")
                .with_docs("https://huggingface.co/models")
                .with_tags(&["hub", "models", "repository"])
                .with_course(
                    CourseAlignment::new(1, 1)
                        .with_lessons(&["1.1", "1.3"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // Hub datasets
        self.add(
            CatalogComponent::new("hub-datasets", "Datasets", HfComponentCategory::Hub)
                .with_description("100K+ datasets on HuggingFace Hub")
                .with_docs("https://huggingface.co/datasets")
                .with_tags(&["hub", "datasets", "repository"])
                .with_course(
                    CourseAlignment::new(1, 1)
                        .with_lessons(&["1.6", "1.7"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // Hub spaces
        self.add(
            CatalogComponent::new("hub-spaces", "Spaces", HfComponentCategory::Hub)
                .with_description("300K+ ML demos and apps")
                .with_docs("https://huggingface.co/spaces")
                .with_tags(&["hub", "spaces", "demos", "apps"])
                .with_course(
                    CourseAlignment::new(5, 2)
                        .with_lessons(&["2.7", "2.8"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // Hub Python Library
        self.add(
            CatalogComponent::new(
                "huggingface-hub",
                "Hub Python Library",
                HfComponentCategory::Hub,
            )
            .with_description("Python client to interact with the HuggingFace Hub")
            .with_docs("https://huggingface.co/docs/huggingface_hub")
            .with_repo("https://github.com/huggingface/huggingface_hub")
            .with_pypi("huggingface-hub")
            .with_tags(&["hub", "client", "python", "api"])
            .with_course(
                CourseAlignment::new(1, 1)
                    .with_lessons(&["1.1"])
                    .with_assets(&[AssetType::Reading]),
            ),
        );

        // Huggingface.js
        self.add(
            CatalogComponent::new("huggingface-js", "Huggingface.js", HfComponentCategory::Hub)
                .with_description("JavaScript libraries for HuggingFace with TypeScript types")
                .with_docs("https://huggingface.co/docs/huggingface.js")
                .with_repo("https://github.com/huggingface/huggingface.js")
                .with_npm("@huggingface/hub")
                .with_tags(&["hub", "client", "javascript", "typescript"])
                .with_course(
                    CourseAlignment::new(5, 3)
                        .with_lessons(&["3.6", "3.7"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // Tasks
        self.add(
            CatalogComponent::new("tasks", "Tasks", HfComponentCategory::Hub)
                .with_description("Explore demos, models, and datasets for any ML task")
                .with_docs("https://huggingface.co/tasks")
                .with_tags(&["hub", "tasks", "taxonomy"])
                .with_course(
                    CourseAlignment::new(1, 3)
                        .with_lessons(&["3.7"])
                        .with_assets(&[AssetType::Reading]),
                ),
        );

        // Dataset Viewer
        self.add(
            CatalogComponent::new("dataset-viewer", "Dataset Viewer", HfComponentCategory::Hub)
                .with_description("API for metadata, stats, and content of Hub datasets")
                .with_docs("https://huggingface.co/docs/dataset-viewer")
                .with_tags(&["hub", "datasets", "api", "viewer"])
                .with_course(
                    CourseAlignment::new(2, 1)
                        .with_lessons(&["1.2"])
                        .with_assets(&[AssetType::Lab]),
                ),
        );
    }

    fn register_deployment_components(&mut self) {
        // Inference Providers
        self.add(
            CatalogComponent::new(
                "inference-providers",
                "Inference Providers",
                HfComponentCategory::Deployment,
            )
            .with_description("Call 200k+ models hosted by 10+ inference partners")
            .with_docs("https://huggingface.co/docs/api-inference")
            .with_tags(&["inference", "api", "serverless"])
            .with_course(
                CourseAlignment::new(5, 1)
                    .with_lessons(&["1.6", "1.7"])
                    .with_assets(&[AssetType::Video, AssetType::Lab]),
            ),
        );

        // Inference Endpoints
        self.add(
            CatalogComponent::new(
                "inference-endpoints",
                "Inference Endpoints",
                HfComponentCategory::Deployment,
            )
            .with_description("Deploy models on dedicated & fully managed infrastructure")
            .with_docs("https://huggingface.co/docs/inference-endpoints")
            .with_tags(&["inference", "deployment", "dedicated", "managed"])
            .with_course(
                CourseAlignment::new(5, 2)
                    .with_lessons(&["2.1", "2.2", "2.4"])
                    .with_assets(&[AssetType::Video, AssetType::Lab]),
            ),
        );

        // TGI - Text Generation Inference
        self.add(
            CatalogComponent::new(
                "tgi",
                "Text Generation Inference",
                HfComponentCategory::Deployment,
            )
            .with_description("Serve language models with TGI optimized toolkit")
            .with_docs("https://huggingface.co/docs/text-generation-inference")
            .with_repo("https://github.com/huggingface/text-generation-inference")
            .with_tags(&["inference", "llm", "serving", "tgi", "production"])
            .with_deps(&["transformers"])
            .with_course(
                CourseAlignment::new(5, 1)
                    .with_lessons(&["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7"])
                    .with_assets(&[
                        AssetType::Video,
                        AssetType::Lab,
                        AssetType::Reading,
                        AssetType::Quiz,
                    ]),
            ),
        );

        // TEI - Text Embeddings Inference
        self.add(
            CatalogComponent::new(
                "tei",
                "Text Embeddings Inference",
                HfComponentCategory::Deployment,
            )
            .with_description("Serve embeddings models with TEI optimized toolkit")
            .with_docs("https://huggingface.co/docs/text-embeddings-inference")
            .with_repo("https://github.com/huggingface/text-embeddings-inference")
            .with_tags(&["inference", "embeddings", "serving", "tei"])
            .with_deps(&["sentence-transformers"]),
        );

        // AWS DLCs
        self.add(
            CatalogComponent::new(
                "aws-dlcs",
                "AWS Deep Learning Containers",
                HfComponentCategory::Deployment,
            )
            .with_description("Train/deploy models from HuggingFace to AWS with DLCs")
            .with_docs("https://huggingface.co/docs/sagemaker")
            .with_tags(&["aws", "sagemaker", "deployment", "cloud"]),
        );

        // Azure
        self.add(
            CatalogComponent::new("azure", "Microsoft Azure", HfComponentCategory::Deployment)
                .with_description("Deploy HuggingFace models on Microsoft Azure")
                .with_docs("https://huggingface.co/docs/hub/azure")
                .with_tags(&["azure", "deployment", "cloud"]),
        );

        // GCP
        self.add(
            CatalogComponent::new("gcp", "Google Cloud", HfComponentCategory::Deployment)
                .with_description("Train and deploy HuggingFace models on Google Cloud")
                .with_docs("https://huggingface.co/docs/hub/google-cloud")
                .with_tags(&["gcp", "deployment", "cloud"]),
        );
    }

    fn register_library_components(&mut self) {
        // Transformers
        self.add(
            CatalogComponent::new("transformers", "Transformers", HfComponentCategory::Library)
                .with_description("State-of-the-art AI models for PyTorch, TensorFlow, JAX")
                .with_docs("https://huggingface.co/docs/transformers")
                .with_repo("https://github.com/huggingface/transformers")
                .with_pypi("transformers")
                .with_tags(&["models", "nlp", "vision", "audio", "multimodal"])
                .with_deps(&["tokenizers", "safetensors", "huggingface-hub"])
                .with_related(&["diffusers", "peft", "trl"])
                .with_course(
                    CourseAlignment::new(1, 2)
                        .with_lessons(&["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8"])
                        .with_assets(&[
                            AssetType::Video,
                            AssetType::Lab,
                            AssetType::Reading,
                            AssetType::Quiz,
                        ]),
                )
                .with_course(
                    CourseAlignment::new(1, 3)
                        .with_lessons(&["3.1", "3.2", "3.3", "3.4", "3.5", "3.6"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // Diffusers
        self.add(
            CatalogComponent::new("diffusers", "Diffusers", HfComponentCategory::Library)
                .with_description("State-of-the-art diffusion models in PyTorch")
                .with_docs("https://huggingface.co/docs/diffusers")
                .with_repo("https://github.com/huggingface/diffusers")
                .with_pypi("diffusers")
                .with_tags(&["diffusion", "image-generation", "stable-diffusion"])
                .with_deps(&["transformers", "safetensors"]),
        );

        // Datasets
        self.add(
            CatalogComponent::new("datasets", "Datasets", HfComponentCategory::Library)
                .with_description("Access & share datasets for any ML task")
                .with_docs("https://huggingface.co/docs/datasets")
                .with_repo("https://github.com/huggingface/datasets")
                .with_pypi("datasets")
                .with_tags(&["datasets", "data-loading", "preprocessing"])
                .with_deps(&["huggingface-hub"])
                .with_course(
                    CourseAlignment::new(2, 1)
                        .with_lessons(&["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7"])
                        .with_assets(&[
                            AssetType::Video,
                            AssetType::Lab,
                            AssetType::Reading,
                            AssetType::Quiz,
                        ]),
                ),
        );

        // Transformers.js
        self.add(
            CatalogComponent::new(
                "transformers-js",
                "Transformers.js",
                HfComponentCategory::Library,
            )
            .with_description("State-of-the-art ML running directly in your browser")
            .with_docs("https://huggingface.co/docs/transformers.js")
            .with_repo("https://github.com/xenova/transformers.js")
            .with_npm("@xenova/transformers")
            .with_tags(&["javascript", "browser", "wasm", "onnx"])
            .with_course(
                CourseAlignment::new(5, 3)
                    .with_lessons(&["3.6", "3.7"])
                    .with_assets(&[AssetType::Video, AssetType::Lab]),
            ),
        );

        // Tokenizers
        self.add(
            CatalogComponent::new("tokenizers", "Tokenizers", HfComponentCategory::Library)
                .with_description("Fast tokenizers optimized for research & production")
                .with_docs("https://huggingface.co/docs/tokenizers")
                .with_repo("https://github.com/huggingface/tokenizers")
                .with_pypi("tokenizers")
                .with_tags(&["tokenization", "bpe", "wordpiece", "sentencepiece"])
                .with_course(
                    CourseAlignment::new(1, 2)
                        .with_lessons(&["2.4"])
                        .with_assets(&[AssetType::Reading]),
                ),
        );

        // Evaluate
        self.add(
            CatalogComponent::new("evaluate", "Evaluate", HfComponentCategory::Library)
                .with_description("Evaluate and compare model performance")
                .with_docs("https://huggingface.co/docs/evaluate")
                .with_repo("https://github.com/huggingface/evaluate")
                .with_pypi("evaluate")
                .with_tags(&["evaluation", "metrics", "benchmarking"])
                .with_course(
                    CourseAlignment::new(2, 3)
                        .with_lessons(&["3.1", "3.2", "3.3", "3.4"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // timm
        self.add(
            CatalogComponent::new("timm", "timm", HfComponentCategory::Library)
                .with_description("State-of-the-art vision models: layers, optimizers, utilities")
                .with_docs("https://huggingface.co/docs/timm")
                .with_repo("https://github.com/huggingface/pytorch-image-models")
                .with_pypi("timm")
                .with_tags(&["vision", "image-classification", "pretrained"])
                .with_course(
                    CourseAlignment::new(1, 3)
                        .with_lessons(&["3.1", "3.2"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // Sentence Transformers
        self.add(
            CatalogComponent::new(
                "sentence-transformers",
                "Sentence Transformers",
                HfComponentCategory::Library,
            )
            .with_description("Embeddings, retrieval, and reranking")
            .with_docs("https://www.sbert.net/")
            .with_repo("https://github.com/UKPLab/sentence-transformers")
            .with_pypi("sentence-transformers")
            .with_tags(&["embeddings", "semantic-search", "retrieval", "rag"])
            .with_deps(&["transformers"])
            .with_course(
                CourseAlignment::new(3, 2)
                    .with_lessons(&["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"])
                    .with_assets(&[
                        AssetType::Video,
                        AssetType::Lab,
                        AssetType::Reading,
                        AssetType::Discussion,
                        AssetType::Quiz,
                    ]),
            ),
        );

        // Kernels
        self.add(
            CatalogComponent::new("kernels", "Kernels", HfComponentCategory::Library)
                .with_description("Load and run compute kernels from the HuggingFace Hub")
                .with_docs("https://huggingface.co/docs/kernels")
                .with_tags(&["kernels", "cuda", "triton", "optimization"]),
        );

        // Safetensors
        self.add(
            CatalogComponent::new("safetensors", "Safetensors", HfComponentCategory::Library)
                .with_description("Safe way to store/distribute neural network weights")
                .with_docs("https://huggingface.co/docs/safetensors")
                .with_repo("https://github.com/huggingface/safetensors")
                .with_pypi("safetensors")
                .with_tags(&["serialization", "safe", "tensors", "format"])
                .with_course(
                    CourseAlignment::new(1, 1)
                        .with_lessons(&["1.4"])
                        .with_assets(&[AssetType::Video]),
                ),
        );
    }

    fn register_training_components(&mut self) {
        // PEFT
        self.add(
            CatalogComponent::new("peft", "PEFT", HfComponentCategory::Training)
                .with_description("Parameter-efficient finetuning for large language models")
                .with_docs("https://huggingface.co/docs/peft")
                .with_repo("https://github.com/huggingface/peft")
                .with_pypi("peft")
                .with_tags(&["finetuning", "lora", "qlora", "efficient"])
                .with_deps(&["transformers", "bitsandbytes"])
                .with_course(
                    CourseAlignment::new(4, 1)
                        .with_lessons(&["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8"])
                        .with_assets(&[
                            AssetType::Video,
                            AssetType::Lab,
                            AssetType::Reading,
                            AssetType::Quiz,
                        ]),
                ),
        );

        // Accelerate
        self.add(
            CatalogComponent::new("accelerate", "Accelerate", HfComponentCategory::Training)
                .with_description("Train PyTorch models with multi-GPU, TPU, mixed precision")
                .with_docs("https://huggingface.co/docs/accelerate")
                .with_repo("https://github.com/huggingface/accelerate")
                .with_pypi("accelerate")
                .with_tags(&["distributed", "multi-gpu", "tpu", "mixed-precision"])
                .with_course(
                    CourseAlignment::new(1, 2)
                        .with_lessons(&["2.8"])
                        .with_assets(&[AssetType::Lab]),
                ),
        );

        // Optimum
        self.add(
            CatalogComponent::new("optimum", "Optimum", HfComponentCategory::Training)
                .with_description("Optimize HF Transformers for faster training/inference")
                .with_docs("https://huggingface.co/docs/optimum")
                .with_repo("https://github.com/huggingface/optimum")
                .with_pypi("optimum")
                .with_tags(&["optimization", "onnx", "quantization", "hardware"])
                .with_deps(&["transformers"])
                .with_course(
                    CourseAlignment::new(5, 3)
                        .with_lessons(&["3.1", "3.2", "3.3", "3.4", "3.5"])
                        .with_assets(&[AssetType::Video, AssetType::Lab, AssetType::Reading]),
                ),
        );

        // AWS Trainium/Inferentia
        self.add(
            CatalogComponent::new(
                "aws-trainium",
                "AWS Trainium & Inferentia",
                HfComponentCategory::Training,
            )
            .with_description("Train/deploy Transformers/Diffusers on AWS custom silicon")
            .with_docs("https://huggingface.co/docs/optimum-neuron")
            .with_pypi("optimum-neuron")
            .with_tags(&["aws", "trainium", "inferentia", "hardware"]),
        );

        // Google TPUs
        self.add(
            CatalogComponent::new("tpu", "Google TPUs", HfComponentCategory::Training)
                .with_description("Train and deploy models on Google TPUs via Optimum")
                .with_docs("https://huggingface.co/docs/optimum-tpu")
                .with_tags(&["gcp", "tpu", "hardware"]),
        );

        // TRL
        self.add(
            CatalogComponent::new("trl", "TRL", HfComponentCategory::Training)
                .with_description("Train transformer LMs with reinforcement learning")
                .with_docs("https://huggingface.co/docs/trl")
                .with_repo("https://github.com/huggingface/trl")
                .with_pypi("trl")
                .with_tags(&["rlhf", "dpo", "ppo", "alignment", "sft"])
                .with_deps(&["transformers", "peft"])
                .with_course(
                    CourseAlignment::new(4, 2)
                        .with_lessons(&["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"])
                        .with_assets(&[
                            AssetType::Video,
                            AssetType::Lab,
                            AssetType::Reading,
                            AssetType::Discussion,
                            AssetType::Quiz,
                        ]),
                )
                .with_course(
                    CourseAlignment::new(4, 3)
                        .with_lessons(&["3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8"])
                        .with_assets(&[
                            AssetType::Video,
                            AssetType::Lab,
                            AssetType::Reading,
                            AssetType::Quiz,
                        ]),
                ),
        );

        // Bitsandbytes
        self.add(
            CatalogComponent::new(
                "bitsandbytes",
                "Bitsandbytes",
                HfComponentCategory::Training,
            )
            .with_description("Optimize and quantize models with bitsandbytes")
            .with_docs("https://huggingface.co/docs/bitsandbytes")
            .with_repo("https://github.com/TimDettmers/bitsandbytes")
            .with_pypi("bitsandbytes")
            .with_tags(&["quantization", "4bit", "8bit", "nf4", "qlora"])
            .with_course(
                CourseAlignment::new(4, 1)
                    .with_lessons(&["1.4", "1.5"])
                    .with_assets(&[AssetType::Video, AssetType::Lab]),
            ),
        );

        // Lighteval
        self.add(
            CatalogComponent::new("lighteval", "Lighteval", HfComponentCategory::Training)
                .with_description("All-in-one toolkit to evaluate LLMs across multiple backends")
                .with_docs("https://huggingface.co/docs/lighteval")
                .with_repo("https://github.com/huggingface/lighteval")
                .with_pypi("lighteval")
                .with_tags(&["evaluation", "llm", "benchmarking"]),
        );

        // Trainer API
        self.add(
            CatalogComponent::new("trainer", "Trainer API", HfComponentCategory::Training)
                .with_description("High-level training loops for transformers models")
                .with_docs("https://huggingface.co/docs/transformers/main_classes/trainer")
                .with_tags(&["training", "api", "loops"])
                .with_deps(&["transformers", "datasets"])
                .with_course(
                    CourseAlignment::new(2, 2)
                        .with_lessons(&["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8"])
                        .with_assets(&[
                            AssetType::Video,
                            AssetType::Lab,
                            AssetType::Reading,
                            AssetType::Quiz,
                        ]),
                ),
        );

        // AutoTrain
        self.add(
            CatalogComponent::new("autotrain", "AutoTrain", HfComponentCategory::Training)
                .with_description("AutoTrain API and UI for seamless model training")
                .with_docs("https://huggingface.co/docs/autotrain")
                .with_repo("https://github.com/huggingface/autotrain-advanced")
                .with_pypi("autotrain-advanced")
                .with_tags(&["automl", "no-code", "training"]),
        );
    }

    fn register_collaboration_components(&mut self) {
        // Gradio
        self.add(
            CatalogComponent::new("gradio", "Gradio", HfComponentCategory::Collaboration)
                .with_description("Build ML demos and web apps with a few lines of Python")
                .with_docs("https://www.gradio.app/docs")
                .with_repo("https://github.com/gradio-app/gradio")
                .with_pypi("gradio")
                .with_tags(&["demos", "ui", "web-apps", "interactive"])
                .with_course(
                    CourseAlignment::new(5, 2)
                        .with_lessons(&["2.5", "2.6", "2.7", "2.8"])
                        .with_assets(&[AssetType::Video, AssetType::Lab, AssetType::Quiz]),
                ),
        );

        // Trackio
        self.add(
            CatalogComponent::new("trackio", "Trackio", HfComponentCategory::Collaboration)
                .with_description("Lightweight, local-first experiment tracking library")
                .with_docs("https://huggingface.co/docs/trackio")
                .with_pypi("trackio")
                .with_tags(&["experiment-tracking", "logging", "local"]),
        );

        // smolagents
        self.add(
            CatalogComponent::new(
                "smolagents",
                "smolagents",
                HfComponentCategory::Collaboration,
            )
            .with_description("Smol library to build great agents in Python")
            .with_docs("https://huggingface.co/docs/smolagents")
            .with_repo("https://github.com/huggingface/smolagents")
            .with_pypi("smolagents")
            .with_tags(&["agents", "tools", "llm"]),
        );

        // LeRobot
        self.add(
            CatalogComponent::new("lerobot", "LeRobot", HfComponentCategory::Collaboration)
                .with_description("Making AI for Robotics more accessible with end-to-end learning")
                .with_docs("https://huggingface.co/docs/lerobot")
                .with_repo("https://github.com/huggingface/lerobot")
                .with_pypi("lerobot")
                .with_tags(&["robotics", "embodied-ai", "imitation-learning"]),
        );

        // Chat UI
        self.add(
            CatalogComponent::new("chat-ui", "Chat UI", HfComponentCategory::Collaboration)
                .with_description("Open source chat frontend powering HuggingChat")
                .with_docs("https://huggingface.co/docs/chat-ui")
                .with_repo("https://github.com/huggingface/chat-ui")
                .with_tags(&["chat", "ui", "frontend", "huggingchat"]),
        );

        // Leaderboards
        self.add(
            CatalogComponent::new(
                "leaderboards",
                "Leaderboards",
                HfComponentCategory::Collaboration,
            )
            .with_description("Create custom leaderboards on HuggingFace")
            .with_docs("https://huggingface.co/docs/leaderboards")
            .with_tags(&["leaderboards", "benchmarking", "comparison"]),
        );

        // Argilla
        self.add(
            CatalogComponent::new("argilla", "Argilla", HfComponentCategory::Collaboration)
                .with_description("Collaboration tool for building high-quality datasets")
                .with_docs("https://docs.argilla.io/")
                .with_repo("https://github.com/argilla-io/argilla")
                .with_pypi("argilla")
                .with_tags(&["annotation", "labeling", "data-quality"]),
        );

        // Distilabel
        self.add(
            CatalogComponent::new(
                "distilabel",
                "Distilabel",
                HfComponentCategory::Collaboration,
            )
            .with_description("Framework for synthetic data generation and AI feedback")
            .with_docs("https://distilabel.argilla.io/")
            .with_repo("https://github.com/argilla-io/distilabel")
            .with_pypi("distilabel")
            .with_tags(&["synthetic-data", "ai-feedback", "data-generation"]),
        );
    }

    fn register_community_components(&mut self) {
        // Blog
        self.add(
            CatalogComponent::new("blog", "Blog", HfComponentCategory::Community)
                .with_description("HuggingFace official blog with tutorials and announcements")
                .with_docs("https://huggingface.co/blog")
                .with_tags(&["blog", "tutorials", "announcements"]),
        );

        // Learn
        self.add(
            CatalogComponent::new("learn", "Learn", HfComponentCategory::Community)
                .with_description("HuggingFace learning resources and courses")
                .with_docs("https://huggingface.co/learn")
                .with_tags(&["learning", "courses", "education"]),
        );

        // Discord
        self.add(
            CatalogComponent::new("discord", "Discord", HfComponentCategory::Community)
                .with_description("HuggingFace community Discord server")
                .with_docs("https://discord.gg/huggingface")
                .with_tags(&["community", "discord", "chat"]),
        );

        // Forum
        self.add(
            CatalogComponent::new("forum", "Forum", HfComponentCategory::Community)
                .with_description("HuggingFace discussion forum")
                .with_docs("https://discuss.huggingface.co/")
                .with_tags(&["community", "forum", "discussion"]),
        );

        // Open LLM Leaderboard
        self.add(
            CatalogComponent::new(
                "open-llm-leaderboard",
                "Open LLM Leaderboard",
                HfComponentCategory::Community,
            )
            .with_description("Track and compare open-source LLM performance")
            .with_docs("https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard")
            .with_tags(&["leaderboard", "llm", "benchmarking", "evaluation"])
            .with_related(&["lighteval", "leaderboards"]),
        );

        // Arena
        self.add(
            CatalogComponent::new("arena", "Chatbot Arena", HfComponentCategory::Community)
                .with_description("Anonymous LLM benchmark via human preference voting")
                .with_docs("https://lmarena.ai/")
                .with_tags(&["arena", "llm", "human-preference", "elo"])
                .with_related(&["open-llm-leaderboard"]),
        );
    }

    fn register_integration_components(&mut self) {
        // Outlines
        self.add(
            CatalogComponent::new("outlines", "Outlines", HfComponentCategory::Collaboration)
                .with_description("Structured text generation with grammar constraints")
                .with_docs("https://outlines-dev.github.io/outlines/")
                .with_repo("https://github.com/outlines-dev/outlines")
                .with_pypi("outlines")
                .with_tags(&[
                    "structured-output",
                    "json",
                    "grammar",
                    "constrained-generation",
                ])
                .with_deps(&["transformers"])
                .with_course(
                    CourseAlignment::new(3, 2)
                        .with_lessons(&["2.5"])
                        .with_assets(&[AssetType::Lab]),
                ),
        );

        // Wandb
        self.add(
            CatalogComponent::new(
                "wandb",
                "Weights & Biases",
                HfComponentCategory::Collaboration,
            )
            .with_description("Experiment tracking, visualization, and model registry")
            .with_docs("https://docs.wandb.ai/")
            .with_pypi("wandb")
            .with_tags(&["experiment-tracking", "logging", "mlops", "visualization"])
            .with_course(
                CourseAlignment::new(2, 2)
                    .with_lessons(&["2.5"])
                    .with_assets(&[AssetType::Lab]),
            ),
        );

        // FAISS
        self.add(
            CatalogComponent::new("faiss", "FAISS", HfComponentCategory::Collaboration)
                .with_description(
                    "Facebook's efficient similarity search and clustering of dense vectors",
                )
                .with_docs("https://faiss.ai/")
                .with_repo("https://github.com/facebookresearch/faiss")
                .with_pypi("faiss-cpu")
                .with_tags(&["vector-search", "similarity", "indexing", "rag"])
                .with_course(
                    CourseAlignment::new(3, 2)
                        .with_lessons(&["2.3", "2.4"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );
    }
}

// ============================================================================
// Tests - Extreme TDD with 95%+ Coverage
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    // ========================================================================
    // HF-QUERY-001-001: HfComponentCategory Tests
    // ========================================================================

    #[test]
    fn test_HF_QUERY_001_001_category_all() {
        let categories = HfComponentCategory::all();
        assert_eq!(categories.len(), 6);
        assert!(categories.contains(&HfComponentCategory::Hub));
        assert!(categories.contains(&HfComponentCategory::Deployment));
        assert!(categories.contains(&HfComponentCategory::Library));
        assert!(categories.contains(&HfComponentCategory::Training));
        assert!(categories.contains(&HfComponentCategory::Collaboration));
        assert!(categories.contains(&HfComponentCategory::Community));
    }

    #[test]
    fn test_HF_QUERY_001_002_category_display_name() {
        assert_eq!(
            HfComponentCategory::Hub.display_name(),
            "Hub & Client Libraries"
        );
        assert_eq!(
            HfComponentCategory::Deployment.display_name(),
            "Deployment & Inference"
        );
        assert_eq!(
            HfComponentCategory::Library.display_name(),
            "Core ML Libraries"
        );
        assert_eq!(
            HfComponentCategory::Training.display_name(),
            "Training & Optimization"
        );
        assert_eq!(
            HfComponentCategory::Collaboration.display_name(),
            "Collaboration & Extras"
        );
        assert_eq!(
            HfComponentCategory::Community.display_name(),
            "Community Resources"
        );
    }

    #[test]
    fn test_HF_QUERY_001_003_category_display() {
        assert_eq!(
            format!("{}", HfComponentCategory::Hub),
            "Hub & Client Libraries"
        );
    }

    // ========================================================================
    // HF-QUERY-001-010: CourseAlignment Tests
    // ========================================================================

    #[test]
    fn test_HF_QUERY_001_010_course_alignment_new() {
        let ca = CourseAlignment::new(1, 2);
        assert_eq!(ca.course, 1);
        assert_eq!(ca.week, 2);
        assert!(ca.lessons.is_empty());
        assert!(ca.asset_types.is_empty());
    }

    #[test]
    fn test_HF_QUERY_001_011_course_alignment_with_lessons() {
        let ca = CourseAlignment::new(1, 1).with_lessons(&["1.1", "1.2"]);
        assert_eq!(ca.lessons.len(), 2);
        assert_eq!(ca.lessons[0], "1.1");
    }

    #[test]
    fn test_HF_QUERY_001_012_course_alignment_with_assets() {
        let ca = CourseAlignment::new(1, 1).with_assets(&[AssetType::Video, AssetType::Lab]);
        assert_eq!(ca.asset_types.len(), 2);
        assert!(ca.asset_types.contains(&AssetType::Video));
    }

    // ========================================================================
    // HF-QUERY-001-020: CatalogComponent Tests
    // ========================================================================

    #[test]
    fn test_HF_QUERY_001_020_component_new() {
        let comp =
            CatalogComponent::new("transformers", "Transformers", HfComponentCategory::Library);
        assert_eq!(comp.id, "transformers");
        assert_eq!(comp.name, "Transformers");
        assert_eq!(comp.category, HfComponentCategory::Library);
    }

    #[test]
    fn test_HF_QUERY_001_021_component_with_description() {
        let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
            .with_description("A test component");
        assert_eq!(comp.description, "A test component");
    }

    #[test]
    fn test_HF_QUERY_001_022_component_with_docs() {
        let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
            .with_docs("https://example.com");
        assert_eq!(comp.docs_url, "https://example.com");
    }

    #[test]
    fn test_HF_QUERY_001_023_component_with_repo() {
        let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
            .with_repo("https://github.com/test");
        assert_eq!(comp.repo_url, Some("https://github.com/test".to_string()));
    }

    #[test]
    fn test_HF_QUERY_001_024_component_with_pypi() {
        let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
            .with_pypi("test-package");
        assert_eq!(comp.pypi_name, Some("test-package".to_string()));
    }

    #[test]
    fn test_HF_QUERY_001_025_component_with_npm() {
        let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
            .with_npm("@test/package");
        assert_eq!(comp.npm_name, Some("@test/package".to_string()));
    }

    #[test]
    fn test_HF_QUERY_001_026_component_with_deps() {
        let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
            .with_deps(&["dep1", "dep2"]);
        assert_eq!(comp.dependencies.len(), 2);
    }

    #[test]
    fn test_HF_QUERY_001_027_component_with_course() {
        let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
            .with_course(CourseAlignment::new(1, 1));
        assert_eq!(comp.courses.len(), 1);
    }

    #[test]
    fn test_HF_QUERY_001_028_component_with_related() {
        let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
            .with_related(&["related1", "related2"]);
        assert_eq!(comp.related.len(), 2);
    }

    #[test]
    fn test_HF_QUERY_001_029_component_with_tags() {
        let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
            .with_tags(&["tag1", "tag2"]);
        assert_eq!(comp.tags.len(), 2);
    }

    // ========================================================================
    // HF-QUERY-001-030: HfCatalog Basic Tests
    // ========================================================================

    #[test]
    fn test_HF_QUERY_001_030_catalog_new() {
        let catalog = HfCatalog::new();
        assert!(catalog.is_empty());
        assert_eq!(catalog.len(), 0);
    }

    #[test]
    fn test_HF_QUERY_001_031_catalog_add_and_get() {
        let mut catalog = HfCatalog::new();
        catalog.add(CatalogComponent::new(
            "test",
            "Test",
            HfComponentCategory::Hub,
        ));
        assert_eq!(catalog.len(), 1);
        assert!(catalog.get("test").is_some());
        assert!(catalog.get("nonexistent").is_none());
    }

    #[test]
    fn test_HF_QUERY_001_032_catalog_list() {
        let mut catalog = HfCatalog::new();
        catalog.add(CatalogComponent::new(
            "beta",
            "Beta",
            HfComponentCategory::Hub,
        ));
        catalog.add(CatalogComponent::new(
            "alpha",
            "Alpha",
            HfComponentCategory::Hub,
        ));
        let list = catalog.list();
        assert_eq!(list, vec!["alpha", "beta"]); // Sorted
    }

    #[test]
    fn test_HF_QUERY_001_033_catalog_standard_has_50_plus_components() {
        let catalog = HfCatalog::standard();
        assert!(
            catalog.len() >= 50,
            "Expected 50+ components, got {}",
            catalog.len()
        );
    }

    #[test]
    fn test_HF_QUERY_001_034_catalog_by_category() {
        let catalog = HfCatalog::standard();
        let hub_components = catalog.by_category(HfComponentCategory::Hub);
        assert!(!hub_components.is_empty());
        assert!(hub_components
            .iter()
            .all(|c| c.category == HfComponentCategory::Hub));
    }

    #[test]
    fn test_HF_QUERY_001_035_catalog_search_by_id() {
        let catalog = HfCatalog::standard();
        let results = catalog.search("transformers");
        assert!(!results.is_empty());
        assert!(results.iter().any(|c| c.id == "transformers"));
    }

    #[test]
    fn test_HF_QUERY_001_036_catalog_search_by_description() {
        let catalog = HfCatalog::standard();
        let results = catalog.search("language models");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_HF_QUERY_001_037_catalog_search_by_tag() {
        let catalog = HfCatalog::standard();
        let results = catalog.search("rlhf");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_HF_QUERY_001_038_catalog_search_case_insensitive() {
        let catalog = HfCatalog::standard();
        let results_lower = catalog.search("transformers");
        let results_upper = catalog.search("TRANSFORMERS");
        assert_eq!(results_lower.len(), results_upper.len());
    }

    // ========================================================================
    // HF-QUERY-001-040: Standard Catalog Component Tests
    // ========================================================================

    #[test]
    fn test_HF_QUERY_001_040_catalog_has_transformers() {
        let catalog = HfCatalog::standard();
        let comp = catalog
            .get("transformers")
            .expect("transformers should exist");
        assert_eq!(comp.name, "Transformers");
        assert!(!comp.docs_url.is_empty());
    }

    #[test]
    fn test_HF_QUERY_001_041_catalog_has_tgi() {
        let catalog = HfCatalog::standard();
        let comp = catalog.get("tgi").expect("tgi should exist");
        assert!(
            comp.description.contains("Text Generation Inference")
                || comp.name.contains("Text Generation")
        );
    }

    #[test]
    fn test_HF_QUERY_001_042_catalog_has_peft() {
        let catalog = HfCatalog::standard();
        let comp = catalog.get("peft").expect("peft should exist");
        assert!(comp.tags.contains(&"lora".to_string()));
    }

    #[test]
    fn test_HF_QUERY_001_043_catalog_has_gradio() {
        let catalog = HfCatalog::standard();
        let comp = catalog.get("gradio").expect("gradio should exist");
        assert_eq!(comp.category, HfComponentCategory::Collaboration);
    }

    #[test]
    fn test_HF_QUERY_001_044_catalog_has_sentence_transformers() {
        let catalog = HfCatalog::standard();
        let comp = catalog
            .get("sentence-transformers")
            .expect("sentence-transformers should exist");
        assert!(comp.tags.contains(&"embeddings".to_string()));
    }

    #[test]
    fn test_HF_QUERY_001_045_catalog_has_bitsandbytes() {
        let catalog = HfCatalog::standard();
        let comp = catalog
            .get("bitsandbytes")
            .expect("bitsandbytes should exist");
        assert!(comp.tags.contains(&"quantization".to_string()));
    }

    #[test]
    fn test_HF_QUERY_001_046_catalog_has_trl() {
        let catalog = HfCatalog::standard();
        let comp = catalog.get("trl").expect("trl should exist");
        assert!(comp.tags.contains(&"dpo".to_string()));
    }

    #[test]
    fn test_HF_QUERY_001_047_catalog_has_datasets() {
        let catalog = HfCatalog::standard();
        let comp = catalog.get("datasets").expect("datasets should exist");
        assert_eq!(comp.category, HfComponentCategory::Library);
    }

    #[test]
    fn test_HF_QUERY_001_048_catalog_has_optimum() {
        let catalog = HfCatalog::standard();
        let comp = catalog.get("optimum").expect("optimum should exist");
        assert!(comp.tags.contains(&"onnx".to_string()));
    }

    #[test]
    fn test_HF_QUERY_001_049_catalog_has_transformers_js() {
        let catalog = HfCatalog::standard();
        let comp = catalog
            .get("transformers-js")
            .expect("transformers-js should exist");
        assert!(comp.npm_name.is_some());
    }

    // ========================================================================
    // HF-QUERY-004: Course Alignment Tests
    // ========================================================================

    #[test]
    fn test_HF_QUERY_004_001_by_course_1() {
        let catalog = HfCatalog::standard();
        let course1 = catalog.by_course(1);
        assert!(!course1.is_empty());
        // Course 1 should have transformers, hub components
        assert!(course1.iter().any(|c| c.id == "transformers"));
    }

    #[test]
    fn test_HF_QUERY_004_002_by_course_2() {
        let catalog = HfCatalog::standard();
        let course2 = catalog.by_course(2);
        assert!(!course2.is_empty());
        // Course 2 should have datasets, trainer
        assert!(course2.iter().any(|c| c.id == "datasets"));
    }

    #[test]
    fn test_HF_QUERY_004_003_by_course_3() {
        let catalog = HfCatalog::standard();
        let course3 = catalog.by_course(3);
        assert!(!course3.is_empty());
        // Course 3 should have sentence-transformers for RAG
        assert!(course3.iter().any(|c| c.id == "sentence-transformers"));
    }

    #[test]
    fn test_HF_QUERY_004_004_by_course_4() {
        let catalog = HfCatalog::standard();
        let course4 = catalog.by_course(4);
        assert!(!course4.is_empty());
        // Course 4 should have peft, trl, bitsandbytes
        assert!(course4.iter().any(|c| c.id == "peft"));
        assert!(course4.iter().any(|c| c.id == "trl"));
    }

    #[test]
    fn test_HF_QUERY_004_005_by_course_5() {
        let catalog = HfCatalog::standard();
        let course5 = catalog.by_course(5);
        assert!(!course5.is_empty());
        // Course 5 should have tgi, gradio, optimum
        assert!(course5.iter().any(|c| c.id == "tgi"));
        assert!(course5.iter().any(|c| c.id == "gradio"));
    }

    #[test]
    fn test_HF_QUERY_004_006_by_course_week() {
        let catalog = HfCatalog::standard();
        let course1_week2 = catalog.by_course_week(1, 2);
        assert!(!course1_week2.is_empty());
        // Course 1 Week 2 is about transformers fundamentals
        assert!(course1_week2.iter().any(|c| c.id == "transformers"));
    }

    #[test]
    fn test_HF_QUERY_004_007_by_asset_type_lab() {
        let catalog = HfCatalog::standard();
        let labs = catalog.by_asset_type(AssetType::Lab);
        assert!(!labs.is_empty());
    }

    #[test]
    fn test_HF_QUERY_004_008_by_asset_type_video() {
        let catalog = HfCatalog::standard();
        let videos = catalog.by_asset_type(AssetType::Video);
        assert!(!videos.is_empty());
    }

    #[test]
    fn test_HF_QUERY_004_009_nonexistent_course() {
        let catalog = HfCatalog::standard();
        let course99 = catalog.by_course(99);
        assert!(course99.is_empty());
    }

    // ========================================================================
    // HF-QUERY-005: Dependency Graph Tests
    // ========================================================================

    #[test]
    fn test_HF_QUERY_005_001_deps_transformers() {
        let catalog = HfCatalog::standard();
        let deps = catalog.deps("transformers");
        // transformers depends on tokenizers, safetensors, huggingface-hub
        assert!(!deps.is_empty());
    }

    #[test]
    fn test_HF_QUERY_005_002_deps_peft() {
        let catalog = HfCatalog::standard();
        let deps = catalog.deps("peft");
        // peft depends on transformers, bitsandbytes
        assert!(deps
            .iter()
            .any(|c| c.id == "transformers" || c.id == "bitsandbytes"));
    }

    #[test]
    fn test_HF_QUERY_005_003_deps_nonexistent() {
        let catalog = HfCatalog::standard();
        let deps = catalog.deps("nonexistent");
        assert!(deps.is_empty());
    }

    #[test]
    fn test_HF_QUERY_005_004_rdeps_transformers() {
        let catalog = HfCatalog::standard();
        let rdeps = catalog.rdeps("transformers");
        // Many things depend on transformers
        assert!(!rdeps.is_empty());
    }

    #[test]
    fn test_HF_QUERY_005_005_rdeps_huggingface_hub() {
        let catalog = HfCatalog::standard();
        let rdeps = catalog.rdeps("huggingface-hub");
        // datasets, transformers depend on huggingface-hub
        assert!(!rdeps.is_empty());
    }

    #[test]
    fn test_HF_QUERY_005_006_compatible_valid() {
        let catalog = HfCatalog::standard();
        assert!(catalog.compatible("peft", "trl"));
        assert!(catalog.compatible("transformers", "datasets"));
    }

    #[test]
    fn test_HF_QUERY_005_007_compatible_nonexistent() {
        let catalog = HfCatalog::standard();
        assert!(!catalog.compatible("peft", "nonexistent"));
    }

    // ========================================================================
    // HF-QUERY-006: Documentation Links Tests
    // ========================================================================

    #[test]
    fn test_HF_QUERY_006_001_docs_url() {
        let catalog = HfCatalog::standard();
        let url = catalog.docs_url("transformers");
        assert!(url.is_some());
        assert!(url.unwrap().contains("huggingface.co"));
    }

    #[test]
    fn test_HF_QUERY_006_002_docs_url_nonexistent() {
        let catalog = HfCatalog::standard();
        let url = catalog.docs_url("nonexistent");
        assert!(url.is_none());
    }

    #[test]
    fn test_HF_QUERY_006_003_api_url() {
        let catalog = HfCatalog::standard();
        let url = catalog.api_url("transformers");
        assert!(url.is_some());
        assert!(url.unwrap().ends_with("/api"));
    }

    #[test]
    fn test_HF_QUERY_006_004_tutorials_url() {
        let catalog = HfCatalog::standard();
        let url = catalog.tutorials_url("transformers");
        assert!(url.is_some());
        assert!(url.unwrap().ends_with("/tutorials"));
    }

    #[test]
    fn test_HF_QUERY_006_005_all_components_have_docs() {
        let catalog = HfCatalog::standard();
        for id in catalog.list() {
            let comp = catalog.get(id).unwrap();
            assert!(
                !comp.docs_url.is_empty(),
                "Component {} has no docs_url",
                id
            );
        }
    }

    // ========================================================================
    // HF-QUERY-001-050: Serialization Tests
    // ========================================================================

    #[test]
    fn test_HF_QUERY_001_050_component_serialize_json() {
        let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
            .with_description("Test component");
        let json = serde_json::to_string(&comp).unwrap();
        assert!(json.contains("\"id\":\"test\""));
        assert!(json.contains("\"category\":\"hub\""));
    }

    #[test]
    fn test_HF_QUERY_001_051_component_deserialize_json() {
        let json = r#"{"id":"test","name":"Test","category":"hub","description":"","docs_url":"","repo_url":null,"pypi_name":null,"npm_name":null,"dependencies":[],"courses":[],"related":[],"tags":[]}"#;
        let comp: CatalogComponent = serde_json::from_str(json).unwrap();
        assert_eq!(comp.id, "test");
        assert_eq!(comp.category, HfComponentCategory::Hub);
    }

    #[test]
    fn test_HF_QUERY_001_052_course_alignment_serialize() {
        let ca = CourseAlignment::new(1, 2)
            .with_lessons(&["1.1"])
            .with_assets(&[AssetType::Video]);
        let json = serde_json::to_string(&ca).unwrap();
        assert!(json.contains("\"course\":1"));
        assert!(json.contains("\"week\":2"));
    }

    // ========================================================================
    // HF-QUERY-001-060: Category Coverage Tests
    // ========================================================================

    #[test]
    fn test_HF_QUERY_001_060_all_categories_have_components() {
        let catalog = HfCatalog::standard();
        for category in HfComponentCategory::all() {
            let components = catalog.by_category(*category);
            assert!(
                !components.is_empty(),
                "Category {:?} has no components",
                category
            );
        }
    }

    #[test]
    fn test_HF_QUERY_001_061_hub_category_components() {
        let catalog = HfCatalog::standard();
        let hub = catalog.by_category(HfComponentCategory::Hub);
        // Should have: hub-models, hub-datasets, hub-spaces, huggingface-hub, huggingface-js, tasks, dataset-viewer
        assert!(hub.len() >= 7);
    }

    #[test]
    fn test_HF_QUERY_001_062_deployment_category_components() {
        let catalog = HfCatalog::standard();
        let deployment = catalog.by_category(HfComponentCategory::Deployment);
        // Should have: inference-providers, inference-endpoints, tgi, tei, aws-dlcs, azure, gcp
        assert!(deployment.len() >= 7);
    }

    #[test]
    fn test_HF_QUERY_001_063_library_category_components() {
        let catalog = HfCatalog::standard();
        let library = catalog.by_category(HfComponentCategory::Library);
        // Should have: transformers, diffusers, datasets, transformers-js, tokenizers, evaluate, timm, sentence-transformers, kernels, safetensors
        assert!(library.len() >= 10);
    }

    #[test]
    fn test_HF_QUERY_001_064_training_category_components() {
        let catalog = HfCatalog::standard();
        let training = catalog.by_category(HfComponentCategory::Training);
        // Should have: peft, accelerate, optimum, aws-trainium, tpu, trl, bitsandbytes, lighteval, trainer, autotrain
        assert!(training.len() >= 10);
    }

    #[test]
    fn test_HF_QUERY_001_065_collaboration_category_components() {
        let catalog = HfCatalog::standard();
        let collab = catalog.by_category(HfComponentCategory::Collaboration);
        // Should have: gradio, trackio, smolagents, lerobot, chat-ui, leaderboards, argilla, distilabel
        assert!(collab.len() >= 8);
    }

    #[test]
    fn test_HF_QUERY_001_066_community_category_components() {
        let catalog = HfCatalog::standard();
        let community = catalog.by_category(HfComponentCategory::Community);
        // Should have: blog, learn, discord, forum
        assert!(community.len() >= 4);
    }
}
