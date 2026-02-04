//! Tests for HuggingFace Ecosystem Catalog operations
//!
//! Extreme TDD with 95%+ Coverage

use super::*;

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
