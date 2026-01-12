//! HuggingFace Ecosystem Catalog Demo
//!
//! Demonstrates the HF catalog for querying 50+ ecosystem components,
//! course alignment, and dependency graphs.
//!
//! Run with: cargo run --example hf_catalog_demo

use batuta::hf::catalog::{AssetType, HfCatalog, HfComponentCategory};

fn main() {
    println!("🤗 HuggingFace Ecosystem Catalog Demo");
    println!("Query 50+ components with course alignment and dependency graphs\n");

    // Load the standard catalog
    let catalog = HfCatalog::standard();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. CATALOG OVERVIEW");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("📦 Total components: {}\n", catalog.len());

    println!("📂 Components by category:");
    for category in HfComponentCategory::all() {
        let count = catalog.by_category(*category).len();
        println!("  {:30} {:>3} components", category.display_name(), count);
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. ADVANCED FINE-TUNING COMPONENTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Query training components
    let training = catalog.by_category(HfComponentCategory::Training);
    println!(
        "🎓 Training & Optimization ({} components):\n",
        training.len()
    );

    for comp in &training {
        println!("  {} - {}", comp.id, comp.name);
        println!("    {}", comp.description);
        println!("    Tags: {}", comp.tags.join(", "));
        if !comp.dependencies.is_empty() {
            println!("    Deps: {}", comp.dependencies.join(", "));
        }
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. SEARCH BY TAG");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Search for RLHF-related components
    let rlhf_results = catalog.search("rlhf");
    println!("🔍 Search results for 'rlhf':");
    for comp in &rlhf_results {
        println!("  - {} ({})", comp.name, comp.category);
    }
    println!();

    // Search for quantization
    let quant_results = catalog.search("quantization");
    println!("🔍 Search results for 'quantization':");
    for comp in &quant_results {
        println!("  - {} ({})", comp.name, comp.category);
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. COURSE ALIGNMENT (Coursera Specialization)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let course_names = [
        "Foundations of HuggingFace",
        "Fine-Tuning and Datasets",
        "RAG and Retrieval",
        "Advanced Training (RLHF, DPO, PPO)",
        "Production Deployment",
    ];

    for (i, name) in course_names.iter().enumerate() {
        let course_num = (i + 1) as u8;
        let components = catalog.by_course(course_num);
        println!(
            "📚 Course {}: {} ({} components)",
            course_num,
            name,
            components.len()
        );
        for comp in &components {
            // Find which weeks this component appears in
            let weeks: Vec<u8> = comp
                .courses
                .iter()
                .filter(|ca| ca.course == course_num)
                .map(|ca| ca.week)
                .collect();
            println!("    - {} (Week {:?})", comp.name, weeks);
        }
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. DEPENDENCY GRAPH");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Show PEFT dependencies
    println!("📊 PEFT dependency tree:");
    if catalog.get("peft").is_some() {
        println!("  peft");
        for dep in catalog.deps("peft") {
            println!("  └── {} ({})", dep.id, dep.category);
        }
    }
    println!();

    // Show what depends on transformers
    println!("📊 Components that depend on 'transformers':");
    let rdeps = catalog.rdeps("transformers");
    for comp in &rdeps {
        println!("  - {}", comp.id);
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. ASSET TYPES (Content Planning)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!(
        "🎬 Components with Video content: {}",
        catalog.by_asset_type(AssetType::Video).len()
    );
    println!(
        "🔬 Components with Lab content: {}",
        catalog.by_asset_type(AssetType::Lab).len()
    );
    println!(
        "📖 Components with Reading content: {}",
        catalog.by_asset_type(AssetType::Reading).len()
    );
    println!(
        "📝 Components with Quiz content: {}",
        catalog.by_asset_type(AssetType::Quiz).len()
    );
    println!(
        "💬 Components with Discussion content: {}",
        catalog.by_asset_type(AssetType::Discussion).len()
    );
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("7. DOCUMENTATION LINKS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let key_components = ["transformers", "peft", "trl", "datasets", "tgi"];
    for id in key_components {
        if let Some(url) = catalog.docs_url(id) {
            println!("📄 {}: {}", id, url);
        }
    }
    println!();

    println!("✅ Demo complete! Use `batuta hf catalog` for CLI access.");
}
