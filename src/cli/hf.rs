//! HuggingFace command implementations
//!
//! This module contains all HuggingFace-related CLI commands extracted from main.rs.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use crate::hf;
use std::path::{Path, PathBuf};

/// HuggingFace subcommand
#[derive(Debug, Clone, clap::Subcommand)]
pub enum HfCommand {
    /// Query 50+ HuggingFace ecosystem components
    Catalog {
        /// Component ID to get details for
        #[arg(long)]
        component: Option<String>,

        /// Filter by category (hub, deployment, library, training, collaboration, community)
        #[arg(long)]
        category: Option<String>,

        /// Filter by tag
        #[arg(long)]
        tag: Option<String>,

        /// List all available components
        #[arg(long)]
        list: bool,

        /// List all categories with component counts
        #[arg(long)]
        categories: bool,

        /// List all available tags
        #[arg(long)]
        tags: bool,

        /// Output format (table, json)
        #[arg(long, default_value = "table")]
        format: String,
    },

    /// Query by Coursera course alignment (5 courses, 15 weeks, 60 hours)
    Course {
        /// Course number (1-5)
        #[arg(long)]
        course: Option<u8>,

        /// Week number within course
        #[arg(long)]
        week: Option<u8>,

        /// List all courses
        #[arg(long)]
        list: bool,

        /// Show course-to-component mapping
        #[arg(long)]
        mapping: bool,

        /// Output format (table, json)
        #[arg(long, default_value = "table")]
        format: String,
    },

    /// Display HuggingFace ecosystem tree
    Tree {
        /// Show PAIML-HuggingFace integration map
        #[arg(long)]
        integration: bool,

        /// Output format (ascii, json)
        #[arg(long, default_value = "ascii")]
        format: String,
    },

    /// Search HuggingFace Hub (models, datasets, spaces)
    Search {
        /// Asset type to search
        #[arg(value_enum)]
        asset_type: HfAssetType,

        /// Search query
        query: String,

        /// Filter by task (for models)
        #[arg(long)]
        task: Option<String>,

        /// Limit results
        #[arg(long, default_value = "10")]
        limit: usize,
    },

    /// Get info about a HuggingFace asset
    Info {
        /// Asset type
        #[arg(value_enum)]
        asset_type: HfAssetType,

        /// Repository ID (e.g., "meta-llama/Llama-2-7b-hf")
        repo_id: String,
    },

    /// Pull model/dataset from HuggingFace Hub
    Pull {
        /// Asset type
        #[arg(value_enum)]
        asset_type: HfAssetType,

        /// Repository ID
        repo_id: String,

        /// Output directory
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Model quantization (for models)
        #[arg(long)]
        quantization: Option<String>,
    },

    /// Push model/dataset to HuggingFace Hub
    Push {
        /// Asset type
        #[arg(value_enum)]
        asset_type: HfAssetType,

        /// Local path to push
        path: PathBuf,

        /// Target repository ID
        #[arg(long)]
        repo: String,

        /// Commit message
        #[arg(long, default_value = "Update from batuta")]
        message: String,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum HfAssetType {
    /// Model repository
    Model,
    /// Dataset repository
    Dataset,
    /// Space (app) repository
    Space,
}

/// Main HF command dispatcher
pub fn cmd_hf(command: HfCommand) -> anyhow::Result<()> {
    match command {
        HfCommand::Catalog {
            component,
            category,
            tag,
            list,
            categories,
            tags,
            format,
        } => {
            cmd_hf_catalog(component, category, tag, list, categories, tags, &format)?;
        }
        HfCommand::Course {
            course,
            week,
            list,
            mapping,
            format,
        } => {
            cmd_hf_course(course, week, list, mapping, &format)?;
        }
        HfCommand::Tree {
            integration,
            format,
        } => {
            cmd_hf_tree(integration, &format)?;
        }
        HfCommand::Search {
            asset_type,
            query,
            task,
            limit,
        } => {
            cmd_hf_search(asset_type, &query, task.as_deref(), limit)?;
        }
        HfCommand::Info {
            asset_type,
            repo_id,
        } => {
            cmd_hf_info(asset_type, &repo_id)?;
        }
        HfCommand::Pull {
            asset_type,
            repo_id,
            output,
            quantization,
        } => {
            cmd_hf_pull(asset_type, &repo_id, output, quantization)?;
        }
        HfCommand::Push {
            asset_type,
            path,
            repo,
            message,
        } => {
            cmd_hf_push(asset_type, &path, &repo, &message)?;
        }
    }
    Ok(())
}

/// Options for the HF catalog command
struct CatalogOptions<'a> {
    component: Option<String>,
    category: Option<String>,
    tag: Option<String>,
    list: bool,
    categories: bool,
    tags: bool,
    format: &'a str,
}

fn cmd_hf_catalog(
    component: Option<String>,
    category: Option<String>,
    tag: Option<String>,
    list: bool,
    categories: bool,
    tags: bool,
    format: &str,
) -> anyhow::Result<()> {
    use hf::catalog::HfCatalog;

    let catalog = HfCatalog::standard();
    let opts = CatalogOptions {
        component,
        category,
        tag,
        list,
        categories,
        tags,
        format,
    };

    if opts.categories {
        return catalog_show_categories(&catalog);
    }
    if opts.tags {
        return catalog_show_tags(&catalog);
    }
    if let Some(ref comp_id) = opts.component {
        return catalog_show_component(&catalog, comp_id, opts.format);
    }

    catalog_list_or_filter(&catalog, &opts)
}

fn catalog_show_categories(catalog: &hf::catalog::HfCatalog) -> anyhow::Result<()> {
    use hf::catalog::HfComponentCategory;

    println!(
        "{}",
        "📂 HuggingFace Ecosystem Categories".bright_cyan().bold()
    );
    println!("{}", "═".repeat(60).dimmed());
    println!();

    for cat in HfComponentCategory::all() {
        let count = catalog.by_category(*cat).len();
        println!("  {} ({} components)", cat.display_name().yellow(), count);
    }
    println!();
    println!("Total: {} components", catalog.len());
    Ok(())
}

fn catalog_show_tags(catalog: &hf::catalog::HfCatalog) -> anyhow::Result<()> {
    println!("{}", "🏷️  HuggingFace Component Tags".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    let mut all_tags: Vec<String> = catalog.all().flat_map(|c| c.tags.iter().cloned()).collect();
    all_tags.sort();
    all_tags.dedup();

    for tag in &all_tags {
        let count = catalog.by_tag(tag).len();
        println!("  {} ({})", tag.cyan(), count);
    }
    println!();
    println!("Total: {} unique tags", all_tags.len());
    Ok(())
}

fn catalog_show_component(
    catalog: &hf::catalog::HfCatalog,
    comp_id: &str,
    format: &str,
) -> anyhow::Result<()> {
    let comp = catalog
        .get(comp_id)
        .ok_or_else(|| anyhow::anyhow!("Component '{}' not found in catalog", comp_id))?;

    if format == "json" {
        println!("{}", serde_json::to_string_pretty(comp)?);
        return Ok(());
    }

    println!("{}", format!("📦 {}", comp.name).bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();
    println!("ID:          {}", comp.id.yellow());
    println!("Category:    {}", comp.category.display_name());
    println!("Description: {}", comp.description);
    println!("Docs:        {}", comp.docs_url.cyan());
    if let Some(ref repo) = comp.repo_url {
        println!("Repository:  {}", repo);
    }
    if let Some(ref pypi) = comp.pypi_name {
        println!("PyPI:        {}", pypi);
    }
    if let Some(ref npm) = comp.npm_name {
        println!("npm:         {}", npm);
    }
    if !comp.tags.is_empty() {
        println!("Tags:        {}", comp.tags.join(", ").dimmed());
    }
    if !comp.dependencies.is_empty() {
        println!("Dependencies: {}", comp.dependencies.join(", "));
    }
    if !comp.courses.is_empty() {
        println!();
        println!("Course Alignments:");
        for ca in &comp.courses {
            println!(
                "  Course {}, Week {}: {}",
                ca.course,
                ca.week,
                ca.lessons.join(", ")
            );
        }
    }
    Ok(())
}

fn catalog_parse_category(cat_name: &str) -> anyhow::Result<hf::catalog::HfComponentCategory> {
    use hf::catalog::HfComponentCategory;
    match cat_name.to_lowercase().as_str() {
        "hub" => Ok(HfComponentCategory::Hub),
        "deployment" => Ok(HfComponentCategory::Deployment),
        "library" => Ok(HfComponentCategory::Library),
        "training" => Ok(HfComponentCategory::Training),
        "collaboration" => Ok(HfComponentCategory::Collaboration),
        "community" => Ok(HfComponentCategory::Community),
        _ => anyhow::bail!(
            "Unknown category: {}. Valid: hub, deployment, library, training, collaboration, community",
            cat_name
        ),
    }
}

fn catalog_show_summary(catalog: &hf::catalog::HfCatalog) {
    use hf::catalog::HfComponentCategory;

    println!(
        "{}",
        "🤗 HuggingFace Ecosystem Catalog".bright_cyan().bold()
    );
    println!("{}", "═".repeat(60).dimmed());
    println!();
    println!("Total components: {}", catalog.len());
    println!();
    println!("Categories:");
    for cat in HfComponentCategory::all() {
        let count = catalog.by_category(*cat).len();
        println!("  {:30} {}", cat.display_name(), count);
    }
    println!();
    println!("Use --list to see all components");
    println!("Use --category <name> to filter by category");
    println!("Use --tag <name> to filter by tag");
    println!("Use --component <id> to get component details");
}

fn catalog_print_components(
    components: &[&hf::catalog::CatalogComponent],
    format: &str,
) -> anyhow::Result<()> {
    if format == "json" {
        println!("{}", serde_json::to_string_pretty(&components)?);
    } else {
        println!("{}", "📦 HuggingFace Components".bright_cyan().bold());
        println!("{}", "═".repeat(60).dimmed());
        println!();
        for comp in components {
            println!(
                "  {:25} {:30} {}",
                comp.id.yellow(),
                comp.name,
                comp.category.display_name().dimmed()
            );
        }
        println!();
        println!("Total: {} components", components.len());
    }
    Ok(())
}

fn catalog_list_or_filter(
    catalog: &hf::catalog::HfCatalog,
    opts: &CatalogOptions<'_>,
) -> anyhow::Result<()> {
    let components: Vec<&hf::catalog::CatalogComponent> = if let Some(ref cat_name) = opts.category
    {
        let cat = catalog_parse_category(cat_name)?;
        catalog.by_category(cat)
    } else if let Some(ref tag_name) = opts.tag {
        catalog.by_tag(tag_name)
    } else if opts.list {
        catalog.all().collect()
    } else {
        catalog_show_summary(catalog);
        return Ok(());
    };

    catalog_print_components(&components, opts.format)
}

/// Course titles for the Pragmatic AI Labs Coursera specialization
const COURSE_TITLES: [(&str, &str); 5] = [
    ("Course 1", "Foundations of HuggingFace"),
    ("Course 2", "Fine-Tuning and Datasets"),
    ("Course 3", "RAG and Retrieval"),
    ("Course 4", "Advanced Training (RLHF, DPO, PPO)"),
    ("Course 5", "Production Deployment"),
];

fn cmd_hf_course(
    course: Option<u8>,
    week: Option<u8>,
    list: bool,
    mapping: bool,
    format: &str,
) -> anyhow::Result<()> {
    use hf::catalog::HfCatalog;

    let catalog = HfCatalog::standard();

    if list {
        return course_show_list(&catalog);
    }
    if mapping {
        return course_show_mapping(&catalog, format);
    }
    if let Some(course_num) = course {
        return course_show_filtered(&catalog, course_num, week, format);
    }

    course_show_help();
    Ok(())
}

fn course_show_list(catalog: &hf::catalog::HfCatalog) -> anyhow::Result<()> {
    println!(
        "{}",
        "📚 Pragmatic AI Labs HuggingFace Specialization"
            .bright_cyan()
            .bold()
    );
    println!("{}", "═".repeat(60).dimmed());
    println!();
    println!("5 Courses | 15 Weeks | 60 Hours");
    println!();
    for (i, (label, title)) in COURSE_TITLES.iter().enumerate() {
        let course_num = (i + 1) as u8;
        let components = catalog.by_course(course_num);
        println!(
            "  {}: {} ({} components)",
            label,
            title.yellow(),
            components.len()
        );
    }
    Ok(())
}

fn course_show_mapping(catalog: &hf::catalog::HfCatalog, format: &str) -> anyhow::Result<()> {
    if format == "json" {
        return course_show_mapping_json(catalog);
    }

    println!("{}", "📊 Course-to-Component Mapping".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();
    for (i, (label, title)) in COURSE_TITLES.iter().enumerate() {
        let course_num = (i + 1) as u8;
        let components = catalog.by_course(course_num);
        println!("{}: {}", label.yellow(), title);
        for comp in components {
            let weeks: Vec<_> = comp
                .courses
                .iter()
                .filter(|ca| ca.course == course_num)
                .map(|ca| format!("Week {}", ca.week))
                .collect();
            println!("  {} ({})", comp.id.cyan(), weeks.join(", "));
        }
        println!();
    }
    Ok(())
}

fn course_show_mapping_json(catalog: &hf::catalog::HfCatalog) -> anyhow::Result<()> {
    let mut course_map: std::collections::HashMap<u8, Vec<&str>> = std::collections::HashMap::new();
    for i in 1..=5 {
        let comps: Vec<_> = catalog.by_course(i).iter().map(|c| c.id.as_str()).collect();
        course_map.insert(i, comps);
    }
    println!("{}", serde_json::to_string_pretty(&course_map)?);
    Ok(())
}

fn course_show_filtered(
    catalog: &hf::catalog::HfCatalog,
    course_num: u8,
    week: Option<u8>,
    format: &str,
) -> anyhow::Result<()> {
    if !(1..=5).contains(&course_num) {
        anyhow::bail!("Course must be 1-5, got {}", course_num);
    }

    let (label, title) = COURSE_TITLES[(course_num - 1) as usize];
    let components = if let Some(week_num) = week {
        catalog.by_course_week(course_num, week_num)
    } else {
        catalog.by_course(course_num)
    };

    if format == "json" {
        println!("{}", serde_json::to_string_pretty(&components)?);
        return Ok(());
    }

    println!(
        "{}",
        format!("📚 {} - {}", label, title).bright_cyan().bold()
    );
    if let Some(w) = week {
        println!("Week {}", w);
    }
    println!("{}", "═".repeat(60).dimmed());
    println!();
    for comp in &components {
        let weeks: Vec<_> = comp
            .courses
            .iter()
            .filter(|ca| ca.course == course_num)
            .map(|ca| format!("Week {}", ca.week))
            .collect();
        println!("  {:25} {}", comp.id.yellow(), weeks.join(", ").dimmed());
    }
    println!();
    println!("Total: {} components", components.len());
    Ok(())
}

fn course_show_help() {
    println!("{}", "📚 HuggingFace Course Query".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();
    println!("Pragmatic AI Labs HuggingFace Specialization");
    println!("5 Courses | 15 Weeks | 60 Hours");
    println!();
    println!("Use --list to see all courses");
    println!("Use --mapping to see course-to-component mapping");
    println!("Use --course <num> to see components for a course");
    println!("Use --course <num> --week <num> to filter by week");
}

fn cmd_hf_tree(integration: bool, format: &str) -> anyhow::Result<()> {
    use hf::tree::{
        build_hf_tree, build_integration_tree, format_hf_tree_ascii, format_hf_tree_json,
        format_integration_tree_ascii, format_integration_tree_json,
    };

    let output = if integration {
        let tree = build_integration_tree();
        match format {
            "json" => format_integration_tree_json(&tree)?,
            _ => format_integration_tree_ascii(&tree),
        }
    } else {
        let tree = build_hf_tree();
        match format {
            "json" => format_hf_tree_json(&tree)?,
            _ => format_hf_tree_ascii(&tree),
        }
    };

    println!("{}", output);
    Ok(())
}

fn cmd_hf_search(
    asset_type: HfAssetType,
    query: &str,
    task: Option<&str>,
    limit: usize,
) -> anyhow::Result<()> {
    println!("{}", "🔍 HuggingFace Hub Search".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    let type_str = match asset_type {
        HfAssetType::Model => "models",
        HfAssetType::Dataset => "datasets",
        HfAssetType::Space => "spaces",
    };

    println!("Searching {} for: {}", type_str.cyan(), query.yellow());
    if let Some(t) = task {
        println!("Task filter: {}", t.cyan());
    }
    println!("Limit: {}", limit);
    println!();

    // Hub API search integration planned for v0.2
    // See roadmap item HUB-API for implementation plan
    println!(
        "{}",
        "⚠️  Hub API integration not yet implemented.".yellow()
    );
    println!("Will query: https://huggingface.co/api/{}", type_str);

    Ok(())
}

fn cmd_hf_info(asset_type: HfAssetType, repo_id: &str) -> anyhow::Result<()> {
    println!("{}", "📋 HuggingFace Asset Info".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    let type_str = match asset_type {
        HfAssetType::Model => "model",
        HfAssetType::Dataset => "dataset",
        HfAssetType::Space => "space",
    };

    println!("Type: {}", type_str.cyan());
    println!("Repository: {}", repo_id.yellow());
    println!();

    // Hub API info integration planned for v0.2
    // See roadmap item HUB-API for implementation plan
    println!(
        "{}",
        "⚠️  Hub API integration not yet implemented.".yellow()
    );
    println!(
        "Will fetch: https://huggingface.co/api/{}s/{}",
        type_str, repo_id
    );

    Ok(())
}

fn cmd_hf_pull(
    asset_type: HfAssetType,
    repo_id: &str,
    output: Option<PathBuf>,
    quantization: Option<String>,
) -> anyhow::Result<()> {
    println!("{}", "⬇️  HuggingFace Pull".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    let type_str = match asset_type {
        HfAssetType::Model => "model",
        HfAssetType::Dataset => "dataset",
        HfAssetType::Space => "space",
    };

    println!("Type: {}", type_str.cyan());
    println!("Repository: {}", repo_id.yellow());
    if let Some(ref out) = output {
        println!("Output: {}", out.display());
    }
    if let Some(ref q) = quantization {
        println!("Quantization: {}", q.cyan());
    }
    println!();

    // Hub API download integration planned for v0.2
    // See roadmap item HUB-API for implementation plan
    println!(
        "{}",
        "⚠️  Hub API integration not yet implemented.".yellow()
    );
    println!("Will download from: https://huggingface.co/{}", repo_id);

    Ok(())
}

fn cmd_hf_push(
    asset_type: HfAssetType,
    path: &Path,
    repo: &str,
    message: &str,
) -> anyhow::Result<()> {
    println!("{}", "⬆️  HuggingFace Push".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    let type_str = match asset_type {
        HfAssetType::Model => "model",
        HfAssetType::Dataset => "dataset",
        HfAssetType::Space => "space",
    };

    println!("Type: {}", type_str.cyan());
    println!("Local path: {}", path.display());
    println!("Target repo: {}", repo.yellow());
    println!("Message: {}", message);
    println!();

    // Hub API upload integration planned for v0.2
    // See roadmap item HUB-API for implementation plan
    println!(
        "{}",
        "⚠️  Hub API integration not yet implemented.".yellow()
    );
    println!("Will push to: https://huggingface.co/{}", repo);

    Ok(())
}
