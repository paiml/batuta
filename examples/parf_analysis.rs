//! PARF (Pattern and Reference Finder) Analysis Example (BATUTA-012)
//!
//! Demonstrates cross-codebase pattern analysis and reference finding
//! for understanding code dependencies, usage patterns, and migration planning.
//!
//! Run with: cargo run --example parf_analysis

use batuta::parf::{CodePattern, ParfAnalyzer, SymbolKind};
use std::path::Path;

fn main() {
    println!("🔍 PARF Analysis Demo (BATUTA-012)");
    println!("===================================\n");

    // Analyze the Batuta codebase itself
    let mut analyzer = ParfAnalyzer::new();

    println!("📁 Indexing codebase...");
    let codebase_path = Path::new("src");

    match analyzer.index_codebase(codebase_path) {
        Ok(()) => println!("   ✅ Indexing complete\n"),
        Err(e) => {
            eprintln!("   ❌ Indexing failed: {}", e);
            return;
        }
    }

    // Generate overall report
    println!("📊 Analysis Report");
    println!("------------------\n");
    let report = analyzer.generate_report();
    println!("{}", report);

    // Find references to specific symbols
    println!("\n🔎 Symbol Reference Analysis");
    println!("----------------------------\n");

    let symbols_to_search = vec![
        ("BackendSelector", SymbolKind::Class),
        ("select_with_moe", SymbolKind::Function),
        ("NumPyConverter", SymbolKind::Class),
    ];

    for (symbol, kind) in symbols_to_search {
        let refs = analyzer.find_references(symbol, kind);
        println!("Symbol: {} ({:?})", symbol, kind);
        println!("  References found: {}", refs.len());

        if !refs.is_empty() {
            println!("  Sample references:");
            for (i, r) in refs.iter().take(3).enumerate() {
                println!("    {}. {}:{}", i + 1, r.file.display(), r.line);
            }
            if refs.len() > 3 {
                println!("    ... and {} more", refs.len() - 3);
            }
        }
        println!();
    }

    // Pattern detection
    println!("🎯 Code Pattern Detection");
    println!("-------------------------\n");

    let patterns = analyzer.detect_patterns();

    // Group patterns by type
    let mut tech_debt = Vec::new();
    let mut error_handling = Vec::new();
    let mut resource_mgmt = Vec::new();
    let mut deprecated = Vec::new();

    for pattern in patterns {
        match pattern {
            CodePattern::TechDebt { .. } => tech_debt.push(pattern),
            CodePattern::ErrorHandling { .. } => error_handling.push(pattern),
            CodePattern::ResourceManagement { .. } => resource_mgmt.push(pattern),
            CodePattern::DeprecatedApi { .. } => deprecated.push(pattern),
            _ => {}
        }
    }

    println!("Technical Debt (TODO/FIXME): {}", tech_debt.len());
    if !tech_debt.is_empty() {
        for (i, pattern) in tech_debt.iter().take(5).enumerate() {
            if let CodePattern::TechDebt { message, file, line } = pattern {
                println!("  {}. {}:{} - {}", i + 1, file.display(), line, message);
            }
        }
        if tech_debt.len() > 5 {
            println!("  ... and {} more", tech_debt.len() - 5);
        }
    }
    println!();

    println!("Error Handling Issues: {}", error_handling.len());
    if !error_handling.is_empty() {
        for (i, pattern) in error_handling.iter().take(5).enumerate() {
            if let CodePattern::ErrorHandling { pattern: p, file, line } = pattern {
                println!("  {}. {}:{} - {}", i + 1, file.display(), line, p);
            }
        }
        if error_handling.len() > 5 {
            println!("  ... and {} more", error_handling.len() - 5);
        }
    }
    println!();

    println!("Resource Management: {}", resource_mgmt.len());
    if !resource_mgmt.is_empty() {
        for (i, pattern) in resource_mgmt.iter().take(5).enumerate() {
            if let CodePattern::ResourceManagement { resource_type, file, line } = pattern {
                println!("  {}. {}:{} - {} resource", i + 1, file.display(), line, resource_type);
            }
        }
        if resource_mgmt.len() > 5 {
            println!("  ... and {} more", resource_mgmt.len() - 5);
        }
    }
    println!();

    println!("Deprecated APIs: {}", deprecated.len());
    if !deprecated.is_empty() {
        for (i, pattern) in deprecated.iter().take(5).enumerate() {
            if let CodePattern::DeprecatedApi { api, file, line } = pattern {
                println!("  {}. {}:{} - {}", i + 1, file.display(), line, api);
            }
        }
        if deprecated.len() > 5 {
            println!("  ... and {} more", deprecated.len() - 5);
        }
    }
    println!();

    // Dependency analysis
    println!("📦 Dependency Analysis");
    println!("---------------------\n");

    let dependencies = analyzer.analyze_dependencies();
    println!("Total dependencies: {}", dependencies.len());

    if !dependencies.is_empty() {
        println!("Sample dependencies:");
        for (i, dep) in dependencies.iter().take(10).enumerate() {
            println!("  {}. {} → {} ({:?})",
                i + 1,
                dep.from.display(),
                dep.to.display(),
                dep.kind
            );
        }
        if dependencies.len() > 10 {
            println!("  ... and {} more", dependencies.len() - 10);
        }
    }
    println!();

    // Dead code analysis
    println!("💀 Dead Code Analysis");
    println!("--------------------\n");

    let dead_code = analyzer.find_dead_code();
    println!("Potentially unused symbols: {}", dead_code.len());

    if !dead_code.is_empty() {
        println!("\nTop candidates for removal:");
        for (i, dc) in dead_code.iter().take(10).enumerate() {
            println!("  {}. {} ({:?}) in {}:{}",
                i + 1,
                dc.symbol,
                dc.kind,
                dc.file.display(),
                dc.line
            );
            println!("     Reason: {}", dc.reason);
        }
        if dead_code.len() > 10 {
            println!("  ... and {} more", dead_code.len() - 10);
        }
    }
    println!();

    // Use case summary
    println!("💡 PARF Use Cases");
    println!("----------------\n");
    println!("  1. Code Understanding:");
    println!("     • Find all usages of a function or type");
    println!("     • Understand call graphs and dependencies");
    println!("     • Navigate unfamiliar codebases");
    println!();
    println!("  2. Refactoring:");
    println!("     • Identify dead code for safe removal");
    println!("     • Find all references before renaming");
    println!("     • Detect duplicate patterns for consolidation");
    println!();
    println!("  3. Migration Planning:");
    println!("     • Map dependencies for phased migration");
    println!("     • Identify technical debt to address");
    println!("     • Find deprecated APIs for replacement");
    println!();
    println!("  4. Code Quality:");
    println!("     • Detect error handling anti-patterns");
    println!("     • Find resource leaks (unclosed files)");
    println!("     • Track TODO/FIXME items");

    println!("\n✅ Analysis Complete!");
    println!("   PARF provides actionable insights for:");
    println!("   - Migration planning");
    println!("   - Refactoring safety");
    println!("   - Code quality improvement");
    println!("   - Technical debt reduction");
}
