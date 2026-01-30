/// Local Workspace Oracle demonstration
/// Discovers and analyzes PAIML projects in ~/src with development state awareness
use batuta::oracle::local_workspace::{DevState, LocalWorkspaceOracle};

fn main() -> anyhow::Result<()> {
    println!("🏠 Local Workspace Oracle Demo");
    println!("Discover PAIML projects and their development state\n");

    // Initialize the local workspace oracle
    let mut oracle = LocalWorkspaceOracle::new()?;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. DISCOVERING LOCAL PROJECTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Discover all PAIML projects
    let projects = oracle.discover_projects()?;
    println!("📁 Found {} PAIML projects in ~/src:\n", projects.len());

    for project in projects.values() {
        let state_icon = match project.dev_state {
            DevState::Clean => "✅",
            DevState::Dirty => "🔧",
            DevState::Unpushed => "📤",
        };

        println!(
            "  {} {} v{} ({:?})",
            state_icon, project.name, project.local_version, project.dev_state
        );

        // Show git status if there are changes
        let status = &project.git_status;
        if status.modified_count > 0 || status.unpushed_commits > 0 {
            println!(
                "     └─ {} modified, {} ahead of remote",
                status.modified_count, status.unpushed_commits
            );
        }
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. DEVELOPMENT STATE AWARENESS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Understanding DevState:");
    println!("  ✅ Clean    - No uncommitted changes, safe to use local version");
    println!("  🔧 Dirty    - Active development, use crates.io version for deps");
    println!("  📤 Unpushed - Clean but has unpushed commits\n");

    // Count by state
    let projects = oracle.projects();
    let clean = projects
        .values()
        .filter(|p| p.dev_state == DevState::Clean)
        .count();
    let dirty = projects
        .values()
        .filter(|p| p.dev_state == DevState::Dirty)
        .count();
    let unpushed = projects
        .values()
        .filter(|p| p.dev_state == DevState::Unpushed)
        .count();

    println!("📊 Project States:");
    println!("  ✅ Clean:    {}", clean);
    println!("  🔧 Dirty:    {}", dirty);
    println!("  📤 Unpushed: {}", unpushed);
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. DIRTY PROJECTS (Active Development)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("🔧 Projects with uncommitted changes:\n");
    let dirty_projects: Vec<_> = projects
        .values()
        .filter(|p| p.dev_state == DevState::Dirty)
        .collect();

    if dirty_projects.is_empty() {
        println!("  (none - all projects are clean!)");
    } else {
        for project in &dirty_projects {
            println!("  🔧 {}", project.name);
            let status = &project.git_status;
            println!("     {} modified files", status.modified_count);
            println!("     Local:     v{}", project.local_version);
            if let Some(crates_ver) = &project.published_version {
                println!(
                    "     Crates.io: v{} (stable - use this for deps)",
                    crates_ver
                );
            }
            println!();
        }
    }

    println!("💡 Key Insight: Dirty projects don't block the stack!");
    println!("   The crates.io version is stable and should be used for dependencies.\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. VERSION DRIFT DETECTION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("🔍 Comparing local versions vs crates.io:\n");

    for project in projects.values() {
        if let Some(published) = &project.published_version {
            let (drift_icon, drift_desc) = if project.local_version > *published {
                ("📈", "LocalAhead")
            } else if project.local_version < *published {
                ("📉", "LocalBehind")
            } else {
                ("✓", "InSync")
            };
            if project.local_version != *published {
                println!("  {} {} ({})", drift_icon, project.name, drift_desc);
                println!(
                    "     Local: v{}  →  Crates.io: v{}",
                    project.local_version, published
                );
                println!();
            }
        } else {
            println!("  🆕 {} (NotPublished)", project.name);
            println!("     Local: v{}", project.local_version);
            println!();
        }
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. PUBLISH ORDER (Topological Sort)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let publish_order = oracle.suggest_publish_order();

    println!("📦 Safe publish order (respects dependencies):\n");
    for (i, step) in publish_order.order.iter().enumerate() {
        let ready_icon = if step.needs_publish { "📤" } else { "✅" };
        println!(
            "  {}. {} {} v{}",
            i + 1,
            ready_icon,
            step.name,
            step.version,
        );
        if !step.blocked_by.is_empty() {
            for blocker in &step.blocked_by {
                println!("     ⚠️  Blocked by: {}", blocker);
            }
        }
    }

    if !publish_order.cycles.is_empty() {
        println!("\n⚠️  Detected dependency cycles:");
        for cycle in &publish_order.cycles {
            println!("     {}", cycle.join(" → "));
        }
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. WORKSPACE SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let summary = oracle.summary();
    println!("📊 Workspace Overview:");
    println!("  Total PAIML projects:   {}", summary.total_projects);
    println!(
        "  With uncommitted:       {}",
        summary.projects_with_changes
    );
    println!(
        "  With unpushed commits:  {}",
        summary.projects_with_unpushed
    );
    println!("  Workspace projects:     {}", summary.workspace_count);
    println!();

    println!("✅ Local Workspace Oracle ready!");
    println!("   Run: batuta oracle --local");
    println!("   Run: batuta oracle --dirty");
    println!("   Run: batuta oracle --publish-order");

    Ok(())
}
