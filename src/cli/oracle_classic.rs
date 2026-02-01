//! Classic Oracle Commands
//!
//! This module contains the classic oracle query interface, component recommendations,
//! and display functions.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use crate::oracle;

use super::oracle::OracleOutputFormat;

/// Options for the oracle command.
///
/// Groups all oracle command parameters to avoid too-many-arguments warnings.
pub struct OracleOptions {
    pub query: Option<String>,
    pub recommend: bool,
    pub problem: Option<String>,
    pub data_size: Option<String>,
    pub integrate: Option<String>,
    pub capabilities: Option<String>,
    pub list: bool,
    pub show: Option<String>,
    pub interactive: bool,
    pub format: OracleOutputFormat,
}

fn oracle_show_help() {
    println!("{}", "🔮 Batuta Oracle Mode".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();
    println!(
        "{}",
        "Query the Sovereign AI Stack for recommendations".dimmed()
    );
    println!();
    println!("{}", "Knowledge Graph:".bright_yellow());
    println!(
        "  {} {}",
        "batuta oracle".cyan(),
        "\"How do I train a random forest?\"".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --recommend --problem".cyan(),
        "\"image classification\"".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --capabilities".cyan(),
        "aprender".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --integrate".cyan(),
        "\"aprender,realizar\"".dimmed()
    );
    println!("  {} {}", "batuta oracle --list".cyan(), "".dimmed());
    println!("  {} {}", "batuta oracle --interactive".cyan(), "".dimmed());
    println!();
    println!("{}", "Cookbook (Practical Recipes):".bright_yellow());
    println!(
        "  {} {}",
        "batuta oracle --cookbook".cyan(),
        "# List all recipes".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --recipe".cyan(),
        "wasm-zero-js       # Show recipe".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --recipes-by-tag".cyan(),
        "ml        # By tag".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --search-recipes".cyan(),
        "\"gpu\"   # Search".dimmed()
    );
    println!();
}

fn oracle_handle_recommend(
    recommender: &oracle::Recommender,
    problem: Option<String>,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let query_text = problem.unwrap_or_else(|| "general ML task".into());
    let response = recommender.query(&query_text);
    display_oracle_response(&response, format)
}

fn oracle_handle_query(
    recommender: &oracle::Recommender,
    query_text: &str,
    data_size: Option<String>,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use oracle::OracleQuery;

    let parsed_size = data_size.and_then(|s| parse_data_size(&s));
    let mut oracle_query = OracleQuery::new(query_text);
    if let Some(size) = parsed_size {
        oracle_query = oracle_query.with_data_size(size);
    }
    let response = recommender.query_structured(&oracle_query);
    display_oracle_response(&response, format)
}

pub fn cmd_oracle(opts: OracleOptions) -> anyhow::Result<()> {
    let OracleOptions {
        query,
        recommend,
        problem,
        data_size,
        integrate,
        capabilities,
        list,
        show,
        interactive,
        format,
    } = opts;
    use oracle::Recommender;

    let recommender = Recommender::new();

    if list {
        return display_component_list(&recommender, format);
    }
    if let Some(component_name) = show {
        return display_component_details(&recommender, &component_name, format);
    }
    if let Some(component_name) = capabilities {
        return display_capabilities(&recommender, &component_name, format);
    }
    if let Some(components) = integrate {
        return display_integration(&recommender, &components, format);
    }
    if interactive {
        return run_interactive_oracle(&recommender);
    }
    if let Some(query_text) = query {
        return oracle_handle_query(&recommender, &query_text, data_size, format);
    }
    if recommend {
        return oracle_handle_recommend(&recommender, problem, format);
    }

    oracle_show_help();
    Ok(())
}

fn display_component_list(
    recommender: &oracle::Recommender,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    println!(
        "{}",
        "🔮 Sovereign AI Stack Components".bright_cyan().bold()
    );
    println!("{}", "─".repeat(50).dimmed());
    println!();

    let components: Vec<_> = recommender.list_components();

    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&components)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## Sovereign AI Stack Components\n");
            println!("| Component | Version | Layer | Description |");
            println!("|-----------|---------|-------|-------------|");
            for name in &components {
                if let Some(comp) = recommender.get_component(name) {
                    println!(
                        "| {} | {} | {} | {} |",
                        comp.name, comp.version, comp.layer, comp.description
                    );
                }
            }
        }
        OracleOutputFormat::Text => {
            // Group by layer
            for layer in oracle::StackLayer::all() {
                let layer_components: Vec<_> = components
                    .iter()
                    .filter_map(|name| recommender.get_component(name))
                    .filter(|c| c.layer == layer)
                    .collect();

                if !layer_components.is_empty() {
                    println!("{} {}", "Layer".bold(), format!("{}", layer).cyan());
                    for comp in layer_components {
                        println!(
                            "  {} {} {} - {}",
                            "•".bright_blue(),
                            comp.name.bright_green(),
                            format!("v{}", comp.version).dimmed(),
                            comp.description.dimmed()
                        );
                    }
                    println!();
                }
            }
        }
    }

    Ok(())
}

fn display_component_details(
    recommender: &oracle::Recommender,
    name: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let Some(comp) = recommender.get_component(name) else {
        println!("{} Component '{}' not found", "❌".red(), name);
        return Ok(());
    };

    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&comp)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## {}\n", comp.name);
            println!("**Version:** {}\n", comp.version);
            println!("**Layer:** {}\n", comp.layer);
            println!("**Description:** {}\n", comp.description);
            println!("### Capabilities\n");
            for cap in &comp.capabilities {
                println!(
                    "- **{}**{}",
                    cap.name,
                    cap.description
                        .as_ref()
                        .map(|d| format!(": {}", d))
                        .unwrap_or_default()
                );
            }
        }
        OracleOutputFormat::Text => {
            println!("{}", format!("📦 {}", comp.name).bright_cyan().bold());
            println!("{}", "─".repeat(50).dimmed());
            println!();
            println!("{}: {}", "Version".bold(), comp.version.cyan());
            println!("{}: {}", "Layer".bold(), format!("{}", comp.layer).cyan());
            println!("{}: {}", "Description".bold(), comp.description);
            println!();
            println!("{}", "Capabilities:".bright_yellow());
            for cap in &comp.capabilities {
                let desc = cap
                    .description
                    .as_ref()
                    .map(|d| format!(" - {}", d.dimmed()))
                    .unwrap_or_default();
                println!("  {} {}{}", "•".bright_blue(), cap.name.green(), desc);
            }
            println!();
        }
    }

    Ok(())
}

fn display_capabilities(
    recommender: &oracle::Recommender,
    name: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let caps = recommender.get_capabilities(name);

    if caps.is_empty() {
        println!("{} No capabilities found for '{}'", "❌".red(), name);
        return Ok(());
    }

    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&caps)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## Capabilities of {}\n", name);
            for cap in &caps {
                println!("- {}", cap);
            }
        }
        OracleOutputFormat::Text => {
            println!(
                "{}",
                format!("🔧 Capabilities of {}", name).bright_cyan().bold()
            );
            println!("{}", "─".repeat(50).dimmed());
            println!();
            for cap in &caps {
                println!("  {} {}", "•".bright_blue(), cap.green());
            }
            println!();
        }
    }

    Ok(())
}

fn display_integration(
    recommender: &oracle::Recommender,
    components: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let parts: Vec<&str> = components.split(',').map(|s| s.trim()).collect();
    if parts.len() != 2 {
        println!(
            "{} Please specify two components separated by comma",
            "❌".red()
        );
        println!(
            "  Example: {} {}",
            "batuta oracle --integrate".cyan(),
            "\"aprender,realizar\"".dimmed()
        );
        return Ok(());
    }

    let from = parts[0];
    let to = parts[1];

    let Some(pattern) = recommender.get_integration(from, to) else {
        println!(
            "{} No integration pattern found from '{}' to '{}'",
            "❌".red(),
            from,
            to
        );
        return Ok(());
    };

    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&pattern)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## Integration: {} → {}\n", from, to);
            println!("**Pattern:** {}\n", pattern.pattern_name);
            println!("**Description:** {}\n", pattern.description);
            if let Some(template) = &pattern.code_template {
                println!("### Code Example\n");
                println!("```rust\n{}\n```", template);
            }
        }
        OracleOutputFormat::Text => {
            println!(
                "{}",
                format!("🔗 Integration: {} → {}", from, to)
                    .bright_cyan()
                    .bold()
            );
            println!("{}", "─".repeat(50).dimmed());
            println!();
            println!("{}: {}", "Pattern".bold(), pattern.pattern_name.cyan());
            println!("{}: {}", "Description".bold(), pattern.description);
            println!();
            if let Some(template) = &pattern.code_template {
                println!("{}", "Code Example:".bright_yellow());
                println!("{}", "─".repeat(40).dimmed());
                for line in template.lines() {
                    println!("  {}", line.dimmed());
                }
                println!("{}", "─".repeat(40).dimmed());
            }
            println!();
        }
    }

    Ok(())
}

fn display_response_markdown(response: &oracle::OracleResponse) {
    println!("## Oracle Recommendation\n");
    println!("**Problem Class:** {}\n", response.problem_class);
    if let Some(algo) = &response.algorithm {
        println!("**Algorithm:** {}\n", algo);
    }
    println!("### Primary Recommendation\n");
    println!("- **Component:** {}", response.primary.component);
    if let Some(path) = &response.primary.path {
        println!("- **Module:** `{}`", path);
    }
    println!(
        "- **Confidence:** {:.0}%",
        response.primary.confidence * 100.0
    );
    println!("- **Rationale:** {}\n", response.primary.rationale);

    if !response.supporting.is_empty() {
        println!("### Supporting Components\n");
        for rec in &response.supporting {
            println!(
                "- **{}** ({:.0}%): {}",
                rec.component,
                rec.confidence * 100.0,
                rec.rationale
            );
        }
        println!();
    }

    println!("### Compute Backend\n");
    println!("- **Backend:** {}", response.compute.backend);
    println!("- **Rationale:** {}\n", response.compute.rationale);

    if response.distribution.needed {
        println!("### Distribution\n");
        println!(
            "- **Tool:** {}",
            response.distribution.tool.as_deref().unwrap_or("N/A")
        );
        println!("- **Rationale:** {}\n", response.distribution.rationale);
    }

    if let Some(code) = &response.code_example {
        println!("### Code Example\n");
        println!("```rust\n{}\n```\n", code);
    }
}

fn display_response_text_primary(response: &oracle::OracleResponse) {
    println!("{}", "🎯 Primary Recommendation".bright_yellow().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!(
        "  {}: {}",
        "Component".bold(),
        response.primary.component.bright_green()
    );
    if let Some(path) = &response.primary.path {
        println!("  {}: {}", "Module".bold(), path.cyan());
    }
    println!(
        "  {}: {}",
        "Confidence".bold(),
        format!("{:.0}%", response.primary.confidence * 100.0).bright_green()
    );
    println!(
        "  {}: {}",
        "Rationale".bold(),
        response.primary.rationale.dimmed()
    );
    println!();
}

fn display_response_text_supporting(response: &oracle::OracleResponse) {
    if response.supporting.is_empty() {
        return;
    }
    println!("{}", "🔧 Supporting Components".bright_yellow().bold());
    println!("{}", "─".repeat(50).dimmed());
    for rec in &response.supporting {
        println!(
            "  {} {} ({:.0}%)",
            "•".bright_blue(),
            rec.component.green(),
            rec.confidence * 100.0
        );
        println!("    {}", rec.rationale.dimmed());
    }
    println!();
}

fn display_response_text_distribution(response: &oracle::OracleResponse) {
    if !response.distribution.needed {
        return;
    }
    println!("{}", "🌐 Distribution".bright_yellow().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!(
        "  {}: {}",
        "Tool".bold(),
        response
            .distribution
            .tool
            .as_deref()
            .unwrap_or("N/A")
            .bright_green()
    );
    if let Some(nodes) = response.distribution.node_count {
        println!("  {}: {}", "Nodes".bold(), nodes);
    }
    println!("  {}", response.distribution.rationale.dimmed());
    println!();
}

fn display_response_text(response: &oracle::OracleResponse) {
    println!();
    println!("{}", "🔮 Oracle Recommendation".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    println!(
        "{} {}: {}",
        "📊".bright_blue(),
        "Problem Class".bold(),
        response.problem_class.cyan()
    );
    if let Some(algo) = &response.algorithm {
        println!(
            "{} {}: {}",
            "🧮".bright_blue(),
            "Algorithm".bold(),
            algo.cyan()
        );
    }
    println!();

    display_response_text_primary(response);
    display_response_text_supporting(response);

    println!("{}", "⚡ Compute Backend".bright_yellow().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!(
        "  {}: {}",
        "Backend".bold(),
        format!("{}", response.compute.backend).bright_green()
    );
    println!("  {}", response.compute.rationale.dimmed());
    println!();

    display_response_text_distribution(response);

    if let Some(code) = &response.code_example {
        println!("{}", "💡 Example Code".bright_yellow().bold());
        println!("{}", "─".repeat(50).dimmed());
        for line in code.lines() {
            println!("  {}", line.dimmed());
        }
        println!("{}", "─".repeat(50).dimmed());
        println!();
    }

    if !response.related_queries.is_empty() {
        println!("{}", "❓ Related Queries".bright_yellow());
        for query in &response.related_queries {
            println!("  {} {}", "→".bright_blue(), query.dimmed());
        }
        println!();
    }

    println!("{}", "═".repeat(60).dimmed());
}

pub fn display_oracle_response(
    response: &oracle::OracleResponse,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        OracleOutputFormat::Markdown => display_response_markdown(response),
        OracleOutputFormat::Text => display_response_text(response),
    }
    Ok(())
}

fn interactive_show_help() {
    println!();
    println!("{}", "Commands:".bright_yellow());
    println!("  {} - Ask a question about the stack", "any text".cyan());
    println!("  {} - List all components", "list".cyan());
    println!("  {} - Show component details", "show <component>".cyan());
    println!("  {} - Show capabilities", "caps <component>".cyan());
    println!("  {} - Exit interactive mode", "exit".cyan());
    println!();
}

/// Check if input is an exit command
fn is_exit_command(input: &str) -> bool {
    input == "exit" || input == "quit"
}

/// Handle an interactive command. Returns Ok(true) to continue, Ok(false) to exit.
fn interactive_handle_command(
    input: &str,
    recommender: &oracle::Recommender,
) -> anyhow::Result<bool> {
    if input.is_empty() {
        return Ok(true);
    }

    if is_exit_command(input) {
        println!();
        println!("{}", "👋 Goodbye!".bright_cyan());
        return Ok(false);
    }

    match input {
        "help" => interactive_show_help(),
        "list" => display_component_list(recommender, OracleOutputFormat::Text)?,
        _ if input.starts_with("show ") => {
            let name = input.strip_prefix("show ").unwrap_or("").trim();
            display_component_details(recommender, name, OracleOutputFormat::Text)?;
        }
        _ if input.starts_with("caps ") => {
            let name = input.strip_prefix("caps ").unwrap_or("").trim();
            display_capabilities(recommender, name, OracleOutputFormat::Text)?;
        }
        _ => {
            let response = recommender.query(input);
            display_oracle_response(&response, OracleOutputFormat::Text)?;
        }
    }
    Ok(true)
}

fn run_interactive_oracle(recommender: &oracle::Recommender) -> anyhow::Result<()> {
    use std::io::{self, Write};

    println!();
    println!("{}", "🔮 Batuta Oracle Mode v1.0".bright_cyan().bold());
    println!(
        "{}",
        "   Ask questions about the Sovereign AI Stack".dimmed()
    );
    println!("{}", "   Type 'exit' or 'quit' to leave".dimmed());
    println!();

    loop {
        print!("{} ", ">".bright_cyan());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !interactive_handle_command(input.trim(), recommender)? {
            break;
        }
    }

    Ok(())
}

fn parse_data_size(s: &str) -> Option<oracle::DataSize> {
    super::parse_data_size_value(s).map(oracle::DataSize::samples)
}
