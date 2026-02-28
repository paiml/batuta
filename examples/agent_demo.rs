//! Agent Runtime Demo
//!
//! Demonstrates the sovereign agent runtime using MockDriver.
//! Run with: `cargo run --example agent_demo --features agents`

#[cfg(feature = "agents")]
fn main() {
    use batuta::agent::driver::mock::MockDriver;
    use batuta::agent::memory::in_memory::InMemorySubstrate;
    use batuta::agent::runtime::run_agent_loop;
    use batuta::agent::tool::memory::MemoryTool;
    use batuta::agent::tool::ToolRegistry;
    use batuta::agent::AgentManifest;

    println!("🤖 Batuta Agent Runtime Demo");
    println!("═══════════════════════════════════════");
    println!();

    // 1. Create a manifest
    let manifest = AgentManifest::default();
    println!("Agent: {}", manifest.name);
    println!("Privacy: {:?}", manifest.privacy);
    println!("Capabilities: {:?}", manifest.capabilities);
    println!();

    // 2. Create a MockDriver that returns a tool call then a response
    let driver = MockDriver::tool_then_response(
        "memory",
        serde_json::json!({
            "action": "remember",
            "content": "The user likes Rust."
        }),
        "I've remembered that you like Rust!",
    );

    // 3. Create components
    let mut registry = ToolRegistry::default();

    // Register memory tool
    let memory = std::sync::Arc::new(InMemorySubstrate::new());
    let memory_tool = MemoryTool::new(
        memory.clone(),
        "demo-agent".to_string(),
    );
    registry.register(Box::new(memory_tool));

    println!("Tools: {:?}", registry.tool_names());
    println!();

    // 4. Run the agent loop
    println!("─── Running Agent Loop ───");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let result = rt.block_on(run_agent_loop(
        &manifest,
        "Remember that I like Rust.",
        &driver,
        &registry,
        memory.as_ref(),
        None, // no stream events
    ));

    match result {
        Ok(result) => {
            println!();
            println!("─── Result ───");
            println!("Response: {}", result.text);
            println!("Iterations: {}", result.iterations);
            println!(
                "Tokens: {} in / {} out",
                result.usage.input_tokens,
                result.usage.output_tokens
            );
            println!("Tool calls: {}", result.tool_calls);
        }
        Err(e) => {
            eprintln!("Agent error: {e}");
        }
    }

    // 5. Verify memory was stored
    println!();
    println!("─── Memory Check ───");
    let recall_result = rt.block_on(
        batuta::agent::memory::MemorySubstrate::recall(
            memory.as_ref(),
            "Rust",
            5,
            None,
            None,
        ),
    );
    match recall_result {
        Ok(fragments) => {
            println!(
                "Recalled {} fragment(s) for 'Rust':",
                fragments.len()
            );
            for f in &fragments {
                println!("  - {}", f.content);
            }
        }
        Err(e) => eprintln!("Recall error: {e}"),
    }
}

#[cfg(not(feature = "agents"))]
fn main() {
    eprintln!(
        "This example requires the 'agents' feature."
    );
    eprintln!(
        "Run: cargo run --example agent_demo --features agents"
    );
}
