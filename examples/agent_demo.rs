//! Agent Runtime Demo
//!
//! Demonstrates the sovereign agent runtime with:
//! - MockDriver for deterministic testing
//! - MemoryTool for persistent agent state
//! - AgentBuilder for ergonomic API
//! - Stream events for real-time monitoring
//! - RoutingDriver for local-first, remote fallback
//! - ComputeTool for parallel task execution
//!
//! Run with: `cargo run --example agent_demo --features agents`

#[cfg(feature = "agents")]
fn main() {
    use std::sync::Arc;

    use batuta::agent::capability::Capability;
    use batuta::agent::driver::mock::MockDriver;
    use batuta::agent::driver::StreamEvent;
    use batuta::agent::memory::in_memory::InMemorySubstrate;
    use batuta::agent::memory::MemorySubstrate;
    use batuta::agent::runtime::run_agent_loop;
    use batuta::agent::tool::memory::MemoryTool;
    use batuta::agent::tool::ToolRegistry;
    use batuta::agent::{AgentBuilder, AgentManifest};

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    println!("Batuta Agent Runtime Demo");
    println!("{}", "=".repeat(50));
    println!();

    // ────────────────────────────────────────────────
    // Demo 1: Basic agent loop with tool calls
    // ────────────────────────────────────────────────
    println!("--- Demo 1: Tool Call Loop ---");
    println!();

    let manifest = AgentManifest::default();
    println!("Agent: {}", manifest.name);
    println!("Privacy: {:?}", manifest.privacy);
    println!("Capabilities: {:?}", manifest.capabilities);

    let driver = MockDriver::tool_then_response(
        "memory",
        serde_json::json!({
            "action": "remember",
            "content": "The user likes Rust."
        }),
        "I've remembered that you like Rust!",
    );

    let memory = Arc::new(InMemorySubstrate::new());
    let mut registry = ToolRegistry::default();
    registry.register(Box::new(MemoryTool::new(
        memory.clone(),
        "demo-agent".to_string(),
    )));

    println!("Tools: {:?}", registry.tool_names());
    println!();

    let result = rt
        .block_on(run_agent_loop(
            &manifest,
            "Remember that I like Rust.",
            &driver,
            &registry,
            memory.as_ref(),
            None,
        ))
        .expect("agent loop failed");

    println!("Response: {}", result.text);
    println!("Iterations: {}", result.iterations);
    println!("Tool calls: {}", result.tool_calls);

    // Verify memory stored
    let recalled = rt
        .block_on(memory.recall("Rust", 5, None, None))
        .expect("recall failed");
    println!(
        "Memory check: {} fragment(s) for 'Rust'",
        recalled.len()
    );
    for f in &recalled {
        println!("  - {}", f.content);
    }
    println!();

    // ────────────────────────────────────────────────
    // Demo 2: AgentBuilder API
    // ────────────────────────────────────────────────
    println!("--- Demo 2: AgentBuilder API ---");
    println!();

    let builder_driver =
        MockDriver::single_response("Built with AgentBuilder!");

    let result = rt
        .block_on(
            AgentBuilder::new(&manifest)
                .driver(&builder_driver)
                .run("Hello from the builder"),
        )
        .expect("builder run failed");

    println!("Response: {}", result.text);
    println!("Iterations: {}", result.iterations);
    println!();

    // ────────────────────────────────────────────────
    // Demo 3: Stream events
    // ────────────────────────────────────────────────
    println!("--- Demo 3: Stream Events ---");
    println!();

    let stream_driver = MockDriver::tool_then_response(
        "memory",
        serde_json::json!({"action": "recall", "query": "demo"}),
        "Stream demo complete.",
    );

    let stream_memory = Arc::new(InMemorySubstrate::new());
    let mut stream_tools = ToolRegistry::default();
    stream_tools.register(Box::new(MemoryTool::new(
        stream_memory.clone(),
        "stream-demo".to_string(),
    )));

    let (tx, mut rx) = tokio::sync::mpsc::channel(64);

    rt.block_on(run_agent_loop(
        &manifest,
        "Recall demo info",
        &stream_driver,
        &stream_tools,
        stream_memory.as_ref(),
        Some(tx),
    ))
    .expect("stream loop failed");

    println!("Events received:");
    while let Ok(event) = rx.try_recv() {
        match event {
            StreamEvent::PhaseChange { phase } => {
                println!("  Phase: {phase}");
            }
            StreamEvent::ToolUseStart { name, .. } => {
                println!("  Tool start: {name}");
            }
            StreamEvent::ToolUseEnd { name, .. } => {
                println!("  Tool end: {name}");
            }
            StreamEvent::TextDelta { text } => {
                println!("  Text: {text}");
            }
            StreamEvent::ContentComplete { .. } => {
                println!("  Content complete");
            }
        }
    }
    println!();

    // ────────────────────────────────────────────────
    // Demo 4: Capability enforcement (Poka-Yoke)
    // ────────────────────────────────────────────────
    println!("--- Demo 4: Capability Enforcement ---");
    println!();

    let mut restricted = AgentManifest::default();
    restricted.capabilities = vec![Capability::Rag]; // No Memory!

    let cap_driver = MockDriver::tool_then_response(
        "memory",
        serde_json::json!({"action": "recall", "query": "x"}),
        "Tool was denied, responding anyway.",
    );

    let cap_memory = Arc::new(InMemorySubstrate::new());
    let mut cap_tools = ToolRegistry::default();
    cap_tools.register(Box::new(MemoryTool::new(
        cap_memory.clone(),
        "cap-demo".to_string(),
    )));

    let result = rt
        .block_on(run_agent_loop(
            &restricted,
            "Try memory tool",
            &cap_driver,
            &cap_tools,
            cap_memory.as_ref(),
            None,
        ))
        .expect("should succeed despite denied tool");

    println!("Response: {}", result.text);
    println!(
        "Tool calls: {} (denied by capability system)",
        result.tool_calls
    );
    println!();

    // ────────────────────────────────────────────────
    // Demo 5: RoutingDriver (local-first, remote fallback)
    // ────────────────────────────────────────────────
    #[cfg(feature = "native")]
    {
        use batuta::agent::driver::router::RoutingDriver;

        println!("--- Demo 5: RoutingDriver ---");
        println!();

        // Primary succeeds — no fallback needed
        let primary =
            MockDriver::single_response("local inference ok");
        let fallback =
            MockDriver::single_response("remote fallback");

        let routing =
            RoutingDriver::new(Box::new(primary), Box::new(fallback));
        println!(
            "Privacy tier: {:?}",
            <RoutingDriver as batuta::agent::driver::LlmDriver>::privacy_tier(&routing)
        );

        let result = rt
            .block_on(
                AgentBuilder::new(&manifest)
                    .driver(&routing)
                    .run("test routing"),
            )
            .expect("routing failed");
        println!("Response: {}", result.text);
        println!(
            "Spillovers: {}",
            routing.metrics().spillover_count()
        );
        println!();
    }

    // ────────────────────────────────────────────────
    // Demo 6: ComputeTool (parallel task execution)
    // ────────────────────────────────────────────────
    println!("--- Demo 6: ComputeTool ---");
    println!();

    let compute_driver = MockDriver::tool_then_response(
        "compute",
        serde_json::json!({
            "action": "run",
            "command": "echo 'hello from compute'"
        }),
        "Compute task completed.",
    );

    let compute_manifest = AgentManifest {
        capabilities: vec![
            Capability::Memory,
            Capability::Compute,
        ],
        ..AgentManifest::default()
    };

    let compute_memory = Arc::new(InMemorySubstrate::new());
    let mut compute_tools = ToolRegistry::default();

    let cwd = std::env::current_dir()
        .expect("cwd")
        .to_string_lossy()
        .to_string();
    compute_tools.register(Box::new(
        batuta::agent::tool::compute::ComputeTool::new(cwd),
    ));

    let result = rt
        .block_on(run_agent_loop(
            &compute_manifest,
            "Run a compute task",
            &compute_driver,
            &compute_tools,
            compute_memory.as_ref(),
            None,
        ))
        .expect("compute loop failed");
    println!("Response: {}", result.text);
    println!("Tool calls: {}", result.tool_calls);
    println!();

    println!("{}", "=".repeat(50));
    println!("All demos completed successfully.");
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
