//! Model Serving Ecosystem Demo
//!
//! Demonstrates the unified model serving interface with:
//! - Chat template formatting
//! - Privacy-aware backend selection
//! - Cost circuit breakers
//! - Context window management
//! - Spillover routing
//!
//! Run with: cargo run --example serve_demo --features native

#[cfg(feature = "native")]
fn main() {
    use batuta::serve::{
        BackendSelector, ChatMessage, ChatTemplateEngine, CircuitBreakerConfig, ContextManager,
        CostCircuitBreaker, PrivacyTier, RouterConfig, RoutingDecision, ServingBackend,
        SpilloverRouter, TemplateFormat, TokenPricing, TruncationStrategy,
    };

    println!("🚀 Model Serving Ecosystem Demo");
    println!("Unified interface for local and remote model serving\n");
    println!("{}", "━".repeat(60));

    // ========================================================================
    // 1. Chat Template Engine
    // ========================================================================
    println!("\n1. CHAT TEMPLATE ENGINE: Unified Prompt Formatting");
    println!("{}", "━".repeat(60));

    let messages = vec![
        ChatMessage::system("You are a helpful AI assistant."),
        ChatMessage::user("What is the capital of France?"),
        ChatMessage::assistant("The capital of France is Paris."),
        ChatMessage::user("What about Germany?"),
    ];

    println!("\n📝 Input Messages:");
    for msg in &messages {
        println!("  {:?}: {}", msg.role, msg.content);
    }

    // Format with different templates
    let formats = [
        ("Llama2", TemplateFormat::Llama2),
        ("Mistral", TemplateFormat::Mistral),
        ("ChatML", TemplateFormat::ChatML),
        ("Alpaca", TemplateFormat::Alpaca),
    ];

    for (name, format) in formats {
        let engine = ChatTemplateEngine::new(format);
        let prompt = engine.apply(&messages);
        println!("\n🔧 {} Format (first 200 chars):", name);
        let preview: String = prompt.chars().take(200).collect();
        println!("  {}", preview.replace('\n', "\n  "));
    }

    // Auto-detect from model name
    println!("\n🔍 Auto-detection from model names:");
    let models = [
        "llama-2-70b-chat",
        "mistral-7b-instruct",
        "gpt-4-turbo",
        "claude-3-sonnet",
    ];
    for model in models {
        let format = TemplateFormat::from_model_name(model);
        println!("  {} → {:?}", model, format);
    }

    // ========================================================================
    // 2. Backend Selection with Privacy Tiers
    // ========================================================================
    println!("\n\n2. BACKEND SELECTION: Privacy-Aware Routing");
    println!("{}", "━".repeat(60));

    let tiers = [
        ("Sovereign", PrivacyTier::Sovereign),
        ("Private", PrivacyTier::Private),
        ("Standard", PrivacyTier::Standard),
    ];

    for (name, tier) in tiers {
        let selector = BackendSelector::new().with_privacy(tier);
        let backends = selector.recommend();
        println!("\n🔒 {} Tier:", name);
        println!("  Recommended: {:?}", backends);
    }

    // Check blocked hosts for Sovereign tier
    println!("\n🚫 Blocked Hosts (Sovereign Tier):");
    let blocked = PrivacyTier::Sovereign.blocked_hosts();
    for host in blocked.iter().take(5) {
        println!("  - {}", host);
    }
    println!("  ... and {} more", blocked.len() - 5);

    // Validate backend against privacy tier
    println!("\n✅ Privacy Validation:");
    let selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);
    let validations = [
        (ServingBackend::Ollama, "Local backend"),
        (ServingBackend::OpenAI, "Public API"),
    ];
    for (backend, desc) in validations {
        match selector.validate(backend) {
            Ok(_) => println!("  {:?} ({}): ✓ Allowed", backend, desc),
            Err(e) => println!("  {:?} ({}): ✗ {}", backend, desc, e),
        }
    }

    // ========================================================================
    // 3. Cost Circuit Breaker
    // ========================================================================
    println!("\n\n3. COST CIRCUIT BREAKER: Budget Protection");
    println!("{}", "━".repeat(60));

    let config = CircuitBreakerConfig {
        daily_budget_usd: 10.0,
        warning_threshold: 0.8,
        max_request_cost_usd: 2.0,
        ..Default::default()
    };

    let breaker = CostCircuitBreaker::new(config.clone());
    println!("\n💰 Budget Configuration:");
    println!("  Daily Budget: ${:.2}", config.daily_budget_usd);
    println!("  Warning at: {:.0}%", config.warning_threshold * 100.0);
    println!("  Max per Request: ${:.2}", config.max_request_cost_usd);

    // Simulate some requests
    println!("\n📊 Simulating Requests:");
    let costs = [0.50, 0.75, 1.00, 0.25];
    for (i, cost) in costs.iter().enumerate() {
        match breaker.check(*cost) {
            Ok(_) => {
                breaker.record(*cost);
                println!(
                    "  Request {}: ${:.2} - ✓ Approved (Total: ${:.2})",
                    i + 1,
                    cost,
                    breaker.accumulated_usd()
                );
            }
            Err(e) => println!("  Request {}: ${:.2} - ✗ {}", i + 1, cost, e),
        }
    }

    // Token pricing
    println!("\n💵 Token Pricing (per 1M tokens):");
    let models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "llama-2"];
    for model in models {
        let pricing = TokenPricing::for_model(model);
        println!(
            "  {}: Input ${:.2}, Output ${:.2}",
            model, pricing.input_per_million, pricing.output_per_million
        );
    }

    // ========================================================================
    // 4. Context Window Management
    // ========================================================================
    println!("\n\n4. CONTEXT MANAGEMENT: Token Counting & Truncation");
    println!("{}", "━".repeat(60));

    let model_contexts = [
        ("gpt-4-turbo", "128K tokens"),
        ("claude-3-sonnet", "200K tokens"),
        ("llama-2-7b", "4K tokens"),
        ("mixtral-8x7b", "32K tokens"),
    ];

    println!("\n📏 Context Windows:");
    for (model, desc) in model_contexts {
        let manager = ContextManager::for_model(model);
        println!(
            "  {}: {} available ({} with output reserve)",
            model,
            desc,
            manager.available_tokens()
        );
    }

    // Token estimation
    let manager = ContextManager::for_model("gpt-4");
    println!("\n🔢 Token Estimation:");
    let test_messages = vec![
        ChatMessage::user("Hello, how are you?"),
        ChatMessage::assistant("I'm doing well, thank you for asking!"),
    ];
    println!(
        "  2 messages: ~{} tokens",
        manager.estimate_tokens(&test_messages)
    );
    println!("  Fits in context: {}", manager.fits(&test_messages));

    // Truncation strategies
    println!("\n✂️ Truncation Strategies:");
    let strategies = [
        (TruncationStrategy::SlidingWindow, "Keep recent messages"),
        (TruncationStrategy::MiddleOut, "Keep first and last"),
        (TruncationStrategy::Error, "Fail on overflow"),
    ];
    for (strategy, desc) in strategies {
        println!("  {:?}: {}", strategy, desc);
    }

    // ========================================================================
    // 5. Spillover Router
    // ========================================================================
    println!("\n\n5. SPILLOVER ROUTER: Hybrid Cloud Load Leveling");
    println!("{}", "━".repeat(60));

    let config = RouterConfig {
        spillover_threshold: 5,
        max_queue_depth: 20,
        local_backend: ServingBackend::Realizar,
        spillover_backends: vec![
            ServingBackend::Groq,
            ServingBackend::Together,
            ServingBackend::Fireworks,
        ],
        spillover_enabled: true,
        ..Default::default()
    };

    let router = SpilloverRouter::new(config);
    println!("\n⚙️ Router Configuration:");
    println!("  Local Backend: {:?}", router.config().local_backend);
    println!(
        "  Spillover Threshold: {}",
        router.config().spillover_threshold
    );
    println!("  Max Queue Depth: {}", router.config().max_queue_depth);
    println!(
        "  Spillover Backends: {:?}",
        router.config().spillover_backends
    );

    // Simulate queue filling
    println!("\n📊 Simulating Load:");
    for i in 0..8 {
        let decision = router.route();
        match &decision {
            RoutingDecision::Local(b) => {
                router.start_request(*b);
                println!(
                    "  Request {}: → {:?} (local queue: {})",
                    i + 1,
                    b,
                    router.local_queue_depth()
                );
            }
            RoutingDecision::Spillover(b) => {
                println!("  Request {}: → {:?} (SPILLOVER)", i + 1, b);
            }
            RoutingDecision::Reject(r) => {
                println!("  Request {}: ✗ Rejected ({})", i + 1, r);
            }
        }
    }

    // Stats
    let stats = router.stats();
    println!("\n📈 Router Stats:");
    println!("  Queue Utilization: {:.1}%", stats.utilization());
    println!("  Near Spillover: {}", stats.near_spillover());
    println!("  Currently Spilling: {}", router.is_spilling());

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n\n{}", "━".repeat(60));
    println!("✨ Model Serving Ecosystem - Toyota Way Principles");
    println!("{}", "━".repeat(60));
    println!("\n  📋 Standardized Work: Chat templates ensure consistency");
    println!("  🔒 Poka-Yoke: Privacy gates prevent data leakage");
    println!("  💰 Muda Elimination: Cost breakers prevent waste");
    println!("  ⚡ Jidoka: Failover maintains context on errors");
    println!("  ⚖️ Heijunka: Spillover routing levels load");
    println!();
}

#[cfg(not(feature = "native"))]
fn main() {
    println!("This example requires the 'native' feature.");
    println!("Run with: cargo run --example serve_demo --features native");
}
