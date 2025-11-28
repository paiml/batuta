/// Oracle Mode demonstration
/// Based on oracle-mode-spec.md - Intelligent query interface for Sovereign AI Stack
use batuta::oracle::{
    Backend, DataSize, HardwareSpec, KnowledgeGraph, OracleQuery, ProblemDomain, QueryConstraints,
    Recommender, StackLayer,
};

fn main() {
    println!("🔮 Oracle Mode Demo");
    println!("Intelligent query interface for the Sovereign AI Stack\n");

    // Initialize the Oracle components
    let graph = KnowledgeGraph::sovereign_stack();
    let recommender = Recommender::new();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. KNOWLEDGE GRAPH: Stack Component Registry");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // List all stack layers
    println!("📚 Sovereign AI Stack Layers:");
    for layer in StackLayer::all() {
        let components = graph.components_in_layer(layer);
        println!("  {}: {} components", layer, components.len());
        for comp in components {
            println!("    - {} v{}", comp.name, comp.version);
        }
    }
    println!();

    // Show capability search
    println!("🔍 Finding components with SIMD capability:");
    for comp in graph.find_by_capability("simd") {
        println!("  - {} ({})", comp.name, comp.layer);
    }
    println!();

    // Show domain search
    println!("🧠 Finding components for Supervised Learning domain:");
    for comp in graph.find_by_domain(ProblemDomain::SupervisedLearning) {
        println!("  - {}: {}", comp.name, comp.description);
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. NATURAL LANGUAGE QUERIES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Process natural language queries
    let queries = [
        "How do I run K-means clustering on a large dataset?",
        "What's the fastest way to do matrix multiplication?",
        "I need to transpile PyTorch to Rust with GPU support",
        "Help me optimize random forest training for 10GB of data",
    ];

    for query in &queries {
        println!("📝 Query: \"{}\"", query);
        let response = recommender.query(query);
        println!("   Problem: {}", response.problem_class);
        if let Some(algo) = &response.algorithm {
            println!("   Algorithm: {}", algo);
        }
        println!("   Primary: {} ({:.0}% confidence)", response.primary.component, response.primary.confidence * 100.0);
        println!("   Backend: {:?}", response.compute.backend);
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. RECOMMENDER: Intelligent Component Selection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Backend selection scenarios
    println!("🎯 Backend Selection (Amdahl's Law + PCIe 5× Rule):\n");

    // Scenario 1: Small data, simple operation
    let query1 = OracleQuery::new("matrix multiply")
        .with_data_size(DataSize::samples(64 * 64));
    let result1 = recommender.query_structured(&query1);
    println!("Scenario: Small matrix (64×64)");
    println!("  Backend: {:?}", result1.compute.backend);
    println!("  Rationale: {}\n", result1.compute.rationale);

    // Scenario 2: Large data, complex operation
    let mut constraints2 = QueryConstraints::default();
    constraints2.data_size = Some(DataSize::samples(1_000_000));
    constraints2.hardware = HardwareSpec::with_gpu(16.0);
    let query2 = OracleQuery::new("neural network training large dataset")
        .with_constraints(constraints2);
    let result2 = recommender.query_structured(&query2);
    println!("Scenario: Neural network training (1M samples, GPU available)");
    println!("  Backend: {:?}", result2.compute.backend);
    println!("  Rationale: {}\n", result2.compute.rationale);

    // Scenario 3: Very large distributed workload
    let mut constraints3 = QueryConstraints::default();
    constraints3.data_size = Some(DataSize::samples(1_000_000_000));
    constraints3.hardware.is_distributed = true;
    constraints3.hardware.node_count = Some(8);
    let query3 = OracleQuery::new("random forest distributed repartir")
        .with_constraints(constraints3);
    let result3 = recommender.query_structured(&query3);
    println!("Scenario: Billion-scale random forest (8 nodes)");
    println!("  Backend: {:?}", result3.compute.backend);
    println!("  Distribution: {}", if result3.distribution.needed { "Yes" } else { "No" });
    println!("  Rationale: {}\n", result3.distribution.rationale);

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. INTEGRATION PATTERNS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Show integration patterns
    println!("🔗 Integration patterns from trueno:");
    for pattern in graph.integrations_from("trueno") {
        println!("  trueno → {}: {}", pattern.to, pattern.description);
    }
    println!();

    println!("🔗 Components that integrate with aprender:");
    for pattern in graph.integrations_to("aprender") {
        println!("  {} → aprender: {}", pattern.from, pattern.description);
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. CODE GENERATION EXAMPLES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Show code examples for different queries
    println!("💻 Code example for SIMD matrix operations:");
    let result_simd = recommender.query("matrix multiply simd");
    if let Some(code) = &result_simd.code_example {
        println!("{}\n", code);
    }

    println!("💻 Code example for distributed computing:");
    let mut dist_constraints = QueryConstraints::default();
    dist_constraints.data_size = Some(DataSize::samples(100_000_000));
    dist_constraints.hardware.is_distributed = true;
    dist_constraints.hardware.node_count = Some(4);
    let query_dist = OracleQuery::new("distributed computing cluster repartir")
        .with_constraints(dist_constraints);
    let result_dist = recommender.query_structured(&query_dist);
    if let Some(code) = &result_dist.code_example {
        println!("{}\n", code);
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. BACKEND DECISION CRITERIA");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Backend selection follows Gregg & Hazelwood (2011):\n");
    println!("  {:12} | Criteria", "Backend");
    println!("  {:-<12}-+-{:-<50}", "", "");
    println!("  {:12} | Simple operations, small data", "Scalar");
    println!(
        "  {:12} | SIMD-friendly (matmul, conv, reductions)",
        Backend::SIMD
    );
    println!(
        "  {:12} | compute_time > 5× PCIe_transfer_time",
        Backend::GPU
    );
    println!(
        "  {:12} | Data > single node capacity OR Amdahl P > 0.95",
        Backend::Distributed
    );
    println!();

    println!("✅ Oracle Mode ready for production queries!");
    println!("   Run: batuta oracle --interactive");
}
