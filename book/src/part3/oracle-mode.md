# Oracle Mode

> **"Ask the Oracle, receive the wisdom of the stack."**

Oracle Mode is the intelligent query interface for the Sovereign AI Stack. Instead of manually researching which components to use, Oracle Mode guides you to the optimal solution based on your requirements.

## Overview

Oracle Mode provides:

- **Knowledge Graph:** Complete registry of stack components with capabilities
- **Natural Language Interface:** Query in plain English
- **Intelligent Recommendations:** Algorithm and backend selection
- **Code Generation:** Ready-to-use examples

```
┌──────────────────────────────────────────────────────────────────┐
│                     ORACLE MODE ARCHITECTURE                      │
└──────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  Natural Query  │
                    │   "Train RF"    │
                    └────────┬────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                       QUERY ENGINE                               │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │   Domain    │   │  Algorithm   │   │   Performance        │ │
│  │  Detection  │   │  Extraction  │   │   Hints              │ │
│  └─────────────┘   └──────────────┘   └──────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE GRAPH                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Layer 0: Primitives   → trueno, trueno-db, trueno-graph   │  │
│  │ Layer 1: ML           → aprender                          │  │
│  │ Layer 2: Pipeline     → entrenar, realizar                │  │
│  │ Layer 3: Transpilers  → depyler, decy, bashrs, ruchy      │  │
│  │ Layer 4: Orchestration→ batuta, repartir                  │  │
│  │ Layer 5: Quality      → certeza, pmat, renacer            │  │
│  │ Layer 6: Data         → alimentar                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                      RECOMMENDER                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │  Component  │   │   Backend    │   │   Distribution       │ │
│  │  Selection  │   │   Selection  │   │   Decision           │ │
│  └─────────────┘   └──────────────┘   └──────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                    ┌─────────────────┐
                    │    Response     │
                    │  + Code Example │
                    └─────────────────┘
```

## The Sovereign AI Stack

Oracle Mode knows all 18 components in the stack:

| Layer | Components | Purpose |
|-------|------------|---------|
| **L0: Primitives** | trueno, trueno-db, trueno-graph, trueno-viz | SIMD/GPU compute, vector storage, graph operations |
| **L1: ML** | aprender | First-principles ML algorithms |
| **L2: Pipeline** | entrenar, realizar | Training loops, inference runtime |
| **L3: Transpilers** | depyler, decy, bashrs, ruchy | Python/C transpilers + Rust↔Shell bidirectional |
| **L4: Orchestration** | batuta, repartir, pforge | Migration workflow, distributed compute, MCP servers |
| **L5: Quality** | certeza, pmat, renacer | Testing, profiling, syscall tracing |
| **L6: Data** | alimentar | Data loading and preprocessing |

## Basic Usage

### CLI Interface

```bash
# List all stack components
$ batuta oracle --list

# Show component details
$ batuta oracle --show trueno

# Find components by capability
$ batuta oracle --capabilities simd

# Query integration patterns
$ batuta oracle --integrate aprender realizar

# Interactive mode
$ batuta oracle --interactive
```

### Interactive Mode

```bash
$ batuta oracle --interactive

🔮 Oracle Mode - Ask anything about the Sovereign AI Stack

oracle> How do I train a random forest on 1M samples?

📊 Analysis:
  Problem class: Supervised Learning
  Algorithm: random_forest
  Data size: Large (1M samples)

💡 Primary Recommendation: aprender
   Path: aprender::tree::RandomForest
   Confidence: 95%
   Rationale: Random forest is ideal for large tabular datasets

🔧 Backend: SIMD
   Rationale: SIMD vectorization optimal for 1M samples with High complexity

📦 Supporting Components:
   - trueno (95%): SIMD-accelerated tensor operations
   - alimentar (70%): Parallel data loading

💻 Code Example:
use aprender::tree::RandomForest;
use alimentar::Dataset;

let dataset = Dataset::from_csv("data.csv")?;
let (x, y) = dataset.split_features_target("label")?;

let model = RandomForest::new()
    .n_estimators(100)
    .max_depth(Some(10))
    .n_jobs(-1)  // Use all cores
    .fit(&x, &y)?;

📚 Related Queries:
   - How to optimize random forest hyperparameters?
   - How to serialize trained models with realizar?
   - How to distribute training with repartir?
```

## Backend Selection

Oracle Mode uses **Amdahl's Law** and **PCIe transfer overhead** (Gregg & Hazelwood, 2011) to select the optimal compute backend.

### The 5× Rule

GPU dispatch is only beneficial when compute time exceeds 5× the PCIe transfer time:

```
If compute_time > 5 × transfer_time → Use GPU
Otherwise → Use SIMD
```

### Backend Decision Matrix

| Operation | Complexity | Small Data | Large Data | GPU Available |
|-----------|------------|------------|------------|---------------|
| Element-wise | O(n) | Scalar | SIMD | SIMD (memory-bound) |
| Reductions | O(n) | Scalar | SIMD | SIMD |
| Matrix mult | O(n³) | SIMD | GPU | GPU |
| Conv2D | O(n²k²) | SIMD | GPU | GPU |
| Attention | O(n²d) | SIMD | GPU | GPU |

### Backend Selection Example

```bash
oracle> What backend for 2048×2048 matrix multiplication?

🎯 Backend Selection:
  Operation: Matrix multiplication
  Size: 2048 × 2048 = 4.2M elements
  Complexity: O(n³) = 8.6B FLOPs

  PCIe Transfer: 4.2M × 4 bytes × 2 = 34 MB
  Transfer time: 34 MB / 32 GB/s = 1.06 ms
  Compute time: 8.6B FLOPs / 20 TFLOPS = 0.43 ms

  Ratio: 0.43 / 1.06 = 0.41× (< 5×)

💡 Recommendation: SIMD
   Rationale: PCIe overhead dominates. Use trueno SIMD backend.
   GPU becomes beneficial at ~8192×8192.
```

## Distribution Decision

Oracle uses **Amdahl's Law** for distribution decisions:

```
Speedup = 1 / ((1 - P) + P/N)

Where:
  P = Parallel fraction of workload
  N = Number of nodes
```

### Distribution Example

```bash
oracle> Should I distribute random forest on 4 nodes?

📊 Amdahl's Law Analysis:
  Algorithm: Random Forest
  Parallel fraction: 0.95 (tree training is parallelizable)
  Nodes: 4

  Theoretical speedup: 1 / (0.05 + 0.95/4) = 3.48×
  Communication overhead: ~10% per node = 40%
  Effective speedup: 3.48 × 0.6 = 2.09×

💡 Recommendation: Yes, distribute with repartir
   Expected speedup: 2.09×
   Break-even: 2+ nodes

📦 Code Example:
use repartir::{Executor, WorkStealing};
use aprender::tree::RandomForest;

let executor = Executor::new()
    .with_workers(4)
    .with_scheduler(WorkStealing);

let forest = executor.map(
    trees.chunks(25),
    |chunk| train_tree_subset(chunk, &data)
).await?;
```

## Knowledge Graph Queries

### Find by Capability

```bash
oracle> What components support GPU?

🔍 Components with GPU capability:
  - trueno: SIMD-accelerated tensor operations with GPU dispatch
  - realizar: GPU-accelerated inference runtime
```

### Find by Domain

```bash
oracle> What do I need for graph analytics?

🧠 Graph Analytics Components:
  - trueno-graph: Graph traversal and algorithms
  - trueno-db: Vector storage with graph indexes
```

### Integration Patterns

```bash
oracle> How do I integrate depyler with aprender?

🔗 Integration: depyler → aprender

Pattern: sklearn_migration
Description: Convert sklearn code to aprender

Example:
# Original Python (sklearn)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# After depyler transpilation
use aprender::tree::RandomForest;
let model = RandomForest::new()
    .n_estimators(100)
    .fit(&x, &y)?;
```

## Academic Foundations

Oracle Mode is grounded in peer-reviewed research:

| Concept | Reference | Application |
|---------|-----------|-------------|
| PCIe overhead | Gregg & Hazelwood (2011) | Backend selection |
| Amdahl's Law | Amdahl (1967) | Distribution decisions |
| Roofline model | Williams et al. (2009) | Performance bounds |
| SIMD vectorization | Fog (2022) | Optimization hints |
| Decision trees | Breiman (2001) | Algorithm recommendations |

## JSON Output

For programmatic access, use `--format json`:

```bash
$ batuta oracle --format json "random forest large data"
```

```json
{
  "problem_class": "Supervised Learning",
  "algorithm": "random_forest",
  "primary": {
    "component": "aprender",
    "path": "aprender::tree::RandomForest",
    "confidence": 0.95,
    "rationale": "Random forest is ideal for large tabular datasets"
  },
  "supporting": [
    {
      "component": "trueno",
      "confidence": 0.95,
      "rationale": "SIMD-accelerated tensor operations"
    }
  ],
  "compute": {
    "backend": "SIMD",
    "rationale": "SIMD vectorization optimal for large datasets"
  },
  "distribution": {
    "needed": false,
    "rationale": "Single-node sufficient for this workload size"
  },
  "code_example": "use aprender::tree::RandomForest;..."
}
```

## Programmatic API

Use Oracle Mode from Rust code:

```rust
use batuta::oracle::{Recommender, OracleQuery, DataSize};

// Natural language query
let recommender = Recommender::new();
let response = recommender.query("train random forest on 1M samples");

println!("Primary: {}", response.primary.component);
println!("Backend: {:?}", response.compute.backend);

// Structured query with constraints
let query = OracleQuery::new("neural network training")
    .with_data_size(DataSize::samples(1_000_000))
    .with_hardware(HardwareSpec::with_gpu(16.0))
    .sovereign_only();

let response = recommender.query_structured(&query);

if response.distribution.needed {
    println!("Distribute with: {:?}", response.distribution.tool);
}
```

## Key Takeaways

- **Query naturally:** Ask in plain English, get precise answers
- **Trust the math:** Backend selection based on PCIe and Amdahl analysis
- **Complete stack:** All 18 components indexed with capabilities
- **Code ready:** Get working examples, not just recommendations
- **Reproducible:** JSON output for automation and CI/CD

## Next Steps

Try Oracle Mode yourself:

```bash
# Run the demo
cargo run --example oracle_demo

# Start interactive mode
batuta oracle --interactive

# Query from CLI
batuta oracle "How do I migrate sklearn to Rust?"
```

---

**Previous:** [Renacer: Syscall Tracing](./renacer.md)
**Next:** [Example Overview](../part4/example-overview.md)
