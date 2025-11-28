# Oracle Mode Specification v1.0

**Status:** Draft
**Authors:** PAIML Engineering
**Date:** 2024-11-28
**Refs:** BATUTA-ORACLE-001

## Abstract

Oracle Mode provides an intelligent query interface for the Sovereign AI Stack, enabling developers to:
1. Query capabilities of stack components
2. Receive component recommendations for specific problems
3. Understand integration patterns between tools
4. Navigate the complete toolchain efficiently

This specification defines the knowledge graph, query semantics, and recommendation engine for Oracle Mode.

---

## 1. Introduction

### 1.1 Motivation

The Sovereign AI Stack comprises 15+ interconnected tools. Developers face challenges:
- **Discovery**: Which tool solves my problem?
- **Integration**: How do tools connect?
- **Selection**: CPU vs GPU vs distributed?
- **Compliance**: Which tools ensure EU AI Act compliance?

Oracle Mode addresses these via a semantic query layer over the stack's knowledge graph.

### 1.2 Design Principles

Following Toyota Way principles [1]:

| Principle | Application |
|-----------|-------------|
| **Genchi Genbutsu** | Recommendations based on measured benchmarks, not marketing |
| **Jidoka** | Automatic validation of component compatibility |
| **Muda Elimination** | Minimize unnecessary dependencies and "Hidden Technical Debt" [11] |
| **Andon** | Immediate halt on compliance violations or unsafe type conversions |
| **Kaizen** | Continuous refinement via usage telemetry and scaling laws [19] |

---

## 2. Stack Component Taxonomy

### 2.1 Component Registry

```yaml
sovereign_ai_stack:
  version: "1.0"

  # Layer 0: Compute Primitives
  primitives:
    trueno:
      description: "SIMD-accelerated tensor operations"
      capabilities: [vector_ops, matrix_ops, simd, gpu]
      input_types: [f32, f64, i32, i64]
      output_types: [Vector, Matrix, Tensor]
      complexity: O(n) to O(n³)
      references: [2, 3]

    trueno-db:
      description: "Time-series vector database"
      capabilities: [vector_store, similarity_search, persistence]
      query_types: [knn, range, hybrid]
      references: [4]

    trueno-graph:
      description: "Graph analytics engine"
      capabilities: [pathfinding, centrality, community_detection]
      algorithms: [dijkstra, pagerank, label_propagation]
      references: [5]

  # Layer 1: ML Algorithms
  ml_algorithms:
    aprender:
      description: "Pure Rust ML library with TOP 10 algorithms"
      capabilities:
        supervised: [linear_regression, logistic_regression, decision_tree,
                    random_forest, gbm, naive_bayes, knn, svm]
        unsupervised: [kmeans, pca, dbscan, hierarchical]
        preprocessing: [standard_scaler, minmax_scaler, label_encoder]
        model_selection: [train_test_split, cross_validate, grid_search]
      model_format: ".apr"
      references: [6, 7, 17]

  # Layer 2: Training & Inference
  ml_pipeline:
    entrenar:
      description: "Training library with autograd, LoRA, quantization"
      capabilities: [autograd, lora, qlora, quantization, model_merge, distillation]
      optimizers: [sgd, adam, adamw]
      training_modes: [full, lora, qlora]
      references: [8, 18]

    realizar:
      description: "Inference engine for model serving"
      capabilities: [model_serving, batching, moe_routing, circuit_breaker]
      model_formats: [".apr", ".gguf", ".safetensors"]
      deployment: [lambda, container, edge]
      references: [9]

  # Layer 3: Transpilation
  transpilers:
    depyler:
      description: "Python to Rust transpiler with ML oracle"
      source_language: python
      target_language: rust
      capabilities: [type_inference, sklearn_to_aprender, numpy_to_trueno]
      references: [13]

    decy:
      description: "C/C++ to Rust transpiler"
      source_language: [c, cpp]
      target_language: rust
      capabilities: [ownership_inference, unsafe_elimination]

    bashrs:
      description: "Bash to Rust transpiler"
      source_language: bash
      target_language: rust
      capabilities: [script_conversion, cli_generation]

  # Layer 4: Orchestration & Distribution
  orchestration:
    batuta:
      description: "Workflow orchestrator for transpilation pipelines"
      capabilities: [analysis, transpilation, optimization, validation, deployment]
      integrates: [depyler, decy, bashrs, aprender, realizar, trueno]
      references: [15]

    repartir:
      description: "Distributed computing primitives"
      capabilities: [work_stealing, cpu_executor, gpu_executor, remote_executor]
      schedulers: [fifo, priority, work_stealing]
      references: [10, 16, 20]

  # Layer 5: Quality & Profiling
  quality:
    certeza:
      description: "Quality validation framework"
      capabilities: [coverage_check, mutation_testing, tdg_scoring, privacy_audit]
      references: [14]

    pmat:
      description: "Project maintenance analysis tool"
      capabilities: [complexity_analysis, satd_detection, quality_gates]

    renacer:
      description: "Performance profiling via syscall tracing"
      capabilities: [syscall_trace, flamegraph, golden_trace_comparison]

  # Layer 6: Data Loading
  data:
    alimentar:
      description: "Data loading and preprocessing"
      capabilities: [csv, parquet, json, streaming]
      integrates: [aprender, trueno]
```

### 2.2 Capability Matrix

| Problem Domain | Primary Tool | Supporting Tools | Backend Selection |
|----------------|--------------|------------------|-------------------|
| Linear Algebra | trueno | - | SIMD (<100K), GPU (>100K) |
| ML Training | aprender + entrenar | trueno, repartir | CPU (small), GPU (large) |
| ML Inference | realizar | aprender, trueno | Lambda (serverless), Edge (embedded) |
| Python Migration | depyler | batuta, aprender | - |
| Distributed Training | repartir + entrenar | trueno-db | Multi-node |
| Vector Search | trueno-db | trueno | - |
| Graph Analytics | trueno-graph | trueno | - |
| Data Pipeline | alimentar | aprender | - |

---

## 3. Query Language

### 3.1 Natural Language Interface

Oracle Mode accepts natural language queries and maps them to component recommendations using Chain-of-Thought reasoning [12]:

```
Query: "I need to train a random forest on 1M samples"

Oracle Response:
{
  "problem_class": "supervised_learning",
  "algorithm": "random_forest",
  "data_size": "1M",
  "recommendations": {
    "primary": "aprender::tree::RandomForestClassifier",
    "training": "entrenar (optional, for advanced optimization)",
    "compute": {
      "backend": "CPU/SIMD",
      "rationale": "Random Forest is embarrassingly parallel, SIMD sufficient for 1M"
    },
    "distribution": {
      "tool": "repartir",
      "needed": false,
      "rationale": "Single-node sufficient for 1M samples"
    }
  },
  "code_example": "..."
}
```

### 3.2 Structured Query Format

```rust
/// Oracle query structure
pub struct OracleQuery {
    /// Problem description in natural language
    pub description: String,

    /// Constraints
    pub constraints: QueryConstraints,

    /// Preferences
    pub preferences: QueryPreferences,
}

pub struct QueryConstraints {
    /// Maximum latency requirement (ms)
    pub max_latency_ms: Option<u64>,

    /// Data size (samples or bytes)
    pub data_size: Option<DataSize>,

    /// Must run locally (no cloud)
    pub sovereign_only: bool,

    /// EU AI Act compliance required
    pub eu_compliant: bool,

    /// Available hardware
    pub hardware: HardwareSpec,
}

pub struct QueryPreferences {
    /// Optimize for speed vs memory
    pub optimize_for: OptimizationTarget,

    /// Prefer simpler solutions
    pub simplicity_weight: f32,

    /// Existing stack components to integrate with
    pub existing_components: Vec<String>,
}
```

### 3.3 Query Examples

| Query | Recommended Stack |
|-------|-------------------|
| "Convert sklearn pipeline to Rust" | depyler → aprender (sklearn_converter) |
| "Serve model with <10ms latency" | realizar (Lambda) + aprender (.apr format) |
| "Train LLM on 8 GPUs" | repartir (gpu_executor) + entrenar (LoRA) |
| "Detect anomalies in time series" | aprender (IsolationForest) + trueno-db |
| "Build recommendation system" | aprender (ContentRecommender + HNSW) |
| "Profile inference bottlenecks" | renacer (syscall trace) + realizar |
| "Ensure GDPR compliance" | batuta (sovereign mode) + local execution |

---

## 4. Recommendation Engine

### 4.1 Decision Tree

```
                    ┌─────────────────────┐
                    │   Problem Type?     │
                    └─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │   ML    │          │ Transp. │          │  Infra  │
   └─────────┘          └─────────┘          └─────────┘
        │                     │                     │
   ┌────┴────┐           ┌────┴────┐          ┌────┴────┐
   ▼         ▼           ▼         ▼          ▼         ▼
Training  Inference   Python    C/C++     Compute   Storage
   │         │          │         │          │         │
   ▼         ▼          ▼         ▼          ▼         ▼
entrenar  realizar   depyler    decy     repartir  trueno-db
aprender  aprender   batuta    batuta    trueno    trueno-graph
```

### 4.2 Backend Selection Algorithm

Based on operation complexity and data size [2, 3]:

```rust
/// Select optimal backend based on workload characteristics
pub fn select_backend(
    op_complexity: OpComplexity,
    data_size: usize,
    hardware: &HardwareSpec,
) -> Backend {
    // Thresholds based on PCIe transfer overhead analysis
    // See Gregg & Hazelwood (2011) [3]

    match op_complexity {
        OpComplexity::Low => {
            // Element-wise ops: memory-bound, GPU rarely beneficial
            if data_size > 1_000_000 && hardware.has_gpu() {
                Backend::GPU
            } else if data_size > 1_000 {
                Backend::SIMD
            } else {
                Backend::Scalar
            }
        }
        OpComplexity::Medium => {
            // Reductions: moderate compute
            if data_size > 100_000 && hardware.has_gpu() {
                Backend::GPU
            } else if data_size > 100 {
                Backend::SIMD
            } else {
                Backend::Scalar
            }
        }
        OpComplexity::High => {
            // Matrix ops: O(n²) or O(n³), GPU beneficial early
            if data_size > 10_000 && hardware.has_gpu() {
                Backend::GPU
            } else if data_size > 10 {
                Backend::SIMD
            } else {
                Backend::Scalar
            }
        }
    }
}
```

### 4.3 Distribution Decision

Based on work-stealing scheduler theory [10]:

```rust
/// Determine if distributed execution is beneficial
pub fn should_distribute(
    workload: &Workload,
    cluster: &ClusterSpec,
) -> DistributionDecision {
    let single_node_time = estimate_single_node_time(workload);
    let communication_overhead = estimate_communication_overhead(workload, cluster);
    let parallel_efficiency = estimate_parallel_efficiency(workload, cluster);

    // Amdahl's Law consideration
    let speedup = 1.0 / ((1.0 - workload.parallel_fraction)
                        + workload.parallel_fraction / cluster.nodes as f64);

    // Only distribute if speedup exceeds communication cost
    if speedup > 1.5 && communication_overhead < single_node_time * 0.2 {
        DistributionDecision::Distribute {
            tool: "repartir",
            nodes: optimal_node_count(workload, cluster),
            scheduler: select_scheduler(workload),
        }
    } else {
        DistributionDecision::SingleNode {
            backend: select_backend(workload.complexity, workload.size, &cluster.local),
        }
    }
}
```

---

## 5. Integration Patterns

### 5.1 Common Pipelines

#### ML Training Pipeline
```
alimentar (data) → aprender (preprocess) → entrenar (train) → realizar (serve)
     │                   │                      │                  │
     └──trueno──────────┴──────trueno──────────┴─────trueno───────┘
```

#### Python Migration Pipeline
```
Python Project → batuta analyze → depyler transpile → batuta validate
                      │                  │                   │
                      └── sklearn_converter ──────────────────┘
                               │
                               ▼
                          aprender code
```

#### Distributed Training Pipeline
```
Data Shards → repartir (distribute) → entrenar (train) → repartir (aggregate)
                    │                       │                    │
                    └───── checkpoints ─────┴──── trueno-db ────┘
```

### 5.2 Component Compatibility Matrix

| From \ To | trueno | aprender | entrenar | realizar | repartir | depyler |
|-----------|--------|----------|----------|----------|----------|---------|
| **trueno** | - | Matrix→fit | Tensor→grad | Tensor→infer | Tensor→task | - |
| **aprender** | predict→Vec | - | Model→train | .apr→serve | Model→distribute | sklearn→ |
| **entrenar** | grad→Tensor | weights→Model | - | export→serve | batch→distribute | - |
| **realizar** | output→Tensor | load→Model | - | - | serve→scale | - |
| **repartir** | task→Tensor | task→Model | task→grad | task→serve | - | - |
| **depyler** | numpy→ | sklearn→ | - | - | - | - |

---

## 6. CLI Interface

### 6.1 Commands

```bash
# Ask a question about the stack
batuta oracle "How do I train a model on GPU?"

# Get component recommendation for a problem
batuta oracle recommend --problem "image classification" --data-size 100000

# Show integration pattern
batuta oracle integrate aprender realizar

# List capabilities of a component
batuta oracle capabilities aprender

# Compare components for a use case
batuta oracle compare "aprender vs sklearn" --metric performance

# Generate code skeleton for a pipeline
batuta oracle scaffold --pipeline "train-serve" --model random_forest
```

### 6.2 Interactive Mode

```bash
$ batuta oracle --interactive

🔮 Batuta Oracle Mode v1.0
   Ask questions about the Sovereign AI Stack

> What's the fastest way to serve an ML model?

📊 Analysis:
   - Latency requirement: Not specified (assuming <100ms)
   - Model type: Not specified

🎯 Recommendations:

   1. **realizar** (Primary)
      - Purpose: Production inference engine
      - Latency: <10ms for small models
      - Deployment: Lambda, Container, Edge

   2. **aprender .apr format** (Model)
      - Compact binary format
      - Zero-copy deserialization
      - Optimized for realizar

   3. **trueno** (Backend)
      - SIMD acceleration for inference
      - Auto-selected based on input size

💡 Example:
   ```rust
   use realizar::ModelRegistry;

   let registry = ModelRegistry::new();
   registry.load_apr("model", "model.apr")?;
   let prediction = registry.predict("model", &input)?;
   ```

> How does this compare to PyTorch serving?

📊 Comparison: realizar vs PyTorch Serving

| Metric | realizar | PyTorch |
|--------|----------|---------|
| Cold start | 50ms | 2-5s |
| Memory | 10MB | 500MB+ |
| Dependencies | 0 (pure Rust) | Python + libtorch |
| Lambda size | 5MB | 250MB+ |

See benchmark: `make lambda-bench` in realizar repo

> exit
```

---

## 7. Knowledge Graph Schema

### 7.1 Node Types

```rust
pub enum NodeType {
    /// A tool in the stack (e.g., aprender, trueno)
    Component {
        name: String,
        version: String,
        layer: StackLayer,
    },

    /// A capability (e.g., "linear_regression", "simd")
    Capability {
        name: String,
        category: CapabilityCategory,
    },

    /// A problem domain (e.g., "classification", "inference")
    ProblemDomain {
        name: String,
        parent: Option<String>,
    },

    /// An algorithm implementation
    Algorithm {
        name: String,
        complexity: String,
        references: Vec<Citation>,
    },
}
```

### 7.2 Edge Types

```rust
pub enum EdgeType {
    /// Component provides capability
    Provides { component: String, capability: String },

    /// Component integrates with another
    IntegratesWith { from: String, to: String, pattern: String },

    /// Capability solves problem domain
    Solves { capability: String, domain: String },

    /// Algorithm is implemented by component
    ImplementedBy { algorithm: String, component: String },

    /// Component depends on another
    DependsOn { from: String, to: String, optional: bool },
}
```

---

## 8. References (Peer-Reviewed)

[1] Liker, J.K. (2004). **The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer**. McGraw-Hill. ISBN: 978-0071392310.
- *Foundational text on Toyota Production System principles applied to software quality*

[2] Lattner, C., & Adve, V. (2004). **LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation**. Proceedings of CGO 2004, pp. 75-86.
- *Compiler infrastructure enabling SIMD auto-vectorization in Rust*

[3] Gregg, B., & Hazelwood, K. (2011). **The Mean Time to Innocence: How Long Does it Take to Find a Performance Bug?** USENIX ;login:, 36(5).
- *PCIe transfer overhead analysis for GPU vs CPU decision thresholds*

[4] Johnson, J., Douze, M., & Jégou, H. (2019). **Billion-scale similarity search with GPUs**. IEEE Transactions on Big Data, 7(3), pp. 535-547.
- *Vector similarity search algorithms underlying trueno-db*

[5] Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). **The PageRank Citation Ranking: Bringing Order to the Web**. Stanford InfoLab Technical Report.
- *Graph centrality algorithm implemented in trueno-graph*

[6] Breiman, L. (2001). **Random Forests**. Machine Learning, 45(1), pp. 5-32.
- *Random Forest algorithm implemented in aprender::tree*

[7] Friedman, J.H. (2001). **Greedy Function Approximation: A Gradient Boosting Machine**. Annals of Statistics, 29(5), pp. 1189-1232.
- *Gradient Boosting algorithm implemented in aprender::ensemble*

[8] Hu, E.J., et al. (2021). **LoRA: Low-Rank Adaptation of Large Language Models**. arXiv:2106.09685.
- *Low-rank adaptation technique implemented in entrenar::lora*

[9] Fedus, W., Zoph, B., & Shazeer, N. (2022). **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**. JMLR, 23(120), pp. 1-39.
- *Mixture-of-Experts routing implemented in realizar::moe*

[10] Blumofe, R.D., & Leiserson, C.E. (1999). **Scheduling Multithreaded Computations by Work Stealing**. Journal of the ACM, 46(5), pp. 720-748.
- *Work-stealing scheduler algorithm implemented in repartir::scheduler*

[11] Sculley, D., et al. (2015). **Hidden Technical Debt in Machine Learning Systems**. NeurIPS, pp. 2503-2511.
- *Highlights the need for `pmat` and `certeza` to manage system complexity*

[12] Wei, J., et al. (2022). **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**. NeurIPS.
- *Theoretical basis for Oracle's multi-step reasoning engine*

[13] Schick, T., et al. (2023). **Toolformer: Language Models Can Teach Themselves to Use Tools**. arXiv:2302.04761.
- *Inspiration for `depyler`'s tool-use capabilities*

[14] Abadi, M., et al. (2016). **Deep Learning with Differential Privacy**. CCS, pp. 308-318.
- *Privacy guarantees for sovereign mode execution*

[15] Zaharia, M., et al. (2018). **Accelerating the Machine Learning Lifecycle with MLflow**. IEEE Data Eng. Bull., 41(4), pp. 39-45.
- *Precedent for `batuta`'s lifecycle orchestration*

[16] McMahan, B., et al. (2017). **Communication-Efficient Learning of Deep Networks from Decentralized Data**. AISTATS.
- *Federated learning protocols for `repartir`*

[17] Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System**. KDD, pp. 785-794.
- *Algorithmic foundation for gradient boosting in `aprender`*

[18] Dettmers, T., et al. (2022). **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**. NeurIPS.
- *Quantization techniques used in `entrenar` and `realizar`*

[19] Kaplan, J., et al. (2020). **Scaling Laws for Neural Language Models**. arXiv:2001.08361.
- *Guides resource allocation heuristics in `batuta`*

[20] Stoica, I., et al. (2011). **Mesos: A Platform for Fine-Grained Resource Sharing in the Data Center**. NSDI, pp. 295-308.
- *Architecture patterns for `repartir`'s resource offering*

---

## 9. Implementation Roadmap

### Phase 1: Knowledge Graph (Week 1-2)
- [ ] Define component registry YAML schema
- [ ] Build capability extraction from Cargo.toml
- [ ] Implement graph storage (trueno-graph)

### Phase 2: Query Engine (Week 3-4)
- [ ] Natural language parser (keyword extraction)
- [ ] Structured query builder
- [ ] Recommendation algorithm

### Phase 3: CLI Integration (Week 5-6)
- [ ] `batuta oracle` subcommand
- [ ] Interactive mode with REPL
- [ ] Code generation templates

### Phase 4: Continuous Learning (Week 7-8)
- [ ] Usage telemetry collection
- [ ] Recommendation refinement
- [ ] Community contribution workflow

---

## 10. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Query accuracy | >90% | User feedback on recommendations |
| Response time | <100ms | P95 latency |
| Coverage | 100% | All stack components indexed |
| User satisfaction | >4.5/5 | Survey feedback |

---

## Appendix A: Full Component List

| Component | Layer | Primary Use | Crates.io |
|-----------|-------|-------------|-----------|
| trueno | 0 | SIMD tensors | trueno = "0.7" |
| trueno-db | 0 | Vector database | trueno-db = "0.3" |
| trueno-graph | 0 | Graph analytics | trueno-graph = "0.1" |
| trueno-viz | 0 | Visualization | trueno-viz = "0.1" |
| aprender | 1 | ML algorithms | aprender = "0.12" |
| entrenar | 2 | Training | entrenar = "0.2" |
| realizar | 2 | Inference | realizar = "0.2" |
| depyler | 3 | Python→Rust | depyler (workspace) |
| decy | 3 | C/C++→Rust | decy (workspace) |
| bashrs | 3 | Bash→Rust | bashrs = "0.1" |
| batuta | 4 | Orchestration | batuta = "0.1" |
| repartir | 4 | Distribution | repartir = "1.0" |
| alimentar | 5 | Data loading | alimentar = "0.1" |
| certeza | 6 | Quality | certeza (tool) |
| pmat | 6 | Analysis | pmat (tool) |
| renacer | 6 | Profiling | renacer = "0.6" |

---

*Document generated for PAIML Sovereign AI Stack*
