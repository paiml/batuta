# `batuta oracle`

Query the Sovereign AI Stack knowledge graph for component recommendations, backend selection, and integration patterns.

## Synopsis

```bash
batuta oracle [OPTIONS] [QUERY]
```

## Description

Oracle Mode provides an intelligent query interface to the Sovereign AI Stack. It analyzes your requirements and recommends:

- **Primary component** for your task
- **Supporting components** that integrate well
- **Compute backend** (Scalar/SIMD/GPU/Distributed)
- **Code examples** ready to use

## Options

| Option | Description |
|--------|-------------|
| `--list` | List all stack components |
| `--show <component>` | Show details about a specific component |
| `--capabilities <cap>` | Find components by capability (e.g., simd, ml, transpilation) |
| `--integrate <from> <to>` | Show integration pattern between two components |
| `--interactive` | Start interactive query mode |
| `--format <format>` | Output format: `text` (default) or `json` |
| `-h, --help` | Print help information |

## Examples

### List Stack Components

```bash
$ batuta oracle --list

📚 Sovereign AI Stack Components:

Layer 0: Compute Primitives
  - trueno v0.7.3: SIMD-accelerated tensor operations
  - trueno-db v0.3.3: High-performance vector database
  - trueno-graph v0.1.1: Graph analytics engine
  - trueno-viz v0.1.1: Visualization toolkit

Layer 1: ML Algorithms
  - aprender v0.12.0: First-principles ML library

Layer 2: Training & Inference
  - entrenar v0.2.1: Training loop framework
  - realizar v0.2.1: ML inference runtime
...
```

### Query Component Details

```bash
$ batuta oracle --show aprender

📦 Component: aprender v0.12.0

Layer: ML Algorithms
Description: Next-generation machine learning library in pure Rust

Capabilities:
  - random_forest (Machine Learning)
  - gradient_boosting (Machine Learning)
  - clustering (Machine Learning)
  - neural_networks (Machine Learning)

Integrates with:
  - trueno: Uses SIMD-accelerated tensor operations
  - realizar: Exports models for inference
  - alimentar: Loads training data

References:
  [1] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32
  [2] Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System
```

### Find by Capability

```bash
$ batuta oracle --capabilities simd

🔍 Components with 'simd' capability:
  - trueno: SIMD-accelerated tensor operations
```

### Natural Language Query

```bash
$ batuta oracle "How do I train a random forest on 1M samples?"

📊 Analysis:
  Problem class: Supervised Learning
  Algorithm: random_forest
  Data size: Large (1M samples)

💡 Primary Recommendation: aprender
   Path: aprender::tree::RandomForest
   Confidence: 95%

🔧 Backend: SIMD
   Rationale: SIMD vectorization optimal for 1M samples

💻 Code Example:
use aprender::tree::RandomForest;

let model = RandomForest::new()
    .n_estimators(100)
    .max_depth(Some(10))
    .fit(&x, &y)?;
```

### Integration Patterns

```bash
$ batuta oracle --integrate depyler aprender

🔗 Integration: depyler → aprender

Pattern: sklearn_migration
Description: Convert sklearn code to aprender

Before (Python/sklearn):
  from sklearn.ensemble import RandomForestClassifier
  model = RandomForestClassifier(n_estimators=100)

After (Rust/aprender):
  use aprender::tree::RandomForest;
  let model = RandomForest::new().n_estimators(100);
```

### Interactive Mode

```bash
$ batuta oracle --interactive

🔮 Oracle Mode - Ask anything about the Sovereign AI Stack

oracle> What's the fastest way to do matrix multiplication?

📊 Analysis:
  Problem class: Linear Algebra

💡 Primary Recommendation: trueno
   Confidence: 85%
   Rationale: SIMD-accelerated matrix operations

💻 Code Example:
use trueno::prelude::*;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape([2, 2]);
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape([2, 2]);
let c = a.matmul(&b);

oracle> exit
Goodbye!
```

### JSON Output

```bash
$ batuta oracle --format json "random forest"

{
  "problem_class": "Supervised Learning",
  "algorithm": "random_forest",
  "primary": {
    "component": "aprender",
    "path": "aprender::tree::RandomForest",
    "confidence": 0.9,
    "rationale": "Random forest for supervised learning"
  },
  "compute": {
    "backend": "SIMD",
    "rationale": "SIMD vectorization optimal"
  },
  "distribution": {
    "needed": false,
    "rationale": "Single-node sufficient"
  }
}
```

## Exit Codes

| Code | Description |
|------|-------------|
| `0` | Success |
| `1` | General error |
| `2` | Invalid arguments |

## See Also

- [Oracle Mode: Intelligent Query Interface](../part3/oracle-mode.md) - Full documentation
- [`batuta analyze`](./cli-analyze.md) - Project analysis
- [`batuta transpile`](./cli-transpile.md) - Code transpilation

---

**Previous:** [`batuta reset`](./cli-reset.md)
**Next:** [Migration Strategy](../part7/migration-strategy.md)
