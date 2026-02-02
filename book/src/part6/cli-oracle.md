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
| `--format <format>` | Output format: `text` (default), `json`, `markdown`, or `code` |
| `--rag` | Use RAG-based retrieval from indexed stack documentation |
| `--rag-index` | Index/reindex stack documentation for RAG queries |
| `--rag-index-force` | Clear cache and rebuild index from scratch |
| `--rag-stats` | Show cache statistics (fast, manifest only) |
| `--rag-dashboard` | Launch TUI dashboard for RAG index statistics |
| `--local` | Show local workspace status (~/src PAIML projects) |
| `--dirty` | Show only dirty (uncommitted changes) projects |
| `--publish-order` | Show safe publish order respecting dependencies |
| `-h, --help` | Print help information |

## Examples

### List Stack Components

```bash
$ batuta oracle --list

📚 Sovereign AI Stack Components:

Layer 0: Compute Primitives
  - trueno v0.8.8: SIMD-accelerated tensor operations + simulation testing framework
  - trueno-db v0.3.7: High-performance vector database
  - trueno-graph v0.1.4: Graph analytics engine
  - trueno-viz v0.1.5: Visualization toolkit

Layer 1: ML Algorithms
  - aprender v0.19.0: First-principles ML library

Layer 2: Training & Inference
  - entrenar v0.3.0: Training loop framework
  - realizar v0.3.0: ML inference runtime
...
```

### Query Component Details

```bash
$ batuta oracle --show aprender

📦 Component: aprender v0.19.0

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

### Code Output

Extract raw code snippets for piping to other tools. No ANSI escapes, no metadata — just code. All code output includes **TDD test companions** (`#[cfg(test)]` modules) appended after the main code:

```bash
# Extract code from a recipe (includes test companion)
$ batuta oracle --recipe ml-random-forest --format code
use aprender::tree::RandomForest;

let model = RandomForest::new()
    .n_estimators(100)
    .max_depth(Some(10))
    .fit(&x, &y)?;

#[cfg(test)]
mod tests {
    #[test]
    fn test_random_forest_construction() {
        let n_estimators = 100;
        assert!(n_estimators > 0);
    }
    // ... 2-3 more focused tests
}

# Natural language queries also include test companions
$ batuta oracle "train a model" --format code > example.rs

# Pipe to rustfmt and clipboard
$ batuta oracle --recipe training-lora --format code | rustfmt | pbcopy

# Dump all cookbook recipes as code (each includes test companion)
$ batuta oracle --cookbook --format code > all_recipes.rs

# Count test companions
$ batuta oracle --cookbook --format code 2>/dev/null | grep -c '#\[cfg('
34

# Commands without code exit with code 1
$ batuta oracle --list --format code
No code available for --list (try --format text)
$ echo $?
1
```

When the requested context has no code available (e.g., `--list`, `--capabilities`, `--rag`), the process exits with code 1 and a stderr diagnostic suggesting `--format text`.

### RAG-Based Query

Query using Retrieval-Augmented Generation from indexed stack documentation:

```bash
$ batuta oracle --rag "How do I fine-tune a model with LoRA?"

🔍 RAG Oracle Query: "How do I fine-tune a model with LoRA?"

📄 Retrieved Documents (RRF-fused):
  1. entrenar/CLAUDE.md (score: 0.847)
     "LoRA (Low-Rank Adaptation) enables parameter-efficient fine-tuning..."

  2. aprender/CLAUDE.md (score: 0.623)
     "For training workflows, entrenar provides autograd and optimization..."

💡 Recommendation:
   Use `entrenar` for LoRA fine-tuning with quantization support (QLoRA).

💻 Code Example:
   use entrenar::lora::{LoraConfig, LoraTrainer};

   let config = LoraConfig::new()
       .rank(16)
       .alpha(32.0)
       .target_modules(&["q_proj", "v_proj"]);

   let trainer = LoraTrainer::new(model, config);
   trainer.train(&dataset)?;
```

### Index Stack Documentation

Build or update the RAG index from stack CLAUDE.md files and ground truth corpora:

```bash
$ batuta oracle --rag-index

📚 RAG Indexer (Heijunka Mode)
──────────────────────────────────────────────────

Scanning Rust stack repositories...

  ✓ trueno/CLAUDE.md          ████████████░░░ (12 chunks)
  ✓ trueno/README.md          ████████░░░░░░░ (8 chunks)
  ✓ aprender/CLAUDE.md        ██████████████░ (15 chunks)
  ✓ realizar/CLAUDE.md        ████████░░░░░░░ (8 chunks)
  ...

Scanning Python ground truth corpora...

  ✓ hf-ground-truth-corpus/CLAUDE.md      ██████░░░░░░░░░ (6 chunks)
  ✓ hf-ground-truth-corpus/README.md      ████████████░░░ (12 chunks)
  ✓ src/hf_gtc/hub/search.py              ████░░░░░░░░░░░ (4 chunks)
  ✓ src/hf_gtc/preprocessing/tokenization.py ██████░░░░░░░░ (6 chunks)
  ...

──────────────────────────────────────────────────
Complete: 28 documents, 186 chunks indexed

Vocabulary: 3847 unique terms
Avg doc length: 89.4 tokens

Reindexer: 28 documents tracked
```

### Query Ground Truth Corpora

Query for Python ML patterns and get cross-language results:

```bash
$ batuta oracle --rag "How do I tokenize text for BERT?"

🔍 RAG Oracle Mode
──────────────────────────────────────────────────
Index: 28 documents, 186 chunks

Query: How do I tokenize text for BERT?

1. [hf-ground-truth-corpus] src/hf_gtc/preprocessing/tokenization.py#12 ████████░░ 82%
   def preprocess_text(text: str) -> str:
       text = text.strip().lower()...

2. [trueno] trueno/CLAUDE.md#156 ██████░░░░ 65%
   For text preprocessing, trueno provides...

3. [hf-ground-truth-corpus] hf-ground-truth-corpus/README.md#42 █████░░░░░ 58%
   from hf_gtc.preprocessing.tokenization import preprocess_text...

$ batuta oracle --rag "sentiment analysis pipeline"

# Returns Python pipeline patterns + Rust inference equivalents
```

### RAG Cache Statistics

Show index statistics without a full load (reads manifest only):

```bash
$ batuta oracle --rag-stats

📊 RAG Index Statistics
──────────────────────────────────────────────────
Version: 1.0.0
Batuta version: 0.6.2
Indexed at: 2025-01-30 14:23:45 UTC
Cache path: /home/user/.cache/batuta/rag

Sources:
  - trueno: 4 docs, 42 chunks (commit: abc123)
  - aprender: 3 docs, 38 chunks (commit: def456)
  - hf-ground-truth-corpus: 12 docs, 100 chunks
```

### Force Rebuild Index

Rebuild from scratch, ignoring fingerprint-based skip. The old cache is retained until the new index is saved (crash-safe two-phase write):

```bash
$ batuta oracle --rag-index-force

Force rebuild requested (old cache retained until save)...
📚 RAG Indexer (Heijunka Mode)
──────────────────────────────────────────────────

Scanning Rust stack repositories...
  ✓ trueno/CLAUDE.md          ████████████░░░ (12 chunks)
  ...

Complete: 28 documents, 186 chunks indexed
Index saved to /home/user/.cache/batuta/rag
```

### RAG Dashboard

Launch the TUI dashboard to monitor RAG index health:

```bash
$ batuta oracle --rag-dashboard

┌─────────────────────────────────────────────────────────────┐
│                  RAG Oracle Dashboard                       │
├─────────────────────────────────────────────────────────────┤
│ Index Status: HEALTHY          Last Updated: 2 hours ago   │
├─────────────────────────────────────────────────────────────┤
│ Documents by Priority:                                      │
│   P0 (Critical): ████████████████████ 12 CLAUDE.md         │
│   P1 (High):     ████████████         8 README.md          │
│   P2 (Medium):   ██████               4 docs/              │
│   P3 (Low):      ████                 2 examples/          │
├─────────────────────────────────────────────────────────────┤
│ Retrieval Quality (last 24h):                               │
│   MRR:        0.847  ████████████████░░░░                   │
│   Recall@5:   0.923  ██████████████████░░                   │
│   NDCG@10:    0.891  █████████████████░░░                   │
├─────────────────────────────────────────────────────────────┤
│ Reindex Queue (Heijunka):                                   │
│   - entrenar/CLAUDE.md (staleness: 0.72)                    │
│   - realizar/CLAUDE.md (staleness: 0.45)                    │
└─────────────────────────────────────────────────────────────┘
```

### Local Workspace Discovery

Discover PAIML projects in `~/src` with development state awareness:

```bash
$ batuta oracle --local

🏠 Local Workspace Status (PAIML projects in ~/src)

📊 Summary:
  Total projects: 42
  ✅ Clean:       28
  🔧 Dirty:       10
  📤 Unpushed:    4

┌──────────────────┬──────────┬───────────┬────────┬─────────────────┐
│ Project          │ Local    │ Crates.io │ State  │ Git Status      │
├──────────────────┼──────────┼───────────┼────────┼─────────────────┤
│ trueno           │ 0.11.0   │ 0.11.0    │ ✅ Clean │                 │
│ aprender         │ 0.24.0   │ 0.24.0    │ ✅ Clean │                 │
│ depyler          │ 3.21.0   │ 3.20.0    │ 🔧 Dirty │ 15 mod, 3 new   │
│ entrenar         │ 0.5.0    │ 0.5.0     │ 📤 Unpushed │ 2 ahead       │
│ batuta           │ 0.5.0    │ 0.5.0     │ ✅ Clean │                 │
└──────────────────┴──────────┴───────────┴────────┴─────────────────┘

💡 Dirty projects use crates.io version for deps (stable)
```

### Development State Legend

| State | Icon | Meaning |
|-------|------|---------|
| Clean | ✅ | No uncommitted changes, safe to use local version |
| Dirty | 🔧 | Active development, use crates.io version for deps |
| Unpushed | 📤 | Clean but has unpushed commits |

**Key Insight**: Dirty projects don't block the stack! The crates.io version is stable and should be used for dependencies while local development continues.

### Show Only Dirty Projects

Filter to show only projects with uncommitted changes:

```bash
$ batuta oracle --dirty

🔧 Dirty Projects (active development)

┌──────────────────┬──────────┬───────────┬─────────────────────────┐
│ Project          │ Local    │ Crates.io │ Changes                 │
├──────────────────┼──────────┼───────────┼─────────────────────────┤
│ depyler          │ 3.21.0   │ 3.20.0    │ 15 modified, 3 untracked│
│ renacer          │ 0.10.0   │ 0.9.0     │ 8 modified              │
│ pmat             │ 0.20.0   │ 0.19.0    │ 22 modified, 5 untracked│
└──────────────────┴──────────┴───────────┴─────────────────────────┘

💡 These projects are safe to skip - crates.io versions are stable.
   Focus on --publish-order for clean projects ready to release.
```

### Publish Order

Show the safe publish order respecting inter-project dependencies:

```bash
$ batuta oracle --publish-order

📦 Suggested Publish Order (topological sort)

Step 1: trueno-graph (0.1.9 → 0.1.10)
  ✅ Ready - no blockers
  Dependencies: (none)

Step 2: aprender (0.23.0 → 0.24.0)
  ✅ Ready - no blockers
  Dependencies: trueno

Step 3: entrenar (0.4.0 → 0.5.0)
  ✅ Ready - no blockers
  Dependencies: aprender

Step 4: depyler (3.20.0 → 3.21.0)
  ⚠️  Blocked: 15 uncommitted changes
  Dependencies: aprender, entrenar

Step 5: batuta (0.4.9 → 0.5.0)
  ⚠️  Blocked: waiting for depyler
  Dependencies: all stack components

────────────────────────────────────────
📊 Summary:
  Ready to publish: 3 projects
  Blocked: 2 projects

💡 Run 'cargo publish' in order shown above.
   Skip blocked projects - they'll use crates.io stable versions.
```

### Auto-Update System

The RAG index stays fresh automatically through three layers:

**Layer 1: Shell Auto-Fresh (`ora-fresh`)**

```bash
# Runs automatically on shell login (non-blocking background check)
# Manual invocation:
$ ora-fresh
✅ Index is fresh (3h old)

# When a stack repo has been committed since last index:
$ ora-fresh
📚 Stack changed since last index, refreshing...
```

**Layer 2: Post-Commit Hooks**

All 26 stack repos have a post-commit hook that touches a stale marker:

```bash
# Installed in .git/hooks/post-commit across all stack repos
touch "$HOME/.cache/batuta/rag/.stale" 2>/dev/null
```

**Layer 3: Fingerprint-Based Change Detection**

On reindex, BLAKE3 content fingerprints skip work when nothing changed:

```bash
# Second run detects no changes via fingerprints
$ batuta oracle --rag-index
✅ Index is current (no files changed since last index)

# Force reindex ignores fingerprints (old cache retained until save)
$ batuta oracle --rag-index-force
Force rebuild requested (old cache retained until save)...
📚 RAG Indexer (Heijunka Mode)
...
Complete: 5016 documents, 264369 chunks indexed
```

Each `DocumentFingerprint` tracks:
- Content hash (BLAKE3 of file contents)
- Chunker config hash (detect parameter changes)
- Model hash (detect embedding model changes)

## Exit Codes

| Code | Description |
|------|-------------|
| `0` | Success |
| `1` | General error / no code available (`--format code` on non-code context) |
| `2` | Invalid arguments |

## See Also

- [Oracle Mode: Intelligent Query Interface](../part3/oracle-mode.md) - Full documentation
- [`batuta analyze`](./cli-analyze.md) - Project analysis
- [`batuta transpile`](./cli-transpile.md) - Code transpilation

---

**Previous:** [`batuta reset`](./cli-reset.md)
**Next:** [Migration Strategy](../part7/migration-strategy.md)
