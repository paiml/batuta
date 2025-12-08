# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Batuta is the orchestration framework for the **Sovereign AI Stack** — a pure-Rust ecosystem for privacy-preserving ML infrastructure. It coordinates stack components (trueno, aprender, pacha, realizar) and provides transpilation pipelines for converting Python/C/Shell to Rust.

### Stack Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      batuta (Orchestration)                 │
├─────────────────────────────────────────────────────────────┤
│     realizar (Inference)     │     pacha (Model Registry)   │
├──────────────────────────────┴──────────────────────────────┤
│                  aprender (ML Algorithms)                   │
├─────────────────────────────────────────────────────────────┤
│               trueno (SIMD/GPU Compute Primitives)          │
└─────────────────────────────────────────────────────────────┘
```

## Build and Development Commands

```bash
# Build
cargo build                    # Debug build
cargo build --release --locked # Release build

# Testing (uses nextest for parallelism)
make test-fast                 # Fast unit tests (<30s target)
make test                      # Standard tests (<2min target)
make test-full                 # All features enabled
cargo test --lib               # Unit tests only
cargo test --test '*'          # Integration tests only

# Single test
cargo test test_name           # Run specific test
cargo nextest run test_name    # With nextest

# Linting and formatting
make lint                      # Clippy with -D warnings
make fmt                       # Format code
make fmt-check                 # Check formatting

# Coverage (two-phase pattern, temporarily disables mold linker)
make coverage                  # HTML + LCOV reports in target/coverage/

# Quality tiers (Certeza Methodology)
make tier1                     # On-save (<1s): fmt, clippy, check
make tier2                     # Pre-commit (<5s): test --lib, clippy
make tier3                     # Pre-push (1-5min): full tests
make tier4                     # CI/CD: release tests + pmat analysis

# Mutation testing
make mutants-fast              # Quick sample (~5 min)
make mutants                   # Full suite (~30-60 min)
make mutants-file FILE=src/backend.rs  # Specific file

# WASM build
make wasm                      # Debug WASM
make wasm-release              # Optimized WASM

# Documentation
make book                      # Build mdBook
make book-serve                # Serve at localhost:3000
cargo doc --no-deps --open     # API docs
```

## Architecture

### Core Modules

- **`src/pipeline.rs`**: 5-phase transpilation pipeline (Analysis → Transpilation → Optimization → Validation → Build) with Jidoka stop-on-error validation
- **`src/backend.rs`**: Cost-based GPU/SIMD/Scalar selection using 5× PCIe rule (Gregg & Hazelwood, 2011)
- **`src/oracle/`**: Knowledge graph for stack component recommendations with natural language queries
- **`src/serve/`**: Model serving with failover, circuit breakers, privacy tiers (Sovereign/Private/Standard)
- **`src/stack/`**: Dependency graph management, release orchestration, quality gates across stack components

### ML Converters

- **`src/numpy_converter.rs`**: NumPy → Trueno operation mapping
- **`src/sklearn_converter.rs`**: scikit-learn → Aprender algorithm mapping
- **`src/pytorch_converter.rs`**: PyTorch → Realizar operation mapping (inference-only)

### Feature Flags

- `native` (default): Full CLI, filesystem, tracing, TUI dashboard
- `wasm`: Browser-compatible build (no filesystem, in-memory analysis)
- `trueno-integration`: SIMD/GPU tensor operations
- `oracle-mode`: Knowledge graph with trueno-graph and trueno-db

### External Tool Integration

Batuta orchestrates external transpilers detected via PATH:
- **Depyler**: Python → Rust
- **Bashrs**: Shell → Rust
- **Decy**: C/C++ → Rust
- **PMAT**: Quality analysis and TDG scoring

## Design Principles

Toyota Production System principles applied:
- **Jidoka**: Stop-on-error in pipelines, automatic failover
- **Poka-Yoke**: Privacy tiers prevent data leakage
- **Heijunka**: Load leveling via spillover routing
- **Muda**: Cost circuit breakers prevent waste
- **Kaizen**: Continuous optimization via MoE backend selection

## Quality Standards

- 95% minimum test coverage (90% enforced, 95% preferred)
- Zero clippy warnings (with `-D warnings`)
- Mutation testing target: >80% mutation score
- TDG Score: maintain A grade (≥85)
- Pre-commit checks must complete in <30s

## Sovereign AI Stack Ecosystem

### Checking for Updates

```bash
# Check latest versions of all PAIML stack crates
make stack-versions              # or: batuta stack versions

# JSON output for tooling
make stack-versions-json         # or: batuta stack versions --format json

# Check local vs crates.io
make stack-outdated

# Update dependencies
cargo update trueno aprender realizar pacha renacer
```

### Publish Status (O(1) Cached)

```bash
# Check which crates need publishing - O(1) with cache
make stack-publish-status        # or: batuta stack publish-status

# Force refresh (cold cache)
make stack-publish-status-refresh

# Performance:
# - Cold cache: ~7s (parallel crates.io fetches)
# - Warm cache: <100ms (hash-based invalidation)
```

Cache invalidation triggers:
- Cargo.toml content changed
- Git HEAD moved (new commit)
- crates.io TTL expired (15 min)

### Stack Components (crates.io)

| Layer | Crate | Purpose |
|-------|-------|---------|
| Compute | `trueno` | SIMD/GPU primitives (AVX2/AVX-512/NEON, wgpu) |
| Compute | `trueno-db` | GPU-first analytics database, SQL interface |
| Compute | `trueno-graph` | Graph database for code analysis |
| Compute | `trueno-rag` | RAG pipeline (chunking, BM25+vector, RRF) |
| ML | `aprender` | ML algorithms (regression, trees, GNNs, ARIMA) |
| Training | `entrenar` | Autograd, LoRA/QLoRA, quantization, model merge |
| Inference | `realizar` | GGUF/SafeTensors inference engine |
| Data | `alimentar` | Zero-copy Parquet/Arrow data loading |
| Registry | `pacha` | Model registry with Ed25519 signatures |
| Tracing | `renacer` | Syscall tracer with source correlation |
| Transpilers | `depyler`, `bashrs`, `decy` | Python/Shell/C → Rust |
| Orchestration | `batuta` | Stack coordination and CLI |

### Staying Current

My knowledge has a cutoff date. To get latest stack features:

```bash
# Fetch latest from crates.io (cached 15 min)
batuta stack versions

# Check docs.rs for API changes
# https://docs.rs/trueno, https://docs.rs/aprender, etc.

# RSS feeds for releases
# https://crates.io/api/v1/crates/{crate}/versions.rss
```

## Key Dependencies

- **trueno**: SIMD/GPU compute (always use latest from crates.io)
- **renacer**: Syscall tracing for semantic validation
- **pacha**: Model registry integration
- **petgraph**: Dependency graph analysis

## Project-Specific Commands

```bash
# Stack orchestration
batuta stack check              # Dependency health
batuta stack status             # TUI dashboard
batuta stack versions           # Check crates.io versions
batuta stack quality            # Quality matrix
batuta stack gate               # CI quality gate

# Oracle mode (natural language queries)
batuta oracle "How do I train a model?"
batuta oracle --list            # List all components

# Analysis
batuta analyze --languages --tdg .
```

## Claude Code Integration

This project includes `.claude/commands/` for quick access to common tasks:

- `/stack-versions` - Check latest PAIML crate versions from crates.io
- `/stack-check` - Run dependency health check
- `/quality` - Run full quality gate (fmt, clippy, test, coverage)
- `/update-deps` - Check and apply stack dependency updates

These commands provide pre-configured workflows for maintaining the Sovereign AI Stack.
