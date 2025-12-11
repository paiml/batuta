# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Batuta is the orchestration framework for the **Sovereign AI Stack** — a pure-Rust ecosystem for privacy-preserving ML infrastructure. It coordinates stack components (trueno, aprender, pacha, realizar) and provides transpilation pipelines for converting Python/C/Shell to Rust.

### Stack Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      batuta (Orchestration)                 │
├─────────────────────────────────────────────────────────────┤
│  realizar (Inference)  │  pacha (Registry)  │ jugar (Games) │
├────────────────────────┴────────────────────┴───────────────┤
│   aprender (ML)   │  entrenar (Training)  │ profesor (Edu)  │
├───────────────────┴───────────────────────┴─────────────────┤
│                     simular (Simulation)                    │
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

| Layer | Crate | Version | Purpose |
|-------|-------|---------|---------|
| Compute | `trueno` | 0.8.x | SIMD/GPU primitives (AVX2/AVX-512/NEON, wgpu) |
| Compute | `trueno-db` | 0.3.x | GPU-first analytics database, SQL interface |
| Compute | `trueno-graph` | - | Graph database for code analysis |
| Compute | `trueno-rag` | 0.1.x | RAG pipeline (chunking, BM25+vector, RRF) |
| Compute | `trueno-viz` | 0.1.x | Terminal/PNG visualization |
| ML | `aprender` | 0.17.0 | ML algorithms (regression, trees, GNNs, ARIMA, .apr format) |
| Training | `entrenar` | 0.2.7 | Autograd, LoRA/QLoRA, quantization, model merge, CITL |
| Inference | `realizar` | 0.2.3 | GGUF/SafeTensors inference engine, model serving |
| Simulation | `simular` | 0.1.0 | Unified simulation engine (Monte Carlo, physics, optimization) |
| Games | `jugar` | 0.1.0 | Game engine (ECS, physics, AI, render, audio, WASM) |
| Education | `profesor` | 0.1.0* | Educational platform (courses, quizzes, labs, physics sim) |
| Data | `alimentar` | 0.2.x | Zero-copy Parquet/Arrow data loading |
| Registry | `pacha` | 0.1.x | Model registry with Ed25519 signatures |
| Tracing | `renacer` | 0.7.x | Syscall tracer with source correlation |
| Transpilers | `depyler`, `bashrs`, `decy` | - | Python/Shell/C → Rust |
| Orchestration | `batuta` | 0.1.x | Stack coordination and CLI |

*Not yet published to crates.io

### Stack Quality Metrics (PMAT)

| Crate | Files | Functions | Health | Complexity | Coverage |
|-------|-------|-----------|--------|------------|----------|
| `jugar` | 104 | 429 | 68.3% | 50/100 | 65% |
| `simular` | 47 | 88 | 70.0% | 55/100 | 65% |
| `realizar` | 79 | 446 | 68.3% | 50/100 | 65% |
| `aprender` | 331 | 1008 | 68.3% | 50/100 | 65% |
| `entrenar` | 253 | 3087 | 68.3% | 50/100 | 65% |
| `profesor` | 24 | 53 | 83.3% | 95/100 | 65% |

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

- **trueno**: SIMD/GPU compute (always use latest from crates.io, currently 0.8.x)
- **aprender**: ML algorithms with .apr model format (0.17.0)
- **entrenar**: Training with autograd, LoRA/QLoRA, CITL (0.2.7)
- **realizar**: Inference engine for GGUF/SafeTensors (0.2.3)
- **simular**: Simulation engine with Jidoka guards, Heijunka scheduling (0.1.0)
- **jugar**: Game engine with ECS, physics, AI, WASM support (0.1.0)
- **profesor**: Educational platform with quizzes, labs, physics sim (0.1.0, not on crates.io)
- **renacer**: Syscall tracing for semantic validation (0.7.x)
- **pacha**: Model registry integration (0.1.x)
- **alimentar**: Data loading with Parquet/Arrow (0.2.x)
- **petgraph**: Dependency graph analysis

### Stack Inter-dependencies

```
jugar ─────► trueno (0.8), aprender (0.17)
simular ───► jugar-probar (testing)
realizar ──► trueno (0.7.4), aprender (0.14), alimentar (0.2), pacha (0.1.2)
aprender ──► trueno (0.8.1), alimentar (0.2.2), entrenar (0.2.6)
entrenar ──► trueno (0.8), aprender (0.15), alimentar (0.2.2), trueno-db, trueno-rag
profesor ──► (no_std, minimal deps for WASM)
```

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
