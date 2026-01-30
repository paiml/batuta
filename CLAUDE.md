# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Batuta is the orchestration framework for the **Sovereign AI Stack** — a pure-Rust ecosystem for privacy-preserving ML infrastructure. It coordinates stack components (trueno, aprender, pacha, realizar) and provides transpilation pipelines for converting Python/C/Shell to Rust.

### Stack Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      batuta (Orchestration)                 │
├─────────────────────────────────────────────────────────────┤
│  whisper.apr (ASR)  │  realizar (Inference)  │ pacha (Reg)  │
├─────────────────────┴────────────────────────┴──────────────┤
│   aprender (ML)   │  entrenar (Training)  │ jugar (Games)   │
├───────────────────┴───────────────────────┴─────────────────┤
│   simular (Simulation)   │   profesor (Education)           │
├──────────────────────────┴──────────────────────────────────┤
│                 repartir (Distributed Compute)              │
│           CPU (Rayon) │ GPU (wgpu) │ Remote (TCP/TLS)       │
├─────────────────────────────────────────────────────────────┤
│  trueno-zram (Compression)  │  trueno-ublk (Block Device)   │
├─────────────────────────────┴───────────────────────────────┤
│               trueno (SIMD/GPU Compute Primitives)          │
│         AVX2/AVX-512/NEON │ wgpu │ LZ4/ZSTD compression     │
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
| Compute | `trueno` | **0.14.x** | SIMD/GPU primitives (AVX2/AVX-512/NEON, wgpu, LZ4) |
| Compute | `trueno-db` | 0.3.x | GPU-first analytics database, SQL interface |
| Compute | `trueno-graph` | 0.1.x | Graph database for code analysis |
| Compute | `trueno-rag` | 0.1.x | RAG pipeline (chunking, BM25+vector, RRF) |
| Compute | `trueno-viz` | 0.1.x | Terminal/PNG visualization |
| Compression | `trueno-zram-core` | 0.3.x | SIMD compression (LZ4/ZSTD, AVX2/AVX-512/NEON, CUDA) |
| Block Device | `trueno-ublk` | 0.1.x | GPU-accelerated ZRAM replacement via ublk |
| Distribution | `repartir` | 2.0.x | Distributed compute (CPU/GPU/Remote, work-stealing) |
| ML | `aprender` | **0.24.x** | ML algorithms, APR v2 format (LZ4/ZSTD compression) |
| Training | `entrenar` | 0.5.x | Autograd, LoRA/QLoRA, quantization, model merge, CITL |
| Inference | `realizar` | **0.5.x** | APR v2/GGUF/SafeTensors inference, GPU kernels |
| Speech | `whisper-apr` | 0.1.x | Pure Rust Whisper ASR (WASM-first, Int4/Int8 quant) |
| Simulation | `simular` | 0.1.x | Unified simulation (Monte Carlo, physics, optimization) |
| Games | `jugar` | 0.1.x | Game engine (ECS, physics, AI, render, audio, WASM) |
| Education | `profesor` | 0.1.x* | Educational platform (courses, quizzes, labs) |
| Data | `alimentar` | 0.2.x | Zero-copy Parquet/Arrow data loading |
| Registry | `pacha` | 0.1.x | Model registry with Ed25519 signatures |
| Tracing | `renacer` | 0.7.x | Syscall tracer with source correlation |
| Quality | `apr-qa` | 0.1.x | APR model QA playbook (test gen, runner, reports) |
| Transpilers | `depyler`, `bashrs`, `decy` | - | Python/Shell/C → Rust |
| Orchestration | `batuta` | 0.6.x | Stack coordination and CLI |

*Not yet published to crates.io

### APR v2 Model Format

The `.apr` format is the stack's native model serialization:

| Feature | APR v1 | APR v2 |
|---------|--------|--------|
| Tensor Compression | None | LZ4/ZSTD |
| Index Format | JSON | Binary |
| Zero-Copy Loading | Partial | Full |
| Quantization | Int8 | Int4/Int8 |
| Streaming | No | Yes |

```rust
// APR v2 with compression
use aprender::apr::{AprModel, Compression};

let model = AprModel::load_compressed("model.apr", Compression::Lz4)?;
```

### repartir Feature Flags

| Feature | Purpose |
|---------|---------|
| `cpu` (default) | Local multi-core execution with work-stealing |
| `gpu` | wgpu GPU compute (Vulkan/Metal/DX12/WebGPU) |
| `remote` | TCP-based distributed execution across machines |
| `remote-tls` | TLS-secured remote execution |
| `tensor` | trueno SIMD tensor integration |
| `checkpoint` | trueno-db + Parquet state persistence |
| `tui` | Job flow TUI visualization |
| `full` | All features enabled |

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

- **trueno**: SIMD/GPU compute with LZ4 compression (0.11.x)
- **repartir**: Distributed compute with CPU/GPU/Remote executors (2.0.x)
- **aprender**: ML algorithms with APR v2 format, LZ4/ZSTD compression (0.24.x)
- **realizar**: Inference engine with APR v2, GPU kernels (0.5.x)
- **whisper-apr**: Pure Rust Whisper ASR, WASM-first (0.1.x)
- **trueno-zram-core**: SIMD/GPU memory compression (0.3.x)
- **trueno-ublk**: GPU-accelerated block device via ublk (0.1.x)
- **entrenar**: Training with autograd, LoRA/QLoRA, CITL (0.5.x)
- **simular**: Simulation engine with Jidoka guards, Heijunka scheduling (0.1.x)
- **jugar**: Game engine with ECS, physics, AI, WASM support (0.1.x)
- **profesor**: Educational platform with quizzes, labs (0.1.x, not on crates.io)
- **renacer**: Syscall tracing for semantic validation (0.9.x)
- **pacha**: Model registry integration (0.2.x)
- **alimentar**: Data loading with Parquet/Arrow (0.2.x)

### Stack Inter-dependencies

```
whisper-apr ► trueno (0.11), aprender (0.24), realizar (0.5)
realizar ───► trueno (0.11), aprender (0.24), alimentar (0.2), pacha (0.2)
aprender ───► trueno (0.11), alimentar (0.2), entrenar (0.5)
entrenar ───► trueno (0.11), aprender (0.24), trueno-db, trueno-rag
trueno-zram-core ► trueno (0.11), CUDA optional
trueno-ublk ► trueno-zram-core, trueno-zram-adaptive, libublk
repartir ───► trueno (0.6+), trueno-db (checkpoint), wgpu (gpu)
jugar ──────► trueno (0.11), aprender (0.24)
simular ────► jugar-probar (testing)
profesor ───► (no_std, minimal deps for WASM)
```

### GPU Kernel Capabilities (realizar)

| Kernel | Purpose |
|--------|---------|
| `GemmKernel` | Matrix multiplication (naive, tiled, tensor core) |
| `AttentionKernel` | FlashAttention-style tiled attention |
| `SoftmaxKernel` | Numerically stable with warp shuffle |
| `LayerNormKernel` | Fused layer normalization |
| `QuantizeKernel` | Q4_K dequantization fused with matmul |
| `Q5KKernel` | Q5_K dequantization |
| `Q6KKernel` | Q6_K dequantization |

### trueno-zram Compression

| Algorithm | Throughput | Use Case |
|-----------|------------|----------|
| LZ4 | 3+ GB/s | High-speed, general purpose |
| ZSTD | 13 GB/s (AVX-512) | Better ratio, compressible data |
| Same-Fill | 2048:1 | Zero/repeated pages |

```rust
use trueno_zram_core::{CompressorBuilder, Algorithm};

let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?;

let compressed = compressor.compress(&page)?;
```

### Distributed Computing with repartir

```bash
# Run distributed computing example
cargo run --example repartir_distributed --features distributed

# Start remote worker (on each node)
cargo run --bin repartir-worker --features remote -- --bind 0.0.0.0:9000

# TUI job flow monitor
cargo run --bin job-flow --features tui,remote
```

Multi-machine GPU/SIMD pattern:
```rust
use repartir::{Pool, task::{Task, Backend}};
use repartir::executor::remote::RemoteExecutor;

// Connect to GPU workers across machines
let executor = RemoteExecutor::builder()
    .add_worker("node1:9000")  // GPU node 1
    .add_worker("node2:9000")  // GPU node 2
    .build().await?;

let task = Task::builder()
    .binary("./gpu-workload")
    .backend(Backend::Gpu)
    .build()?;

let result = executor.execute(task).await?;
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

# Oracle RAG mode (indexed documentation search)
batuta oracle --rag-index       # Index stack docs + ground truth corpora
batuta oracle --rag "tokenization"  # Search indexed docs

# Analysis
batuta analyze --languages --tdg .
```

## Ground Truth Corpora

The Oracle RAG mode indexes external ground truth corpora for cross-language knowledge:

### HuggingFace Ground Truth Corpus

Location: `../hf-ground-truth-corpus`

A curated collection of production-ready Python recipes for HuggingFace ML workflows:
- **95%+ test coverage** with property-based testing (Hypothesis)
- **Module structure**: `hf_gtc.hub`, `hf_gtc.inference`, `hf_gtc.preprocessing`, `hf_gtc.training`
- **Cross-references**: Maps Python patterns to Rust equivalents (candle/trueno)

Oracle query examples:
```bash
batuta oracle --rag "How do I tokenize text for BERT?"
# Returns: hf_gtc/preprocessing/tokenization.py + candle equivalent

batuta oracle --rag "sentiment analysis pipeline"
# Returns: hf_gtc/inference/pipelines.py patterns
```

### Extending Ground Truth

To add new ground truth corpora:
1. Add directory to `python_corpus_dirs` in `src/cli/oracle.rs:cmd_oracle_rag_index()`
2. Ensure corpus has CLAUDE.md and README.md for P0/P1 indexing
3. Python source in `src/**/*.py` is indexed as P2
4. Run `batuta oracle --rag-index` to rebuild index

## Claude Code Integration

This project includes `.claude/commands/` for quick access to common tasks:

- `/stack-versions` - Check latest PAIML crate versions from crates.io
- `/stack-check` - Run dependency health check
- `/quality` - Run full quality gate (fmt, clippy, test, coverage)
- `/update-deps` - Check and apply stack dependency updates

These commands provide pre-configured workflows for maintaining the Sovereign AI Stack.
