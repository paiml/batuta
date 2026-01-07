<div align="center">

<p align="center">
  <img src=".github/batuta-hero.svg" alt="batuta" width="800">
</p>

<h1 align="center">batuta</h1>

<p align="center">
  <b>Orchestration framework for the Sovereign AI Stack — privacy-preserving ML infrastructure in pure Rust</b>
</p>

<p align="center">
  <a href="https://github.com/paiml/batuta/actions/workflows/ci.yml"><img src="https://github.com/paiml/batuta/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/batuta"><img src="https://img.shields.io/crates/v/batuta.svg" alt="Crates.io"></a>
  <a href="https://docs.rs/batuta"><img src="https://docs.rs/batuta/badge.svg" alt="Documentation"></a>
  <a href="https://paiml.github.io/batuta/"><img src="https://img.shields.io/badge/📚_book-online-brightgreen" alt="Book"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
</p>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Stack Components](#stack-components)
- [Architecture](#architecture)
- [CLI Reference](#cli-reference)
- [License](#license)

## Overview

Batuta coordinates the **Sovereign AI Stack**, a comprehensive pure-Rust ecosystem for organizations requiring complete control over their ML infrastructure. The stack enables privacy-preserving inference, model management, and data processing without external cloud dependencies.

### Key Capabilities

- **Privacy Tiers**: Sovereign (local-only), Private (VPC), Standard (cloud-enabled)
- **Model Security**: Ed25519 signatures, ChaCha20-Poly1305 encryption, BLAKE3 content addressing
- **API Compatibility**: OpenAI-compatible endpoints for drop-in replacement
- **Observability**: Prometheus metrics, distributed tracing, A/B testing
- **Cost Control**: Circuit breakers with configurable daily budgets

## Installation

```bash
cargo install batuta
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
batuta = "0.4"
```

## Quick Start

```bash
# Analyze project structure and dependencies
batuta analyze --languages --dependencies --tdg

# Query the Sovereign AI Stack
batuta oracle "How do I serve a Llama model locally?"

# Model registry operations
batuta pacha pull llama3-8b-q4
batuta pacha sign model.gguf --identity alice@example.com
batuta pacha verify model.gguf

# Encrypt models for distribution
batuta pacha encrypt model.gguf --password-env MODEL_KEY
batuta pacha decrypt model.gguf.enc --password-env MODEL_KEY
```

## Demo

[![asciicast](https://asciinema.org/a/demo.svg)](https://paiml.github.io/batuta/)

**Live Demo**: [paiml.github.io/batuta](https://paiml.github.io/batuta/) | [API Docs](https://docs.rs/batuta)

**Example Output** (`batuta analyze --tdg`):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📊 Technical Debt Gradient Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Project: my-project
  Language: Rust (confidence: 98%)

  Metrics:
    Cyclomatic Complexity:  4.2 avg (good)
    Test Coverage:          87% (A-)
    Documentation:          92% (A)
    Dependency Health:      95% (A+)

  TDG Score: 91.5/100 (A)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Stack Components

Batuta orchestrates a layered architecture of pure-Rust components:

```
┌─────────────────────────────────────────────────────────────┐
│                    batuta v0.4.5                            │
│                 (Orchestration Layer)                       │
├─────────────────────────────────────────────────────────────┤
│     realizar v0.5        │         pacha v0.2               │
│   (Inference Engine)     │      (Model Registry)            │
├──────────────────────────┴──────────────────────────────────┤
│   aprender v0.24   │  entrenar v0.5  │  alimentar v0.2      │
│    (ML Algorithms) │    (Training)   │   (Data Loading)     │
├─────────────────────────────────────────────────────────────┤
│   trueno v0.11     │  repartir v2.0  │   renacer v0.9       │
│ (SIMD/GPU Compute) │  (Distributed)  │  (Syscall Tracing)   │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Version | Description |
|-----------|---------|-------------|
| [trueno](https://crates.io/crates/trueno) | 0.11 | SIMD/GPU compute primitives (AVX2/AVX-512/NEON, wgpu) |
| [aprender](https://crates.io/crates/aprender) | 0.24 | ML algorithms: regression, trees, clustering, NAS |
| [entrenar](https://crates.io/crates/entrenar) | 0.5 | Training: autograd, LoRA/QLoRA, quantization |
| [realizar](https://crates.io/crates/realizar) | 0.5 | Inference engine for GGUF/SafeTensors models |
| [pacha](https://crates.io/crates/pacha) | 0.2 | Model registry with signatures, encryption, lineage |
| [repartir](https://crates.io/crates/repartir) | 2.0 | Distributed compute (CPU/GPU/Remote executors) |
| [renacer](https://crates.io/crates/renacer) | 0.9 | Syscall tracing with semantic validation |
| [batuta](https://crates.io/crates/batuta) | 0.4 | Stack orchestration, drift detection, CLI |

### Extended Ecosystem

| Component | Version | Description |
|-----------|---------|-------------|
| [trueno-db](https://crates.io/crates/trueno-db) | 0.3 | GPU-accelerated analytics database |
| [trueno-graph](https://crates.io/crates/trueno-graph) | 0.1 | Graph database for code analysis |
| [trueno-rag](https://crates.io/crates/trueno-rag) | 0.1 | RAG pipeline (chunking, BM25+vector, RRF) |
| [trueno-viz](https://crates.io/crates/trueno-viz) | 0.1 | Terminal/PNG visualization |
| [alimentar](https://crates.io/crates/alimentar) | 0.2 | Zero-copy Parquet/Arrow data loading |
| [whisper-apr](https://crates.io/crates/whisper-apr) | 0.1 | Pure Rust Whisper ASR (WASM-first) |
| [jugar](https://crates.io/crates/jugar) | 0.1 | Game engine (ECS, physics, AI, WASM) |
| [simular](https://crates.io/crates/simular) | 0.2 | Simulation engine (Monte Carlo, physics) |
| [bashrs](https://crates.io/crates/bashrs) | 6.51 | Shell-to-Rust transpiler and linter |
| [presentar](https://crates.io/crates/presentar) | 0.1 | Terminal presentation framework |
| [pmat](https://crates.io/crates/pmat) | 2.213 | Project quality analysis toolkit |

## Commands

### `batuta analyze`

Analyze project structure, languages, and dependencies:

```bash
batuta analyze --languages --dependencies --tdg

# Output:
# Primary language: Python
# Dependencies: pip (42 packages), ML frameworks detected
# TDG Score: 73.2/100 (B)
# Recommended: Use Aprender for ML, Realizar for inference
```

### `batuta oracle`

Query the stack for component recommendations:

```bash
# Natural language queries
batuta oracle "Train random forest on 1M samples"

# List all components
batuta oracle --list

# Component details
batuta oracle --show realizar

# Interactive mode
batuta oracle --interactive
```

### `batuta pacha`

Model registry operations:

```bash
# Pull models from registry
batuta pacha pull llama3-8b-q4

# Generate signing keys
batuta pacha keygen --identity alice@example.com

# Sign models for distribution
batuta pacha sign model.gguf --identity alice@example.com

# Verify model signatures
batuta pacha verify model.gguf

# Encrypt models at rest
batuta pacha encrypt model.gguf --password-env MODEL_KEY

# Decrypt for inference
batuta pacha decrypt model.gguf.enc --password-env MODEL_KEY
```

### `batuta content`

Generate structured content with quality constraints:

```bash
# Available content types
batuta content types

# Generate book chapter prompt
batuta content emit --type bch --title "Error Handling" --audience "developers"

# Validate content quality
batuta content validate --type bch chapter.md
```

### `batuta stack`

Manage the Sovereign AI Stack ecosystem:

```bash
# Check stack component versions
batuta stack versions

# Detect version drift across published crates
batuta stack drift

# Generate fix commands for drift issues
batuta stack drift --fix --workspace ~/src

# Check which crates need publishing
batuta stack publish-status

# Quality gate for CI/pre-commit
batuta stack gate
```

**Automatic Drift Detection**: Batuta blocks all commands if published stack crates
are using outdated versions of other stack crates. Use `--unsafe-skip-drift-check`
to bypass in emergencies.

## Privacy Tiers

The stack enforces data sovereignty through configurable privacy tiers:

| Tier | Behavior | Use Case |
|------|----------|----------|
| **Sovereign** | Blocks ALL external API calls | Healthcare, Government |
| **Private** | VPC/dedicated endpoints only | Financial services |
| **Standard** | Public APIs allowed | General deployment |

```rust
use batuta::serve::{BackendSelector, PrivacyTier};

let selector = BackendSelector::new()
    .with_privacy(PrivacyTier::Sovereign);

// Returns only local backends: Realizar, Ollama, LlamaCpp
let backends = selector.recommend();
```

## Model Security

### Digital Signatures (Ed25519)

Verify model integrity before loading:

```rust
use pacha::signing::{SigningKey, sign_model, verify_model};

let signing_key = SigningKey::generate();
let signature = sign_model(&model_data, &signing_key)?;

// Verification fails if model tampered
verify_model(&model_data, &signature)?;
```

### Encryption at Rest (ChaCha20-Poly1305)

Protect models during distribution:

```rust
use pacha::crypto::{encrypt_model, decrypt_model};

let encrypted = encrypt_model(&model_data, "password")?;
let decrypted = decrypt_model(&encrypted, "password")?;
```

## Documentation

- **[The Batuta Book](https://paiml.github.io/batuta/)** — Comprehensive guide
- **[Sovereign AI Stack Book](https://paiml.github.io/sovereign-ai-stack-book/)** — Complete stack tutorial with 22 chapters
- **[API Documentation](https://docs.rs/batuta)** — Rust API reference
- **[Specifications](docs/specifications/)** — Technical specifications

## Design Principles

Batuta applies Toyota Production System principles:

| Principle | Application |
|-----------|-------------|
| **Jidoka** | Automatic failover with context preservation |
| **Poka-Yoke** | Privacy tiers prevent data leakage |
| **Heijunka** | Spillover routing for load leveling |
| **Muda** | Cost circuit breakers prevent waste |
| **Kaizen** | Continuous metrics and optimization |

## Development

```bash
# Clone repository
git clone https://github.com/paiml/batuta.git
cd batuta

# Build
cargo build --release

# Run tests
cargo test

# Build documentation
mdbook build book
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository and create your branch from `main`
2. **Run tests** before submitting: `cargo test --all-features`
3. **Run lints**: `cargo clippy --all-targets --all-features -- -D warnings`
4. **Format code**: `cargo fmt --all`
5. **Update documentation** for any API changes
6. **Submit a pull request** with a clear description

See our [CI workflow](.github/workflows/ci.yml) for the full test suite.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Links

- [crates.io/crates/batuta](https://crates.io/crates/batuta)
- [GitHub Repository](https://github.com/paiml/batuta)
- [Documentation Book](https://paiml.github.io/batuta/)
- [Sovereign AI Stack Specification](docs/specifications/sovereign-ai-spec.md)

---

**Batuta** — Orchestrating sovereign AI infrastructure.
