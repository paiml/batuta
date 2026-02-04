# Example Overview

This chapter provides runnable examples demonstrating batuta's capabilities across the Sovereign AI Stack.

## Running Examples

All examples are in the `examples/` directory and can be run with:

```bash
cargo run --example <example_name>
```

Some examples require specific features:

```bash
# Examples requiring oracle-mode
cargo run --example oracle_demo --features oracle-mode

# Examples requiring inference
cargo run --example serve_demo --features inference

# Examples requiring native features (TUI, tracing)
cargo run --example stack_graph_tui --features native
```

## Example Categories

### Core Pipeline Examples

| Example | Description | Features |
|---------|-------------|----------|
| `pipeline_demo` | 5-phase transpilation pipeline with Jidoka validation | - |
| `backend_selection` | Cost-based GPU/SIMD/Scalar selection | - |
| `moe_routing` | Mixture-of-Experts backend routing | - |
| `full_transpilation` | End-to-end transpilation workflow | - |

### ML Framework Conversion

| Example | Description | Features |
|---------|-------------|----------|
| `numpy_conversion` | NumPy → Trueno operation mapping | - |
| `sklearn_conversion` | scikit-learn → Aprender migration | - |
| `pytorch_conversion` | PyTorch → Realizar conversion | - |

### Oracle Mode Examples

| Example | Description | Features |
|---------|-------------|----------|
| `oracle_demo` | Knowledge graph queries | `oracle-mode` |
| `oracle_local_demo` | Local workspace discovery | `oracle-mode` |
| `rag_oracle_demo` | RAG-enhanced oracle queries | `oracle-mode` |
| `rag_profiling_demo` | RAG query optimization and profiling | - |

### Stack Management

| Example | Description | Features |
|---------|-------------|----------|
| `stack_dogfood` | Self-analysis of batuta codebase | `native` |
| `stack_graph_tui` | TUI visualization of stack dependencies | `native` |
| `stack_quality_demo` | Quality metrics across stack | `native` |
| `stack_diagnostics_demo` | Comprehensive stack health check | `native` |
| `stack_comply_demo` | Cross-project consistency with MinHash+LSH | - |
| `publish_status_demo` | crates.io publish status checker | - |
| `sovereign_stack_e2e` | End-to-end stack validation | - |

### Infrastructure Components

| Example | Description | Features |
|---------|-------------|----------|
| `trueno_zram_demo` | SIMD compression with trueno-zram | - |
| `trueno_ublk_demo` | GPU block device acceleration | - |
| `repartir_distributed` | Distributed computing patterns | - |
| `multi_machine_demo` | Multi-node GPU/SIMD orchestration | - |

### Model Serving

| Example | Description | Features |
|---------|-------------|----------|
| `serve_demo` | Privacy-tiered model serving | `inference` |
| `whisper_apr_demo` | Whisper ASR inference | `inference` |
| `pepita_kernel_demo` | GPU kernel interfaces | - |
| `int8_rescore_demo` | INT8 quantized inference | `inference` |

### Content & Data

| Example | Description | Features |
|---------|-------------|----------|
| `content_demo` | Content analysis and generation | - |
| `hf_catalog_demo` | HuggingFace catalog integration | - |
| `parf_analysis` | PARF (Project ARtifact Format) analysis | - |
| `svg_generation_demo` | Material Design 3 compliant SVG diagrams | - |

### MCP Integration

| Example | Description | Features |
|---------|-------------|----------|
| `mcp_demo` | MCP server integration | - |
| `custom_plugin` | Custom plugin development | - |
| `graph_tui_demo` | Graph visualization TUI | `native` |

## Quick Start Examples

### 1. Pipeline Demo (No Features Required)

```bash
cargo run --example pipeline_demo
```

Demonstrates the 5-phase transpilation pipeline with Jidoka (stop-on-error) validation.

### 2. Oracle Local Demo

```bash
cargo run --example oracle_local_demo --features oracle-mode
```

Discovers PAIML projects in `~/src` and shows their development state (Clean/Dirty/Unpushed).

### 3. Stack Quality Demo

```bash
cargo run --example stack_quality_demo --features native
```

Analyzes quality metrics across the Sovereign AI Stack components.

### 4. Backend Selection Demo

```bash
cargo run --example backend_selection
```

Shows cost-based GPU/SIMD/Scalar backend selection using the 5× PCIe rule.

## Example Dependencies

Some examples have external dependencies:

- **Model files**: Examples in `serve_demo`, `whisper_apr_demo` require GGUF/APR model files
- **GPU**: CUDA examples require NVIDIA GPU with CUDA toolkit
- **Network**: `hf_catalog_demo` requires internet access for HuggingFace API

## Building All Examples

Verify all examples compile:

```bash
cargo check --examples
cargo check --examples --features oracle-mode,native,inference
```

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Next: Python ML Example](./python-ml-example.md)
