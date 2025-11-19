# Batuta 🎵

> Orchestration framework for converting **ANY** project (Python, C/C++, Shell) to modern, first-principles Rust

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)

Batuta orchestrates 9 Pragmatic AI Labs transpiler and foundation library tools to enable **semantic-preserving** conversion of legacy codebases to high-performance Rust, complete with GPU acceleration, SIMD optimization, and ML inference capabilities.

## 🚀 Quick Start

```bash
# Install Batuta
cargo install batuta

# Analyze your project
batuta analyze --languages --dependencies --tdg

# Convert to Rust (coming soon)
batuta transpile --incremental --cache

# Optimize with GPU/SIMD (coming soon)
batuta optimize --enable-gpu --profile aggressive

# Validate equivalence (coming soon)
batuta validate --trace-syscalls --benchmark

# Build final binary (coming soon)
batuta build --release
```

## 📖 Documentation

**[Read The Batuta Book](https://paiml.github.io/Batuta/)** - Comprehensive guide covering:
- Philosophy and core principles (Toyota Way applied to code migration)
- The 5-phase workflow (Analysis → Transpilation → Optimization → Validation → Deployment)
- Tool ecosystem deep-dives (all 9 Pragmatic AI Labs tools)
- Practical examples and case studies
- Configuration reference and best practices

## 🎯 What is Batuta?

Batuta is named after the **conductor's baton** – it orchestrates multiple specialized tools to convert legacy code to Rust while maintaining semantic equivalence. Unlike simple transpilers, Batuta:

- **Preserves semantics** through IR-based analysis and validation
- **Optimizes automatically** with SIMD/GPU acceleration via Trueno
- **Provides gradual migration** through Ruchy scripting language
- **Applies Toyota Way principles** (Muda, Jidoka, Kaizen) for quality

## 🧩 Architecture

Batuta orchestrates **9 core components** from Pragmatic AI Labs:

### Transpilers
- **[Decy](https://github.com/paiml/decy)** - C/C++ → Rust with ownership inference
- **[Depyler](https://github.com/paiml/depyler)** - Python → Rust with type inference
- **[Bashrs](https://github.com/paiml/bashrs)** - Shell scripts → Rust CLI

### Foundation Libraries
- **[Trueno](https://github.com/paiml/trueno)** - Multi-target compute (CPU SIMD, GPU, WASM)
- **[Aprender](https://github.com/paiml/aprender)** - First-principles ML in Rust
- **[Realizar](https://github.com/paiml/realizar)** - ML inference runtime

### Quality & Support Tools
- **[Ruchy](https://github.com/paiml/ruchy)** - Rust-oriented scripting for gradual migration
- **[PMAT](https://github.com/paiml/pmat)** - Quality analysis & roadmap generation
- **[Renacer](https://github.com/paiml/renacer)** - Syscall tracing for validation

## 📊 Commands

### `batuta analyze`

Analyze your project to understand languages, dependencies, and code quality.

```bash
# Full analysis
batuta analyze --languages --dependencies --tdg

# Just detect languages
batuta analyze --languages

# Calculate TDG score only
batuta analyze --tdg
```

**Output includes:**
- Language breakdown with line counts and percentages
- Primary language detection
- Transpiler recommendations
- Dependency manager detection (pip, Cargo, npm, etc.)
- Package counts per dependency file
- TDG quality score (0-100) with letter grade
- ML framework detection
- Next steps guidance

### `batuta init` (Coming Soon)

Initialize a Batuta project and set up conversion configuration.

```bash
batuta init --source ./my-python-app --output ./my-rust-app
```

### `batuta transpile` (Coming Soon)

Convert source code to Rust with incremental compilation and caching.

```bash
# Basic transpilation
batuta transpile

# Incremental mode with caching
batuta transpile --incremental --cache

# Specific modules only
batuta transpile --modules auth,api,db

# Generate Ruchy for gradual migration
batuta transpile --ruchy --repl
```

### `batuta optimize` (Coming Soon)

Apply performance optimizations with GPU/SIMD acceleration.

```bash
# Balanced optimization (default)
batuta optimize

# Aggressive optimization
batuta optimize --profile aggressive --enable-gpu

# Custom GPU threshold
batuta optimize --enable-gpu --gpu-threshold 1000
```

**Optimization profiles:**
- `fast` - Quick compilation, basic optimizations
- `balanced` - Default, good compilation/performance trade-off
- `aggressive` - Maximum performance, slower compilation

### `batuta validate` (Coming Soon)

Verify semantic equivalence between original and transpiled code.

```bash
# Full validation suite
batuta validate --trace-syscalls --diff-output --run-original-tests --benchmark

# Quick syscall validation
batuta validate --trace-syscalls
```

### `batuta build` (Coming Soon)

Build optimized Rust binaries with cross-compilation support.

```bash
# Release build
batuta build --release

# Cross-compile
batuta build --target x86_64-unknown-linux-musl

# WebAssembly
batuta build --wasm
```

### `batuta report` (Coming Soon)

Generate comprehensive migration reports.

```bash
# HTML report (default)
batuta report

# Markdown for documentation
batuta report --format markdown --output MIGRATION.md

# JSON for CI/CD
batuta report --format json --output report.json
```

## 🏗️ 5-Phase Workflow

Batuta implements a **5-phase Kanban workflow** based on Toyota Way principles:

### Phase 1: Analysis
- Detect project languages and structure
- Calculate technical debt grade (TDG)
- Identify dependencies and frameworks
- Recommend transpilation strategy

### Phase 2: Transpilation
- Convert code to Rust/Ruchy using appropriate transpiler
- Preserve semantics through IR analysis
- Generate human-readable output
- Support incremental compilation

### Phase 3: Optimization
- Apply SIMD vectorization (via Trueno)
- Enable GPU acceleration for compute-heavy code
- Optimize memory layout
- Select backends via Mixture-of-Experts routing

### Phase 4: Validation
- Trace syscalls to verify equivalence (via Renacer)
- Run original test suite
- Compare outputs and performance
- Generate diff reports

### Phase 5: Deployment
- Build optimized binaries
- Cross-compile for target platforms
- Package for distribution
- Generate migration documentation

## 🎓 Toyota Way Principles

Batuta applies **Lean Manufacturing** principles to code migration:

### Muda (Waste Elimination)
- **StaticFixer integration** - Eliminate duplicate static analysis (~40% reduction)
- **PMAT adaptive analysis** - Focus on critical code, skip boilerplate
- **Decy diagnostics** - Clear, actionable error messages reduce confusion

### Jidoka (Built-in Quality)
- **Ruchy strictness levels** - Gradual quality at migration boundaries
- **Pipeline validation** - Quality checks at each phase
- **Semantic equivalence** - Automated verification via syscall tracing

### Kaizen (Continuous Improvement)
- **MoE optimization** - Continuous performance tuning
- **Incremental features** - Deliver value progressively
- **Feedback loops** - Learn from each migration

### Heijunka (Level Scheduling)
- **Batuta orchestrator** - Balanced load across transpilers
- **Parallel processing** - Efficient resource utilization

### Kanban (Visual Workflow)
- **5-phase tracking** - Clear stage visibility
- **Dependency management** - Automatic task ordering

### Andon (Problem Visualization)
- **Renacer integration** - Runtime behavior analysis
- **TDG scoring** - Quality visibility

## 📈 Example: Python ML Project

```bash
# 1. Analyze the project
$ batuta analyze --languages --dependencies --tdg

📊 Analysis Results
==================================================
Primary language: Python
Total files: 127
Total lines: 8,432

Dependencies:
  • pip (42 packages)
    File: "./requirements.txt"
  • ℹ ML frameworks detected - consider Aprender/Realizar for ML code

Quality Score:
  • TDG Score: 73.2/100 (B)

Recommended transpiler: Depyler (Python → Rust)

# 2. Transpile to Rust (coming soon)
$ batuta transpile --incremental

🔄 Transpiling with Depyler...
  ✓ Converted 127 files (3,891 warnings, 42 errors addressed)
  ✓ NumPy → Trueno: 23 operations
  ✓ sklearn → Aprender: 5 models
  ✓ PyTorch → Realizar: 2 inference pipelines

# 3. Optimize (coming soon)
$ batuta optimize --enable-gpu --profile aggressive

⚡ Optimizing...
  ✓ SIMD vectorization: 234 loops optimized
  ✓ GPU dispatch: 12 operations (threshold: 500 elements)
  ✓ Memory layout: 18 structs optimized

# 4. Validate (coming soon)
$ batuta validate --trace-syscalls --benchmark

✅ Validation passed!
  ✓ Syscall equivalence: 100%
  ✓ Output identical: ✓
  ✓ Performance: 4.2x faster, 62% less memory
```

## 🛠️ Development Status

**Current Version:** 0.1.0 (Alpha)

- ✅ **Phase 1: Analysis** - Complete
  - ✅ Language detection
  - ✅ Dependency analysis
  - ✅ TDG scoring
  - ✅ Transpiler recommendations

- 🚧 **Phase 2: Core Orchestration** - In Progress
  - ⏳ CLI scaffolding (complete)
  - ⏳ Transpilation engine
  - ⏳ 5-phase workflow
  - ⏳ PMAT integration

- 📋 **Phase 3: Advanced Pipelines** - Planned
  - 📋 NumPy → Trueno
  - 📋 sklearn → Aprender
  - 📋 PyTorch → Realizar

- 📋 **Phase 4: Enterprise Features** - Future
  - 📋 Renacer tracing
  - 📋 PARF reference finder

See [roadmap.yaml](docs/roadmaps/roadmap.yaml) for complete ticket breakdown (12 tickets, 572 hours).

## 📖 Documentation

- [Specification](docs/specifications/batuta-orchestration-decy-depyler-trueno-aprender-realizar-ruchy-spec.md) - Complete technical specification
- [Roadmap](docs/roadmaps/roadmap.yaml) - PMAT-tracked development roadmap
- [PMAT Bug Report](PMAT_BUG_REPORT.md) - Known issues with PMAT workflow

## 🤝 Contributing

Batuta is part of the [Pragmatic AI Labs](https://github.com/paiml) ecosystem. Contributions are welcome!

```bash
# Clone and build
git clone https://github.com/paiml/Batuta.git
cd Batuta
cargo build --release

# Run tests
cargo test

# Install locally
cargo install --path .
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Related Projects

- [Decy](https://github.com/paiml/decy) - C/C++ to Rust transpiler
- [Depyler](https://github.com/paiml/depyler) - Python to Rust transpiler
- [Trueno](https://github.com/paiml/trueno) - Multi-target compute library
- [PMAT](https://github.com/paiml/paiml-mcp-agent-toolkit) - Quality analysis toolkit

## 🙏 Acknowledgments

Batuta applies principles from:
- **Toyota Production System** - Muda, Jidoka, Kaizen, Heijunka, Kanban, Andon
- **Lean Software Development** - Value stream optimization
- **First Principles Thinking** - Rebuild from fundamental truths

---

**Batuta** - Because every great orchestra needs a conductor. 🎵
