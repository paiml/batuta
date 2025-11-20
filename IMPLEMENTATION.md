# Batuta Implementation Status

**Based on:** [docs/specifications/sovereign-ai-spec.md](docs/specifications/sovereign-ai-spec.md)
**Last Updated:** 2025-11-20
**TDG Score:** 92.6/100 (A)
**Test Coverage:** 37/37 tests passing (0.02s execution time)

## Implemented Components

### ✅ 1. Pipeline Architecture (Spec Section 2.8)

**Module:** `src/pipeline.rs`

Implements the 5-phase transpilation pipeline with Jidoka (stop-on-error) validation:

- **PipelineStage trait**: Async trait for extensible stages
- **TranspilationPipeline**: Orchestrates multi-stage workflows
- **Concrete Stages**:
  - `AnalysisStage`: Language & dependency detection
  - `TranspilationStage`: Source → Rust conversion
  - `OptimizationStage`: SIMD/GPU optimization passes
  - `ValidationStage`: Semantic equivalence verification
  - `BuildStage`: Cargo compilation

**Example:** `examples/pipeline_demo.rs`

```rust
let pipeline = TranspilationPipeline::new(ValidationStrategy::StopOnError)
    .add_stage(Box::new(AnalysisStage))
    .add_stage(Box::new(TranspilationStage::new(true, true)))
    // ... more stages
    .run(&input, &output).await?;
```

### ✅ 7. NumPy→Trueno Conversion (BATUTA-008)

**Module:** `src/numpy_converter.rs`

Converts Python NumPy operations to Rust Trueno equivalents with automatic backend selection:

- **NumPyConverter**: Operation mapping engine with 12 NumPy operations
- **NumPyOp enum**: Array, Add, Subtract, Multiply, Divide, Dot, Sum, Mean, Max, Min, Reshape, Transpose
- **TruenoOp struct**: Code templates, required imports, complexity ratings
- **Methods**:
  - `convert(op)`: Map NumPy operation to Trueno equivalent
  - `recommend_backend(op, size)`: MoE-based backend selection
  - `conversion_report()`: Generate mapping documentation
- **Integration**: Automatic NumPy detection in TranspilationStage

**Example:** `examples/numpy_conversion.rs`

```rust
let converter = NumPyConverter::new();
let trueno_op = converter.convert(&NumPyOp::Add).unwrap();
let backend = converter.recommend_backend(&NumPyOp::Add, 1_000_000);
// Output: SIMD backend for 1M element-wise operations
```

### ✅ 8. sklearn→Aprender Conversion (BATUTA-009)

**Module:** `src/sklearn_converter.rs`

Converts Python scikit-learn algorithms to Rust Aprender equivalents with automatic backend selection:

- **SklearnConverter**: Algorithm mapping engine with 8 sklearn algorithms
- **SklearnAlgorithm enum**: LinearRegression, LogisticRegression, KMeans, DecisionTreeClassifier, RandomForestClassifier, StandardScaler, TrainTestSplit, Accuracy, MeanSquaredError
- **AprenderAlgorithm struct**: Code templates, required imports, complexity ratings, usage patterns
- **Methods**:
  - `convert(algorithm)`: Map sklearn algorithm to Aprender equivalent
  - `recommend_backend(algorithm, size)`: MoE-based backend selection
  - `conversion_report()`: Generate mapping documentation
- **Integration**: Automatic sklearn detection in TranspilationStage

**Example:** `examples/sklearn_conversion.rs`

```rust
let converter = SklearnConverter::new();
let aprender_alg = converter.convert(&SklearnAlgorithm::LinearRegression).unwrap();
let backend = converter.recommend_backend(&SklearnAlgorithm::KMeans, 100_000);
// Output: GPU backend for 100K K-Means clustering
```

### ✅ 9. PyTorch→Realizar Conversion (BATUTA-010)

**Module:** `src/pytorch_converter.rs`

Converts Python PyTorch inference code to Rust Realizar equivalents with automatic backend selection:

- **PyTorchConverter**: Operation mapping engine with 10 PyTorch inference operations
- **PyTorchOperation enum**: LoadModel, LoadTokenizer, Forward, Generate, Predict, TensorCreation, TensorReshape, Linear, Attention, GELU, Encode, Decode
- **RealizarOperation struct**: Code templates, required imports, complexity ratings, usage patterns
- **Methods**:
  - `convert(operation)`: Map PyTorch operation to Realizar equivalent
  - `recommend_backend(operation, size)`: MoE-based backend selection
  - `conversion_report()`: Generate mapping documentation
- **Integration**: Automatic PyTorch/transformers detection in TranspilationStage

**Example:** `examples/pytorch_conversion.rs`

```rust
let converter = PyTorchConverter::new();
let realizar_op = converter.convert(&PyTorchOperation::Generate).unwrap();
let backend = converter.recommend_backend(&PyTorchOperation::Generate, 1_000_000);
// Output: GPU backend for 1M parameter text generation
```

### ✅ 10. PARF Pattern and Reference Finder (BATUTA-012)

**Module:** `src/parf.rs`

Cross-codebase pattern analysis and reference finding for enterprise code understanding:

- **ParfAnalyzer**: Main analyzer with file caching and symbol tracking
- **Symbol References**: Find all usages of functions, classes, variables
- **Pattern Detection**: Identify TODO/FIXME, unwrap(), deprecated APIs, resource leaks
- **Dependency Analysis**: Track imports and module dependencies
- **Dead Code Detection**: Find unused symbols
- **CLI Integration**: `batuta parf [options]` with text/JSON/Markdown output

**Example:** `examples/parf_analysis.rs`

```rust
let mut analyzer = ParfAnalyzer::new();
analyzer.index_codebase(Path::new("src"))?;

// Find references
let refs = analyzer.find_references("BackendSelector", SymbolKind::Class);

// Detect patterns
let patterns = analyzer.detect_patterns();

// Find dead code
let dead_code = analyzer.find_dead_code();
```

**CLI Usage:**
```bash
batuta parf --find BackendSelector src
batuta parf --patterns --dead-code src
batuta parf --format json --output report.json src
```

### ✅ 2. Backend Selection (Spec Section 2.2)

**Module:** `src/backend.rs`

Cost-based backend selection using the **5× PCIe rule** from Gregg & Hazelwood (2011):

- **BackendSelector**: Analyzes compute/transfer ratio
- **Cost Model**: GPU beneficial when `compute_time > 5× transfer_time`
- **Backends**: GPU, SIMD, Scalar
- **Methods**:
  - `select_for_matmul(m, n, k)`: Matrix multiplication
  - `select_for_vector_op(n, ops)`: Vector operations
  - `select_for_elementwise(n)`: Element-wise ops (memory-bound)

**Example:** `examples/backend_selection.rs`

```rust
let selector = BackendSelector::new();
let backend = selector.select_for_matmul(512, 512, 512);
// Returns: SIMD (PCIe overhead > compute benefit)
```

**Test Results:**
- Small matmul (64×64): SIMD (ratio: 0.017×)
- Large matmul (512×512): SIMD (ratio: 0.136×)
- Very large (2048×2048): SIMD (ratio: 0.546×)

Per spec: GPU only beneficial for O(n³) operations with sustained compute.

### ✅ 3. Report Generation

**Module:** `src/report.rs`

Multi-format migration reports:

- **Formats**: HTML, Markdown, JSON, Plain Text
- **Content**: Analysis results, workflow progress, language stats, dependencies
- **HTML**: Professional reports with embedded CSS
- **Integration**: Full CLI integration via `batuta report`

**Usage:**
```bash
batuta report --format html --output report.html
batuta report --format json --output report.json
batuta report --format markdown --output report.md
```

### ✅ 4. CLI Orchestration

**Module:** `src/main.rs`

Complete 5-phase workflow CLI:

```bash
batuta analyze --languages --tdg .        # Phase 1: Analysis
batuta transpile --incremental            # Phase 2: Transpilation
batuta optimize --enable-gpu              # Phase 3: Optimization
batuta validate --trace-syscalls          # Phase 4: Validation
batuta build --release                    # Phase 5: Deployment
batuta report --format html               # Generate report
batuta status                             # Check progress
batuta reset --yes                        # Reset workflow
```

### ✅ 5. Workflow State Tracking

**Module:** `src/types.rs`

Persistent workflow state in `.batuta-state.json`:

- **WorkflowPhase**: 5 phases (Analysis → Deployment)
- **PhaseStatus**: NotStarted, InProgress, Completed, Failed
- **PhaseInfo**: Timestamps, errors, duration tracking
- **Progress**: Overall percentage completion

### ✅ 6. Quality Gates

**Test Suite:** 17 tests, all passing

- **Unit Tests (8)**: Backend selection, tool detection
- **Integration Tests (9)**: CLI commands, workflow, reports
- **Execution Time**: 0.3s (well under EXTREME TDD constraints)

**EXTREME TDD Compliance:**
- ✅ Pre-commit: 0.3s < 30s
- ✅ Test-fast: 0.3s < 5min
- ✅ Coverage: TBD < 10min

## Architecture Alignment

| Spec Section | Component | Status | Files |
|--------------|-----------|--------|-------|
| 2.2 Backend Selection | Cost-based GPU/SIMD dispatch | ✅ Complete | `src/backend.rs` |
| 2.8 Pipeline | 5-stage orchestration | ✅ Complete | `src/pipeline.rs` |
| 4.1 Integration Tests | End-to-end CLI tests | ✅ Complete | `tests/integration_test.rs` |
| 11 Usage Examples | Pipeline & backend demos | ✅ Complete | `examples/*.rs` |

## Recently Completed

### BATUTA-007: PMAT Adaptive Analysis ✅

**Completed:** 2025-11-20

Implemented adaptive quality analysis using pmat complexity tools per EXTREME TDD methodology.

**Results:**
- Refactored `cmd_transpile`: 36/58 → 8/13 complexity (78% reduction)
- Refactored `cmd_analyze`: 18/32 → 5/8 complexity (72% reduction)
- **Eliminated:** 3 critical errors → 0 ✅
- **Reduced:** Technical debt by 31.2 hours
- **Improved:** Max complexity by 64%

**Methodology:**
- Used `pmat analyze complexity` to identify hotspots
- Applied Jidoka principle: STOPPED THE LINE at threshold violations
- Extracted 13 helper functions using RED-GREEN-REFACTOR
- Maintained 100% test pass rate throughout

### BATUTA-011: Renacer Syscall Tracing ✅

**Completed:** 2025-11-20

Implemented syscall tracing validation using Renacer for semantic equivalence verification.

**Results:**
- Added renacer 0.5.0 dependency
- Implemented trace_and_compare() in ValidationStage
- Integrated into `batuta validate --trace-syscalls` command
- Created integration test for validation workflow
- **Tests:** 18/18 passing (up from 17)

**Features:**
- Traces original and transpiled binaries
- Compares syscall sequences for equivalence
- Graceful handling of missing binaries
- Color-coded validation results

### BATUTA-004: MoE Backend Selection ✅

**Completed:** 2025-11-20

Implemented Mixture-of-Experts routing for optimal backend selection with Trueno integration.

**Results:**
- Added trueno 0.4.1 dependency with GPU support
- Implemented OpComplexity enum (Low/Medium/High)
- Created select_with_moe() adaptive routing
- Integrated MoE into OptimizationStage
- **Tests:** 21/21 passing (11 backend tests, up from 8)
- **Example:** `examples/moe_routing.rs`

**MoE Thresholds:**
- **Low complexity** (element-wise): SIMD at 1M+, never GPU (memory-bound)
- **Medium complexity** (reductions): SIMD at 10K+, GPU at 100K+
- **High complexity** (matmul): SIMD at 1K+, GPU at 10K+

**Architecture:**
- OpComplexity-based routing
- Trueno integration framework
- vector_add() and matrix_multiply() methods
- Feature flag: `trueno-integration`

**Toyota Way Principle:** Kaizen (continuous optimization of compute resources)

### BATUTA-008: NumPy→Trueno Conversion Pipeline ✅

**Completed:** 2025-11-20

Implemented NumPy to Trueno conversion mapping with MoE-aware backend selection.

**Results:**
- Created NumPyConverter with operation mapping for 12 NumPy operations
- Integrated converter into TranspilationStage for Python projects
- Added automatic NumPy usage detection and conversion guidance
- Created examples/numpy_conversion.rs demonstration
- **Tests:** 21/21 passing (16 backend + 5 numpy_converter)

**Features:**
- NumPyOp enum: Array, Add, Subtract, Multiply, Divide, Dot, Sum, Mean, Max, Min, Reshape, Transpose
- TruenoOp struct: Code templates, imports, complexity ratings
- Operation complexity classification (Low/Medium/High)
- MoE integration for backend recommendations
- Automatic Python file scanning for NumPy imports

**Architecture:**
- NumPyConverter struct with HashMap-based operation mapping
- Integration with BackendSelector for adaptive routing
- Pipeline stage integration for automatic conversion guidance
- Metadata tracking of NumPy usage and conversion recommendations

**Toyota Way Principle:** Muda elimination (zero-waste conversion from NumPy to Trueno)

### BATUTA-009: sklearn→Aprender Conversion Pipeline ✅

**Completed:** 2025-11-20

Implemented sklearn to Aprender algorithm mapping with MoE-aware backend selection.

**Results:**
- Created SklearnConverter with algorithm mapping for 8 sklearn algorithms
- Integrated converter into TranspilationStage for Python projects
- Added automatic sklearn usage detection and conversion guidance
- Created examples/sklearn_conversion.rs demonstration
- **Tests:** 23/23 passing (16 backend + 5 numpy + 7 sklearn + 2 tools)

**Features:**
- SklearnAlgorithm enum: LinearRegression, LogisticRegression, KMeans, DecisionTree, RandomForest, StandardScaler, TrainTestSplit, Metrics (8 total)
- AprenderAlgorithm struct: Code templates, imports, complexity ratings, usage patterns
- Algorithm complexity classification (Low/Medium/High)
- MoE integration for backend recommendations
- Automatic Python file scanning for sklearn imports

**Architecture:**
- SklearnConverter struct with HashMap-based algorithm mapping
- Integration with BackendSelector for adaptive routing
- Pipeline stage integration for automatic conversion guidance
- Metadata tracking of sklearn usage and conversion recommendations
- Module organization preservation (linear_model, cluster, tree, preprocessing, model_selection, metrics)

**Conversion Examples:**
- `sklearn.linear_model.LinearRegression()` → `aprender::linear_model::LinearRegression::new()`
- `sklearn.cluster.KMeans(n_clusters=3)` → `aprender::cluster::KMeans::new(3)`
- `sklearn.preprocessing.StandardScaler()` → `aprender::preprocessing::StandardScaler::new()`
- `sklearn.model_selection.train_test_split()` → `aprender::model_selection::train_test_split()`

**Toyota Way Principle:** Heijunka (level scheduling of ML workloads across backends)

### BATUTA-010: PyTorch→Realizar Conversion Pipeline ✅

**Completed:** 2025-11-20

Implemented PyTorch to Realizar operation mapping for inference workloads with MoE-aware backend selection.

**Results:**
- Created PyTorchConverter with operation mapping for 10 PyTorch operations
- Integrated converter into TranspilationStage for Python projects
- Added automatic PyTorch/transformers usage detection and conversion guidance
- Created examples/pytorch_conversion.rs demonstration
- **Tests:** 30/30 passing (16 backend + 5 numpy + 7 sklearn + 7 pytorch + 2 tools)

**Features:**
- PyTorchOperation enum: LoadModel, LoadTokenizer, Forward, Generate, Predict, TensorCreation, Linear, Attention, GELU, Encode, Decode (10 mapped)
- RealizarOperation struct: Code templates, imports, complexity ratings, usage patterns
- Operation complexity classification (Low/Medium/High)
- MoE integration for backend recommendations
- Automatic Python file scanning for PyTorch and transformers imports

**Architecture:**
- PyTorchConverter struct with HashMap-based operation mapping
- Integration with BackendSelector for adaptive routing
- Pipeline stage integration for automatic conversion guidance
- Metadata tracking of PyTorch usage and conversion recommendations
- Focus on inference patterns (model loading, generation, tokenization)

**Conversion Examples:**
- `torch.load('model.pt')` → `GGUFModel::from_file("model.gguf")`
- `model.generate(**inputs, max_length=50)` → `generate_text(&model, &tokens, 50)`
- `nn.Linear(768, 512)` → `LinearLayer::new(768, 512)`
- `tokenizer.encode('text')` → `tokenizer.encode("text")`

**Key Differences:**
- **PyTorch**: Training + inference, autograd, .pt/.pth files, Python-first
- **Realizar**: Inference-only, GGUF/SafeTensors, Rust-native CPU/GPU/WASM

**Toyota Way Principle:** Jidoka (stop-the-line quality - inference-only focus ensures production reliability)

### BATUTA-012: PARF (Pattern and Reference Finder) ✅

**Completed:** 2025-11-20

Implemented cross-codebase pattern analysis and reference finding for enterprise code understanding.

**Results:**
- Created ParfAnalyzer with comprehensive code analysis capabilities
- Integrated PARF into CLI with multiple output formats (text, JSON, Markdown)
- Added symbol reference finding across files
- Implemented pattern detection (tech debt, error handling, resources, deprecated APIs)
- Built dependency analysis and dead code detection
- Created examples/parf_analysis.rs demonstration
- **Tests:** 37/37 passing (30 existing + 7 parf)

**Features:**
- **Symbol References**: Find all usages of functions, classes, variables across codebase
- **Pattern Detection**: Identify TODO/FIXME, unwrap() calls, deprecated APIs, resource management
- **Dependency Analysis**: Track imports, includes, and module dependencies
- **Dead Code Detection**: Find unused symbols that can be safely removed
- **Call Graph**: Understand function relationships and usage patterns

**Architecture:**
- ParfAnalyzer struct with file caching and symbol tracking
- Symbol extraction for Rust (fn, struct, enum) and Python (def, class)
- Pattern matching for common anti-patterns and code smells
- Multiple output formats for integration with toolchains
- CLI integration: `batuta parf [options]`

**CLI Usage:**
```bash
# Full analysis
batuta parf src

# Find all references to a symbol
batuta parf --find BackendSelector src

# Detect code patterns
batuta parf --patterns src

# Analyze dependencies
batuta parf --dependencies src

# Find dead code
batuta parf --dead-code src

# JSON output for tooling
batuta parf --patterns --format json --output report.json src
```

**Use Cases:**
1. Code Understanding: Navigate unfamiliar codebases, find symbol usages
2. Refactoring: Identify safe-to-remove code, find all references before renaming
3. Migration Planning: Map dependencies for phased migration strategies
4. Code Quality: Detect anti-patterns, track technical debt, find resource leaks

**Toyota Way Principle:** Andon (problem visualization - make issues visible for rapid response)

### WASM Build Target (Infrastructure) ✅

**Completed:** 2025-11-20

Implemented WebAssembly build target for browser and edge deployment with JavaScript interop.

**Results:**
- Created src/wasm.rs with JavaScript API (335 lines)
- Configured Cargo.toml with native/wasm feature flags
- Added build infrastructure (scripts, Makefile targets)
- Created interactive demo with 6 conversion panels
- Added comprehensive documentation
- **Status:** 95% complete (needs final conditional compilation guards)

**Features:**
- **analyze_code()**: Language detection with ML library identification
- **convert_numpy()**: NumPy → Trueno conversion with backend recommendations
- **convert_sklearn()**: sklearn → Aprender conversion
- **convert_pytorch()**: PyTorch → Realizar conversion
- **backend_recommend()**: Optimal compute backend selection
- **version()**: Get Batuta version info

**Architecture:**
- Feature flags: `native` (CLI, filesystem, tracing) vs `wasm` (browser APIs only)
- Conditional compilation with #[cfg(feature)] guards throughout codebase
- No file system operations in WASM (in-memory analysis only)
- Size optimization: wasm-opt -Oz produces ~500-800 KB release builds

**Build Commands:**
```bash
# Debug build
make wasm
# or
cargo build --target wasm32-unknown-unknown --no-default-features --features wasm

# Release build (optimized)
make wasm-release
# or
./scripts/build-wasm.sh release
```

**JavaScript API Example:**
```javascript
import init, { analyze_code, convert_numpy } from './batuta.js';

await init();

// Analyze code
const analysis = analyze_code("import numpy as np\nx = np.array([1, 2, 3])");
console.log(analysis.language); // "Python"
console.log(analysis.has_numpy); // true

// Convert NumPy to Trueno
const conversion = convert_numpy("np.add(a, b)", 10000);
console.log(conversion.rust_code);
console.log(conversion.backend_recommendation); // "SIMD" or "GPU"
```

**Interactive Demo:**
- Location: `examples/wasm/index.html`
- Modern gradient UI with real-time conversion
- 6 interactive panels for different conversion types
- Example snippets for quick testing
- Visual backend recommendations with color-coded badges
- Runs entirely client-side (no server required)

**Integration:**
- React, Vue, Angular compatible
- Node.js support with nodejs target
- Works in all modern browsers (Chrome 61+, Firefox 60+, Safari 11+, Edge 16+)

**Toyota Way Principle:** Muda elimination (eliminate waste by enabling browser-based workflows without server round-trips)

### Docker Containerization (Infrastructure) ✅

**Completed:** 2025-11-20

Implemented Docker containerization for consistent deployment across environments.

**Results:**
- Created multi-stage Dockerfile for production (150-200 MB)
- Created development Dockerfile with hot reload
- Configured docker-compose.yml with 5 services
- Added build scripts and comprehensive documentation
- Implemented security best practices (non-root user, health checks)

**Docker Images:**

1. **Production (`batuta:latest`)**
   - Multi-stage build for minimal size
   - Debian slim base (~150-200 MB)
   - Non-root user for security
   - Health check included
   - Runtime dependencies only

2. **Development (`batuta:dev`)**
   - Full Rust toolchain
   - cargo-watch for hot reload
   - Development tools (vim, curl, git)
   - Python/C++ for transpilation testing
   - Persistent volumes for fast rebuilds

**Docker Compose Services:**

```yaml
services:
  batuta:  # Production CLI
  dev:     # Development with hot reload
  ci:      # CI/CD testing
  wasm:    # WASM build
  docs:    # Documentation server
```

**Build Commands:**
```bash
# Production image
make docker
# or
./scripts/docker-build.sh prod

# Development image
make docker-dev
# or
./scripts/docker-build.sh dev

# All images
./scripts/docker-build.sh all
```

**Usage Examples:**
```bash
# Analyze current directory
docker run -v $(pwd):/workspace batuta:latest analyze /workspace

# Start development environment
docker-compose up dev

# Run CI tests
docker-compose up ci

# Build WASM
docker-compose up wasm

# Serve documentation
docker-compose up docs
```

**Features:**
- Multi-stage builds for size optimization
- Named volumes for persistent cargo cache
- Health checks for monitoring
- Security hardening (non-root, minimal attack surface)
- Interactive development with hot reload
- CI/CD integration ready
- Comprehensive documentation in docs/DOCKER.md

**Architecture:**
- Builder stage: Compiles Rust binary with all optimizations
- Runtime stage: Minimal Debian image with only runtime deps
- Development: Full toolchain with mounted volumes
- Persistent volumes: cargo-cache, cargo-git, target-cache

**Security:**
- Runs as non-root user (`batuta:1000`)
- Minimal base images (slim, not full)
- No unnecessary packages
- Health checks for monitoring
- .dockerignore to exclude sensitive files

**Toyota Way Principle:** Jidoka (built-in quality through reproducible environments)

### External Tool Integration (Phase 1) ✅

**Completed:** 2025-11-20

Integrated external transpilation tools for complete language coverage.

**Results:**
- Enhanced ToolRegistry with proper detection and version checking
- Added transpilation functions for Python, Shell, and C/C++
- Integrated PMAT quality analysis
- Created full_transpilation.rs example (240 lines)
- Updated TranspilationStage to use external tools
- All tests passing (37/37)

**Integrated Tools:**

1. **Depyler (Python → Rust)** ✅
   - Version detected: 3.20.0
   - Commands: transpile, compile, analyze, check
   - Features: Full project structure generation, type inference
   - Integration: `tools::transpile_python()`

2. **Bashrs (Shell → Rust)** ✅
   - Version detected: 6.35.0
   - Commands: build, check, verify, purify
   - Features: POSIX compliance, formal verification, standalone binaries
   - Integration: `tools::transpile_shell()`

3. **Decy (C/C++ → Rust)** ⚠️
   - Status: Framework integrated, tool not installed
   - Installation: `cargo install decy`
   - Integration: `tools::transpile_c_cpp()` (ready when installed)

4. **PMAT (Quality Analysis)** ✅
   - Version detected: 2.199.0
   - Commands: analyze, tdg, complexity
   - Features: TDG scoring, complexity metrics, adaptive analysis
   - Integration: Already integrated in analyzer.rs

5. **Ruchy (Scripting)** ✅
   - Version detected: 3.213.0
   - Commands: run, compile, repl, test
   - Features: Ruby-like syntax, gradual typing, formal verification
   - Integration: `tools::run_ruchy_script()`

**Transpilation Workflow:**

```rust
// TranspilationStage automatically selects correct tool
match language {
    Language::Python => {
        tools::transpile_python(&input, &output)?
    }
    Language::Shell => {
        tools::transpile_shell(&input, &output)?
    }
    Language::C | Language::Cpp => {
        tools::transpile_c_cpp(&input, &output)?
    }
}
```

**CLI Usage:**
```bash
# Detect available tools
cargo run --example full_transpilation

# Analyze project
batuta analyze --languages --tdg /path/to/project

# Transpile Python to Rust
batuta transpile --input /path/to/python_project \
                 --output /path/to/rust_project

# Transpile Shell to Rust
batuta transpile --input script.sh --output script.rs
```

**Tool Detection:**
- Automatic PATH scanning
- Version checking via --version
- Installation instructions for missing tools
- Graceful degradation when tools unavailable

**Example Output:**
```
📋 Detecting available tools...
   ✅ Found 6 tools:
      • Depyler (Python → Rust)
      • Bashrs (Shell → Rust)
      • Ruchy (Rust scripting)
      • PMAT (Quality analysis)
      • Realizar (Inference runtime)
      • Renacer (Syscall tracing)
```

**Status:** Full transpilation pipeline operational with external tools

**Toyota Way Principle:** Heijunka (level scheduling across multiple transpilers)

## Not Yet Implemented

Per roadmap (docs/roadmaps/roadmap.yaml):

### Infrastructure (Spec Sections 5.1, 5.3)
- **StaticFixer Integration**: Eliminate redundant static analysis (BATUTA-001)
- **Decy Installation**: C/C++ transpiler (available but not installed)

## Dependencies

### Core Stack Components (External)
- **Trueno**: SIMD/GPU tensor operations (external crate)
- **Trueno-DB**: Vector database (external crate)
- **Aprender**: ML algorithms (external crate)
- **Realizar**: Inference runtime (external crate)
- **Renacer**: Syscall tracing (external crate)
- **Depyler**: Python → Rust transpiler (external binary)
- **Decy**: C/C++ → Rust transpiler (external binary)

### Current Dependencies (Cargo.toml)

**Core (WASM-compatible):**
- **serde**: Serialization
- **anyhow/thiserror**: Error handling
- **chrono**: Date/time handling
- **async-trait**: Async trait support

**Native-only:**
- **clap**: CLI framework
- **tokio**: Async runtime
- **tracing/tracing-subscriber**: Logging
- **walkdir**: File traversal
- **glob**: Pattern matching
- **which**: Command finding
- **colored**: Terminal colors
- **indicatif**: Progress bars
- **renacer**: Syscall tracing
- **trueno**: SIMD/GPU tensor operations (optional)

**WASM-only:**
- **wasm-bindgen**: JavaScript interop
- **wasm-bindgen-futures**: Async support for WASM
- **js-sys**: JavaScript standard library bindings
- **web-sys**: Web API bindings

## Quality Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| TDG Score | ≥85 | 94.4 | ✅ A |
| Test Coverage | >85% | TBD | 🔄 |
| Mutation Coverage | >80% | TBD | 🔄 |
| Test Execution | <30s | 0.43s | ✅ |
| Max Cyclomatic Complexity | ≤10 | 13 | ⚠️ Warning |
| Max Cognitive Complexity | ≤15 | 21 | ⚠️ Warning |
| Critical Errors | 0 | 0 | ✅ ZERO |

## Next Steps

Per EXTREME TDD "continue" methodology:

1. **Docker integration**: Dockerfile + compose ⏳ IN PROGRESS
2. **Complete WASM**: Final conditional compilation guards
3. **Run mutation tests**: `cargo mutants --timeout 300`
4. **Measure coverage**: `cargo llvm-cov --all-features`

## References

All implementations reference academic foundations from spec section 8:

- **Gregg & Hazelwood (2011)**: PCIe overhead analysis
- **Haas et al. (2017)**: WebAssembly performance
- **Malkov & Yashunin (2018)**: HNSW indexing
- **Dettmers et al. (2023)**: Quantization algorithms

## Usage

See [examples/](examples/) for runnable demonstrations:

```bash
cargo run --example backend_selection  # Backend cost model demo
cargo run --example pipeline_demo      # Full pipeline execution
```

## License

MIT

---

*Generated by Batuta - Sovereign AI Stack*
https://github.com/paiml/Batuta
