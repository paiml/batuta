# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2025-12-06

### Changed
- Updated pforge from v0.1.2 to v0.1.4 (pmcp 1.8.6 compatibility fix)
- Updated renacer from v0.6.5 to v0.7.0 in knowledge graph

## [0.1.2] - 2025-12-05

### Added

#### Content Creation Tooling (Spec v1.1.0)
- **Content Module** - LLM prompt generation for educational content
  - 5 content types: HLO (High-Level Outline), DLO (Detailed Outline), BCH (Book Chapter), BLP (Blog Post), PDM (Presentar Demo)
  - Token budgeting (Heijunka) for Claude 200K, Gemini 1M, GPT-4 128K context windows
  - Source context integration (Genchi Genbutsu) for grounded content
  - Content validation (Jidoka) with quality gates: meta-commentary detection, code block language validation, heading hierarchy checks
  - PromptEmitter with Toyota Way constraints embedded
  - 44 comprehensive tests (~93% coverage)

- **CLI Commands** for content creation
  - `batuta content types` - List all content types with output formats
  - `batuta content emit --type <TYPE>` - Generate structured prompts
  - `batuta content validate --type <TYPE> <file>` - Validate content against quality gates

#### Visualization Frameworks Integration
- **Viz Module** - Data visualization framework comparison tree
  - Support for Matplotlib, Seaborn, Plotly, Bokeh, Altair, D3.js, Vega-Lite
  - PAIML replacements: Trueno-Viz, Presentar
  - CLI command: `batuta viz tree`

#### Experiment Tracking Enhancements
- **Experiment Tree** - Framework comparison visualization
  - MLflow, Weights & Biases, Neptune, Comet ML, Sacred, DVC comparison
  - PAIML replacement mapping (Entrenar integration)
  - CLI command: `batuta experiment tree`

- **Entrenar v1.8.0 Integration** - Full experiment tracking spec
  - Pre-registration support for research reproducibility
  - Cost-performance benchmarking with sovereign hardware metrics
  - Research artifact management with citation metadata
  - Apple Silicon, NVIDIA, AMD, TPU compute device support

#### Examples
- **content_demo.rs** - Comprehensive demo of content creation tooling

### Changed
- Refactored experiment module into multi-file structure (`src/experiment/mod.rs`, `src/experiment/tree.rs`)
- Updated clippy compliance: `DVC` → `Dvc`, `NAS` → `Nas` for acronym conventions
- Improved `TokenBudget::available_margin()` to use `saturating_sub` for safe arithmetic

### Fixed
- Fixed `FromStr` trait implementation for `ContentType` (was standalone method)
- Fixed unused import warnings in content module

## [0.1.1] - 2025-11-28

### Added
- Model Serving Ecosystem integration (native feature flag)
- Data Platforms Integration module

## [0.1.0] - 2025-01-21

### Added

#### Core Features
- **Analysis Phase** - Complete project analysis with language detection, dependency analysis, and TDG scoring
  - Multi-language detection (Python, Rust, C/C++, Shell)
  - Dependency manager detection (pip, Cargo, npm, poetry, conda)
  - Technical Debt Grade (TDG) calculation
  - ML framework detection (NumPy, sklearn, PyTorch, transformers)
  - Transpiler recommendations based on detected languages

#### Conversion Framework
- **NumPy → Trueno Converter** - Convert NumPy operations to Trueno multi-target compute
  - 12 operation types supported (array, add, subtract, multiply, dot, sum, mean, etc.)
  - Automatic backend selection (Scalar, SIMD, GPU) via MoE routing
  - 34 comprehensive tests (~90% coverage)

- **sklearn → Aprender Converter** - Convert scikit-learn to Rust first-principles ML
  - 21 algorithm types (LinearRegression, KMeans, DecisionTree, RandomForest, etc.)
  - All sklearn modules supported (linear_model, cluster, tree, ensemble, preprocessing, metrics, model_selection)
  - 44 comprehensive tests (~90% coverage)

- **PyTorch → Realizar Converter** - Convert PyTorch inference to Rust ML runtime
  - 20+ operation types (load_model, forward, generate, attention, etc.)
  - GGUF/SafeTensors model loading support
  - 39 comprehensive tests (~90% coverage)

#### Infrastructure
- **Backend Selection** - MoE routing for optimal compute backend selection
  - Complexity-based routing (Low/Medium/High)
  - Data size-aware selection
  - Support for Scalar, SIMD, and GPU backends
  - 40 comprehensive tests (~95% coverage)

- **Plugin Architecture** - Extensible plugin system for custom transpilers
  - TranspilerPlugin trait for custom implementations
  - PluginRegistry for plugin management
  - PluginStage for pipeline integration
  - 31 comprehensive tests (~90% coverage)

- **PARF Analyzer** - Pattern analysis and reference finding
  - Symbol definition indexing
  - Reference finding across codebases
  - Pattern detection (tech debt, deprecated APIs, error handling)
  - Dependency analysis
  - Dead code detection
  - 43 comprehensive tests (~90% coverage)

- **Report Generation** - Multi-format migration reports
  - HTML reports with rich formatting
  - Markdown reports for documentation
  - JSON reports for CI/CD integration
  - Text reports for console output
  - 39 comprehensive tests (~95% coverage)

- **5-Phase Workflow** - Kanban-style workflow management
  - Analysis → Transpilation → Optimization → Validation → Deployment
  - State persistence and progress tracking
  - Phase-level status management
  - 46 comprehensive tests (~95% coverage)

- **Tool Registry** - Detection and management of Pragmatic AI Labs tools
  - Automatic tool detection (decy, depyler, bashrs, ruchy, pmat, trueno, aprender, realizar, renacer)
  - Installation instruction generation
  - Transpiler selection based on language
  - 37 comprehensive tests (~95% coverage)

#### WASM Support
- **WebAssembly Interface** - Browser-based code analysis and conversion
  - AnalysisResult for language detection
  - ConversionResult for code transformation
  - JSON serialization for JS interop
  - 6 native-compatible tests

#### Documentation
- **The Batuta Book** - Comprehensive mdBook documentation
  - Philosophy and core principles
  - 5-phase workflow guide
  - Tool ecosystem deep-dives
  - Practical examples
  - Configuration reference

- **Coverage Report** - Detailed test coverage documentation
- **Implementation Guide** - Quality metrics and validation gates
- **README** - Quick start and feature overview

### Quality & Testing
- **529 Total Tests** (487 unit + 36 integration + 6 benchmarks)
- **90%+ Coverage** on all core modules
- **100% Pass Rate** - Zero defects
- **Systematic Test Patterns** - Comprehensive validation framework
- **CI/CD Pipeline** - Automated testing and validation
- **Pre-commit Hooks** - Quality gates (<30s execution)
- **Certeza Integration** - Zero defects tolerance validation

### Performance
- **Sub-nanosecond Backend Selection** - Optimized MoE routing
- **Optimized Release Builds** - LTO, codegen-units=1, strip=true
- **Benchmark Suite** - Performance regression testing

### Technical Debt & Quality
- **TDG Score: 92.6/100 (A)** - Excellent code quality
- **Security Audits** - 0 vulnerabilities
- **Mutation Testing** - 100% on converters

## [Unreleased]

### Planned Features
- Transpilation engine implementation
- Optimization phase with Trueno SIMD/GPU
- Validation phase with Renacer syscall tracing
- Build phase with cross-compilation
- Additional converter implementations (Ridge, Lasso, DBSCAN, etc.)
- Enhanced WASM testing with wasm32 target

---

[0.1.0]: https://github.com/paiml/Batuta/releases/tag/v0.1.0
