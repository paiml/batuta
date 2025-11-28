# Introduction

> **"Batuta orchestrates the conversion of ANY project to modern Rust - not through magic, but through systematic application of proven manufacturing principles to code migration."**

## Welcome to The Batuta Book

This book is your comprehensive guide to **Batuta**, the orchestration framework that transforms legacy codebases (Python, C/C++, Shell scripts) into modern, high-performance Rust applications. Unlike simple transpilers, Batuta provides a **complete 5-phase workflow** that ensures semantic preservation, automatic optimization, and validation of equivalence.

## What is Batuta?

Batuta (Spanish for "conductor's baton") orchestrates the **20-component Sovereign AI Stack** from Pragmatic AI Labs to convert, optimize, and validate code migrations:

![Sovereign AI Stack](./assets/sovereign-stack.svg)

### Layer 0: Compute Primitives
- **[Trueno](https://github.com/paiml/trueno)** v0.7.3 - SIMD/GPU compute primitives
- **[Trueno-DB](https://github.com/paiml/trueno-db)** v0.3.3 - Vector database with HNSW
- **[Trueno-Graph](https://github.com/paiml/trueno-graph)** v0.1.1 - Graph analytics
- **[Trueno-Viz](https://github.com/paiml/trueno-viz)** - SIMD/GPU/WASM visualization
- **[Trueno-RAG](https://github.com/paiml/trueno-rag)** - RAG pipeline (chunking, retrieval, reranking)

### Layer 1: ML Algorithms
- **[Aprender](https://github.com/paiml/aprender)** v0.12.0 - First-principles ML in pure Rust

### Layer 2: Training & Inference
- **[Entrenar](https://github.com/paiml/entrenar)** v0.2.0 - Training with autograd, LoRA, quantization
- **[Realizar](https://github.com/paiml/realizar)** v0.2.1 - LLM inference (GGUF, safetensors, transformers)

### Layer 3: Transpilers
- **[Depyler](https://github.com/paiml/depyler)** - Python → Rust with type inference
- **[Decy](https://github.com/paiml/decy)** - C/C++ → Rust with ownership inference
- **[Bashrs](https://github.com/paiml/bashrs)** v6.41.0 - Rust → Shell (bootstrap scripts)
- **[Ruchy](https://github.com/paiml/ruchy)** v3.213.0 - Script → Rust (systems scripting)

### Layer 4: Orchestration
- **[Batuta](https://github.com/paiml/batuta)** - This framework (5-phase workflow)
- **[Repartir](https://github.com/paiml/repartir)** v1.0.0 - Distributed computing
- **[pforge](https://github.com/paiml/pforge)** v0.1.2 - MCP server framework (rust-mcp-sdk)

### Layer 5: Quality
- **[Certeza](https://github.com/paiml/certeza)** - Quality validation framework
- **[PMAT](https://github.com/paiml/paiml-mcp-agent-toolkit)** v2.205.0 - AI context & code quality
- **[Renacer](https://github.com/paiml/renacer)** v0.6.5 - Syscall tracing & golden traces

### Layer 6: Data & MLOps
- **[Alimentar](https://github.com/paiml/alimentar)** - Data loading with .ald encryption
- **[Pacha](https://github.com/paiml/pacha)** - Model, Data and Recipe Registry

## The Philosophy

Batuta is built on three core principles:

### 1. **Toyota Way Manufacturing**

We apply Lean Manufacturing principles to code migration:

- **Muda** (Waste Elimination) - No redundant analysis or compilation
- **Jidoka** (Built-in Quality) - Phase dependencies enforce correctness
- **Kaizen** (Continuous Improvement) - Iterative optimization
- **Heijunka** (Level Scheduling) - Balanced tool orchestration
- **Kanban** (Visual Workflow) - Clear progress visualization
- **Andon** (Problem Visualization) - Immediate error feedback

### 2. **Semantic Preservation**

**Code migration is NOT a lossy transformation.** Batuta uses:
- IR-based analysis to preserve program semantics
- Syscall tracing (Renacer) to verify runtime equivalence
- Golden trace comparison for deterministic validation
- Output comparison and benchmarking

### 3. **First Principles Thinking**

Rather than blindly translating code, Batuta rebuilds from fundamental truths:
- What does this code *actually do*?
- What is the minimal correct implementation?
- How can we express this idiomatically in Rust?

## The 5-Phase Workflow

Batuta follows a strict **5-phase Kanban workflow**:

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐    ┌────────────┐
│ Analysis │ -> │ Transpilation│ -> │ Optimization │ -> │ Validation│ -> │ Deployment │
└──────────┘    └──────────────┘    └──────────────┘    └───────────┘    └────────────┘
    20%              40%                  60%               80%               100%
```

Each phase has:
- **Clear entry criteria** (dependencies on previous phase)
- **Specific deliverables** (outputs that feed next phase)
- **Quality gates** (validation before proceeding)
- **Automated tracking** (state persistence and progress)

## Sovereign AI: Complete Stack

The Sovereign AI Stack is **100% Rust, no Python/C++ dependencies**:

| Capability | Component | Replaces |
|------------|-----------|----------|
| Tensor ops | Trueno | NumPy |
| Vector DB | Trueno-DB | Pinecone, Milvus |
| ML algorithms | Aprender | scikit-learn |
| Training | Entrenar | PyTorch training |
| Inference | Realizar | vLLM, TensorRT |
| RAG | Trueno-RAG | LangChain, LlamaIndex |
| Data loading | Alimentar | pandas |

**Key differentiators:**
- Pure Rust = WASM, embedded, Lambda deployment
- .apr/.ald formats with AES-256-GCM encryption
- No GIL = true parallelism
- 9.6x faster inference than PyTorch (benchmarked)

## Who is This Book For?

This book is for:

- **Legacy codebase maintainers** drowning in Python/C/C++ technical debt
- **Performance engineers** seeking ML inference speedups (10-100x)
- **Systems programmers** modernizing shell-based infrastructure
- **Engineering managers** planning strategic rewrites
- **AI/ML engineers** building sovereign, private AI systems

## What You'll Learn

By the end of this book, you will:

1. **Understand the philosophy** behind systematic code migration
2. **Master the 5-phase workflow** from analysis to deployment
3. **Use all 20 components** effectively in orchestration
4. **Apply Toyota Way principles** to your migration strategy
5. **Validate semantic equivalence** through syscall tracing
6. **Optimize performance** with SIMD/GPU acceleration
7. **Build RAG pipelines** with Trueno-RAG
8. **Deploy LLM inference** with Realizar (GGUF, safetensors)

## Prerequisites

**Required:**
- Basic understanding of Rust (ownership, lifetimes, traits)
- Familiarity with at least one source language (Python, C, C++, Shell)
- Command-line proficiency

**Helpful but not required:**
- Experience with build systems (Cargo, Make, CMake)
- Understanding of ML frameworks (NumPy, PyTorch, scikit-learn)
- Systems programming background

## How to Read This Book

**If you're brand new to Batuta:**
Read **Part I (Core Philosophy)** to understand the "why", then work through **Part II (5-Phase Workflow)** hands-on with a small example project.

**If you're experienced with transpilers:**
Start with **Part III (Tool Ecosystem)** to understand Batuta's orchestration capabilities, then dive into **Part IV (Practical Examples)** for real-world patterns.

**If you're migrating a specific project:**
Begin with **Part II (5-Phase Workflow)** for the systematic approach, consult **Part V (Configuration)** for customization, and keep **Part VIII (Troubleshooting)** handy.

**If you're building AI/ML systems:**
Focus on **Part III (Tool Ecosystem)** for Trueno/Aprender/Realizar integration, then **Oracle Mode** for intelligent stack queries.

## Oracle Mode

Batuta includes **Oracle Mode** - an intelligent query interface for the Sovereign AI Stack:

```bash
# Ask natural language questions
batuta oracle "How do I train a model on GPU?"

# Get component recommendations
batuta oracle "What's best for vector similarity search?"

# Query capabilities
batuta oracle "Which components support WASM?"
```

Oracle Mode knows all 20 components, their capabilities, and integration patterns.

## Code Examples

All code examples in this book are:
- **Tested and verified** on Rust 1.75+
- **Available in the repository** under `examples/`
- **Self-contained** with full context
- **Annotated** with explanatory comments

## Conventions

Throughout this book:

- **Bold text** emphasizes key concepts
- `Inline code` represents commands, code snippets, or file names
- 💡 Tips provide helpful shortcuts
- ⚠️ Warnings highlight potential pitfalls
- 🎯 Best practices recommend proven approaches

## Community and Support

- **GitHub**: [paiml/Batuta](https://github.com/paiml/Batuta)
- **Book**: [paiml.github.io/batuta](https://paiml.github.io/batuta/)
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences

## Let's Begin

The journey from legacy code to modern Rust is challenging but immensely rewarding. With Batuta orchestrating the 20-component Sovereign AI Stack, you're equipped with:

- **5 compute primitives** (Trueno family)
- **3 ML components** (Aprender, Entrenar, Realizar)
- **4 transpilers** (Depyler, Decy, Bashrs, Ruchy)
- **3 orchestration tools** (Batuta, Repartir, pforge)
- **3 quality tools** (Certeza, PMAT, Renacer)
- **2 data tools** (Alimentar, Pacha)

**Welcome to systematic code migration. Let's conduct this orchestra.** 🎵

---

**Next:** [Part I: Core Philosophy](./part1/orchestration-paradigm.md)
