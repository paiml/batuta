# Batuta Specification Overview

**Version:** 2.5.0
**Date:** 2026-04-04
**Status:** Active

---

## 1. What is Batuta?

Batuta is the orchestration framework for the **Sovereign AI Stack** -- a vertically integrated, pure-Rust ecosystem for privacy-preserving ML infrastructure. It coordinates stack components (trueno, aprender, realizar, entrenar, repartir, pacha) and provides transpilation pipelines for converting Python/C/Shell to Rust.

**Core mission:** Replace Python/CUDA/cloud dependencies with a self-contained Rust stack where data provenance, computation residency, and determinism are architectural guarantees -- not configuration options.

---

## 2. Stack Architecture

```
+-------------------------------------------------------------+
|                    batuta (Orchestration)                    |
+-------------------------------------------------------------+
| whisper-apr (ASR) | realizar (Inference)  | pacha (Registry) |
+-------------------+-----------------------+------------------+
|  aprender (ML)    | entrenar (Training)   | jugar (Games)    |
+-------------------+-----------------------+------------------+
|  simular (Simulation)     | profesor (Education)             |
+---------------------------+----------------------------------+
|               repartir (Distributed Compute)                 |
|          CPU (Rayon) | GPU (wgpu) | Remote (TCP/TLS)         |
+-------------------------------------------------------------+
| trueno-zram (Compression) | trueno-ublk (Block Device)       |
+---------------------------+----------------------------------+
|             trueno (SIMD/GPU Compute Primitives)             |
|       AVX2/AVX-512/NEON | wgpu | LZ4/ZSTD compression       |
+-------------------------------------------------------------+
```

### Layer Summary

| Layer | Crates | Purpose |
|-------|--------|---------|
| **Compute** | trueno, trueno-db, trueno-graph, trueno-rag, trueno-viz | SIMD/GPU primitives, vector DB, graph analytics, RAG, visualization |
| **Compression** | trueno-zram-core, trueno-ublk | SIMD/GPU memory compression, block device |
| **Distribution** | repartir | CPU/GPU/Remote work-stealing executors |
| **ML** | aprender, alimentar | Algorithms, APR v2 format, data loading |
| **Training** | entrenar | Autograd, LoRA/QLoRA, quantization, model merge |
| **Inference** | realizar, whisper-apr | GGUF/APR/SafeTensors inference, ASR |
| **Simulation** | simular, jugar | Monte Carlo, physics, game engine |
| **Education** | profesor | Courses, quizzes, labs (WASM) |
| **Registry** | pacha | Model registry with Ed25519 signatures |
| **Tracing** | renacer | Syscall tracing with source correlation |
| **Transpilers** | depyler, bashrs, decy | Python/Shell/C to Rust |
| **Quality** | pmat, certeza, apr-qa | Static analysis, quality gates, model QA |
| **Orchestration** | batuta | Stack coordination, CLI, agent runtime |

---

## 3. Core Modules

### 3.1 Pipeline (`src/pipeline.rs`)

5-phase transpilation: Analysis -> Transpilation -> Optimization -> Validation -> Build. Uses Jidoka stop-on-error at each phase boundary.

### 3.2 Backend (`src/backend.rs`)

Cost-based GPU/SIMD/Scalar selection using the 5x PCIe rule: dispatch to GPU only when `compute_time > 5 * transfer_time` (Gregg & Hazelwood, 2011).

### 3.3 Oracle (`src/oracle/`)

Knowledge graph for stack component recommendations. Supports natural language queries, RAG-indexed documentation search (SQLite+FTS5), and PMAT function-level code search. 34 cookbook recipes with TDD test companions.

### 3.4 Serve (`src/serve/`)

Model serving with failover, circuit breakers, and privacy tiers (Sovereign/Private/Standard). SpilloverRouter for local-first with remote fallback.

### 3.5 Stack (`src/stack/`)

Dependency graph management, coordinated release orchestration, quality gates across stack components. Publish-status caching with hash-based invalidation.

### 3.6 Agent (`src/agent/`)

Autonomous perceive-reason-act loop using local LLM inference (realizar) and persistent memory. Always Sovereign by default — all inference local, zero network. Primary entrypoint: `batuta code` (or `apr code` via apr-cli).

**Implemented (Phases 1-6b):** Multi-turn conversation with 9 tools (file_read/write/edit, glob, grep, shell, memory, pmat_query, rag), tool definitions injected into prompt for local models, ChatML/Llama3/Generic chat templates auto-detected from model filename, session persistence (JSONL at `~/.apr/sessions/`), `--resume`/`--project` CLI flags, interactive auto-resume prompt for recent sessions (PMAT-165), `/test`/`/quality`/`/context`/`/compact`/`/session` slash commands, auto-compaction at 80% context window, model discovery (APR-preferred over GGUF, with Jidoka validation), APR.md/CLAUDE.md project instruction loading, output sanitization, `apr_model_validity` contract, context-aware prompt budgeting, AprServeDriver with graceful SIGTERM→SIGKILL shutdown (PMAT-166), dedicated `pmat_query` tool (PMAT-163), Qwen3NoThinkTemplate for thinking mode suppression (PMAT-181), **`apr code` subcommand wired in apr-cli** (PMAT-162→PMAT-182).

**Planned:** OS-native sandboxing (Landlock/Seatbelt), presentar-terminal TUI, pre/post tool hooks, multi-provider hybrid routing.

### 3.7 Bug Hunter (`src/bug_hunter/`)

Proactive fault localization using 5-channel SBFL (spectrum, mutation, static, semantic, PMAT quality). Research-based techniques from Zazworka et al. (2011).

---

## 4. Feature Flags

| Flag | Purpose | Default |
|------|---------|---------|
| `native` | Full CLI, filesystem, tracing, TUI dashboard | Yes |
| `rag` | SQLite+FTS5 RAG oracle | Yes |
| `agents` | Autonomous agent runtime (`batuta code`, perceive-reason-act loop) | Yes |
| `wasm` | Browser-compatible build (no filesystem, in-memory) | No |
| `trueno-integration` | SIMD/GPU tensor operations | No |
| `oracle-mode` | Knowledge graph with trueno-graph and trueno-db | No |
| `agents-inference` | Agent with local inference via RealizarDriver (GGUF/APR) | No |
| `agents-rag` | Agent with trueno-rag document retrieval | No |

---

## 5. Design Principles (Toyota Production System)

| Principle | Application in Batuta |
|-----------|----------------------|
| **Jidoka** (Stop-on-error) | Pipeline halts on first defect; circuit breakers in serving |
| **Poka-Yoke** (Mistake-proofing) | Privacy tiers prevent data leakage; type safety at API boundaries |
| **Heijunka** (Level loading) | SpilloverRouter balances local/remote; work-stealing in repartir |
| **Muda** (Waste elimination) | Cost circuit breakers; content-addressed caching in playbooks |
| **Kaizen** (Continuous improvement) | MoE backend selection; calibration feedback loops |
| **Genchi Genbutsu** (Go and see) | Hash-based validation; benchmark on actual hardware |

---

## 6. Quality Standards

| Metric | Target |
|--------|--------|
| Test coverage | >= 95% (90% enforced, 95% preferred) |
| Clippy warnings | Zero (`-D warnings`) |
| Mutation testing | >= 80% kill rate |
| TDG Score | A grade (>= 85) |
| Pre-commit time | < 30s |

---

## 7. Key Commands

```bash
# batuta code / apr code (agentic coding assistant — sovereign-first)
batuta code                       # Interactive — auto-discovers model from ~/.apr/models/
batuta code --model ~/.apr/models/qwen3-1.7b-q4k.apr  # Default go-to model (0.960 tool score, APR preferred)
batuta code -p "Fix the auth bug" # Non-interactive: print response and exit
batuta code --resume              # Resume most recent session for this directory
batuta code --resume=<session-id> # Resume specific session
batuta code --project ../other    # Load APR.md/CLAUDE.md from another directory

# Slash commands inside batuta code:
# /test, /quality, /context, /compact, /session, /sessions, /help, /quit

# Stack management
batuta stack check              # Dependency health
batuta stack status             # TUI dashboard
batuta stack versions           # Check crates.io versions
batuta stack quality            # Quality matrix
batuta stack gate               # CI quality gate

# Oracle (knowledge queries)
batuta oracle "How do I train a model?"
batuta oracle --rag "tokenization"
batuta oracle --recipe ml-random-forest --format code
batuta oracle --pmat-query "error handling"

# Agent runtime (engine underneath apr code)
batuta agent run --manifest agent.toml

# Playbook (DAG pipelines)
batuta playbook run pipeline.yaml
batuta playbook status
batuta playbook validate pipeline.yaml

# Analysis
batuta analyze --languages --tdg .
```

---

## 8. APR v2 Model Format

The `.apr` format is the stack's native model serialization:

| Feature | APR v1 | APR v2 |
|---------|--------|--------|
| Tensor Compression | None | LZ4/ZSTD |
| Index Format | JSON | Binary |
| Zero-Copy Loading | Partial | Full |
| Quantization | Int8 | Int4/Int8 |
| Streaming | No | Yes |

**Layout policy (LAYOUT-002):** The entire stack uses **row-major** tensor layout. GGUF column-major data is transposed at import by aprender.

---

## 9. Component Specifications

| Component Spec | Description | Key Source Files |
|----------------|-------------|-----------------|
| [oracle-and-rag.md](components/oracle-and-rag.md) | Oracle knowledge graph, RAG (SQLite+FTS5), PMAT query integration, code snippets | `src/oracle/`, `src/cli/oracle/` |
| [transpilation-pipeline.md](components/transpilation-pipeline.md) | 5-phase pipeline, transpiler integration (Decy/Depyler/Bashrs), CITL cross-language learning | `src/pipeline.rs`, `src/*_converter.rs` |
| [stack-management.md](components/stack-management.md) | Dependency graph, coordinated releases, quality matrix, QA checklist | `src/stack/`, `src/cli/stack/` |
| [agent-and-playbook.md](components/agent-and-playbook.md) | Autonomous agent runtime (perceive-reason-act), context compaction, parallel tools, OS sandboxing, hooks, session persistence, YAML playbook DAG pipelines | `src/agent/`, planned |
| [multi-provider-api.md](components/multi-provider-api.md) | Provider-agnostic LLM client (Anthropic/OpenAI translation), streaming SSE, exponential backoff, provider failover, cost tracking | `src/agent/driver/remote/` |
| [presentar-probar-integration.md](components/presentar-probar-integration.md) | Agent TUI via presentar-terminal (6 panels), Brick UX contracts, probar pixel coverage + state machine playbooks + M1-M5 mutation testing, visual regression | `src/agent/tui/`, `src/agent/brick/`, `tests/playbooks/` |
| [apr-code.md](components/apr-code.md) | `apr code` / `batuta code` — Sovereign-only agentic coding assistant. Phases 1-4a DONE: 9 tools (incl. pmat_query, rag), multi-turn history, session persistence (JSONL), model discovery (APR-preferred, Jidoka-validated), chat templates, auto-compaction, output sanitization, `apr_model_validity` contract, AprServeDriver (CUDA/GPU via apr serve), `apr code` subcommand in apr-cli (PMAT-162). All inference local via realizar (GGUF/APR). | `src/agent/code.rs`, `src/agent/`, `src/agent/session.rs` |
| [apr-code-tui-testing.md](components/apr-code-tui-testing.md) | Probar-first TUI testing spec: per-panel test harnesses, pixel coverage, visual regression baselines, state machine playbooks, Brick falsification, WCAG AA/AAA accessibility, frame budget benchmarks. Contracts: `tui-rendering-v1`, `tui-panels-v1` | `tests/tui/`, presentar-terminal, jugar-probar |
| [falsification-report.md](components/falsification-report.md) | Cross-spec Popperian falsification: 12 contradictions, 8 unfalsifiable claims, 6 missing failure modes, 4 circular dependencies. Priority fixes applied inline. | All specs |
| [apr-code-feasibility-falsification.md](components/apr-code-feasibility-falsification.md) | Code-verified feasibility of `apr code`: dependency chain (no circular dep), 2 real gaps (REPL + file tools), 77% reuse of 5,000+ existing agent lines | `src/agent/`, `apr-cli` |
| [banco-spec.md](components/banco-spec.md) | Banco AI workbench overview — unified AI studio with OpenAI-compatible API | `src/serve/` |
| [banco-phase1.md](components/banco-phase1.md) | Banco Phase 1: core endpoints (chat, completions, models) | `src/serve/` |
| [banco-phase2.md](components/banco-phase2.md) | Banco Phase 2: model slot, load/unload, system presets, auth | `src/serve/` |
| [banco-phase3.md](components/banco-phase3.md) | Banco Phase 3: inference integration with realizar OwnedQuantizedModel | `src/serve/` |
| [banco-phase4.md](components/banco-phase4.md) | Banco Phase 4: advanced features (streaming, embeddings, fine-tune) | `src/serve/` |
| [banco-infra.md](components/banco-infra.md) | Banco infrastructure: Axum server, CORS, config, deployment | `src/serve/` |
| [banco-contracts.md](components/banco-contracts.md) | Banco provable contracts: 5 YAML contracts for API correctness | `../provable-contracts/contracts/batuta/` |
| [banco-cross-cutting.md](components/banco-cross-cutting.md) | Banco cross-cutting: OpenAI SDK compat, error handling, logging | `src/serve/` |
| [banco-ux.md](components/banco-ux.md) | Banco UX: CLI, TUI dashboard, interactive mode | `src/cli/` |
| [banco-ux-falsification.md](components/banco-ux-falsification.md) | Banco UX Popperian falsification tests | `tests/` |
| [banco-testing.md](components/banco-testing.md) | Banco test strategy: 124 tests, 24 endpoints, falsification | `tests/` |
| [banco-falsification-report.md](components/banco-falsification-report.md) | Banco falsification report: cross-spec contradiction analysis | All banco specs |
| [quality-and-testing.md](components/quality-and-testing.md) | Popperian falsification methodology, testing ecosystem (pmat/oip/probar), bug-hunter PMAT integration | `src/bug_hunter/` |
| [external-integrations.md](components/external-integrations.md) | Data platforms (Databricks/Snowflake/AWS), visualization, content tooling, Apple hardware (manzana) | `src/cli/` |
| [sovereign-ai-architecture.md](components/sovereign-ai-architecture.md) | Formal architecture spec, lifetime/memory model, stack diagnostics and reporting | Core architecture |

---

## 10. Archive

All original specification files are preserved unchanged in [archive/](archive/). The component specs above are condensed versions that consolidate related topics and remove redundancy while preserving technical accuracy.

### Archived Files (29 total)

| Original File | Lines | Consolidated Into |
|---------------|-------|-------------------|
| `oracle-mode-spec.md` | 1229 | oracle-and-rag |
| `apr-powered-rag-oracle.md` | 1002 | oracle-and-rag |
| `sqlite-rag-integration.md` | 1318 | oracle-and-rag |
| `improve-oracle.md` | 1818 | oracle-and-rag |
| `pmat-query-batuta-oracle-integration.md` | 283 | oracle-and-rag |
| `code-snippets.md` | 242 | oracle-and-rag |
| `batuta-orchestration-...spec.md` | 2331 | transpilation-pipeline |
| `citl-cross-language-spec.md` | 698 | transpilation-pipeline |
| `batuta-stack-spec.md` | 1750 | stack-management |
| `stack-quality-matrix-spec.md` | 1094 | stack-management |
| `stack-tree-view.md` | 225 | stack-management |
| `batuta-stack-0.1-100-point-qa-checklist.md` | 186 | stack-management |
| `score-a-plus-spec.md` | 71 | stack-management |
| `book-score-spec.md` | 71 | stack-management |
| `batuta-agent.md` | 2257 | agent-and-playbook |
| `batuta-playbook.md` | 2056 | agent-and-playbook |
| `model-serving-ecosystem-spec.md` | 444 | banco-spec |
| `hugging-face-integration-query-publish-spec.md` | 619 | banco-spec |
| `hugging-face-crud-spec.md` | 448 | banco-spec |
| `retriever-spec.md` | 3019 | banco-spec |
| `popperian-falsification-checklist.md` | 3690 | quality-and-testing |
| `testing-quality-ecosystem-spec.md` | 372 | quality-and-testing |
| `bug-hunter-pmat-quality-integration.md` | 233 | quality-and-testing |
| `data-platforms-integration-spec-query.md` | 1131 | external-integrations |
| `data-visualization-integration-query.md` | 439 | external-integrations |
| `content-creation-tooling-spec.md` | 980 | external-integrations |
| `manzana-apple-hardware-spec.md` | 215 | external-integrations |
| `sovereign-ai-spec.md` | 868 | sovereign-ai-architecture |
| `stack-visualization-diagnostics-reporting.md` | 1223 | sovereign-ai-architecture |
