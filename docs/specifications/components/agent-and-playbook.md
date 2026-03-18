# Agent and Playbook Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> Sources: batuta-agent, batuta-playbook

---

## Part I: Agent Runtime

### 1. Overview

Batuta Agent extends the orchestration framework with an autonomous agent runtime. Agents execute perceive-reason-act loops using local LLM inference (realizar), retrieval-augmented generation (trueno-rag), and persistent memory (trueno-db) -- all sovereign by default, with zero external API dependencies.

### Non-Goals

- Not a chatbot framework (no Slack/Discord adapters)
- Not a workflow engine (batuta playbook handles that)
- Not a model trainer (entrenar does training)
- Not an API gateway (batuta serve handles endpoints)

### 2. Architecture

```
batuta/
  +-- serve/      # routing, privacy, failover, context     (REUSE)
  +-- oracle/     # RAG pipeline, knowledge graph           (REUSE)
  +-- agent/      # perceive-reason-act runtime             (NEW)
  |     +-- driver/   # LlmDriver trait + implementations
  |     +-- tool/     # Tool trait + builtin tools
  |     +-- memory/   # MemorySubstrate trait + implementations
  +-- cli/agent.rs    # CLI subcommand
```

### Reuse Matrix

| Existing Module | Reuse in Agent |
|----------------|----------------|
| `serve::backends::PrivacyTier` | Driver privacy enforcement |
| `serve::context::ContextManager` | Token counting + truncation |
| `serve::circuit_breaker::CostCircuitBreaker` | Cost budget per invocation |
| `serve::router::SpilloverRouter` | Hybrid local->remote routing |
| `serve::templates::ChatTemplateEngine` | Prompt formatting (ChatML/Llama/Mistral) |
| `oracle::rag::RagOracle` | Document retrieval tool |
| `oracle::knowledge_graph` | Stack component queries |

### 3. Agent Loop

```
                    batuta agent run
                          |
                    AgentManifest (TOML)
                          |
              +-----------v-----------+
              |       PERCEIVE        |
              | memory.recall(query)  |
              | -> inject into prompt |
              +-----------+-----------+
                          |
              +-----------v-----------+
              |        REASON         |
              | context.truncate()    |
              | driver.complete()     |
              +-----+----------+-----+
                    |          |
              end_turn    tool_use
                    |          |
                    |    +-----v---------+
                    |    |     ACT       |
                    |    | capability_check|
                    |    | loop_guard    |
                    |    | tool.execute  |
                    |    +---------------+
                    |
              +-----v-----------------+
              |      REMEMBER         |
              | memory.remember()     |
              | -> AgentLoopResult    |
              +-----------------------+
```

### 4. Core Types

#### LlmDriver Trait

```rust
#[async_trait]
pub trait LlmDriver: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, AgentError>;
    async fn stream(&self, request: CompletionRequest, tx: Sender<StreamEvent>) -> Result<CompletionResponse, AgentError>;
    fn context_window(&self) -> usize;
    fn privacy_tier(&self) -> PrivacyTier;
}
```

| Driver | Privacy Tier | Backend | Phase |
|--------|-------------|---------|-------|
| `RealizarDriver` | Sovereign | Local GGUF/APR | 1 |
| `MockDriver` | Sovereign | Deterministic (testing) | 1 |
| `RemoteDriver` | Standard | HTTP to OpenAI/Anthropic | 2 |
| `RoutingDriver` | Configurable | SpilloverRouter: local-first | 2 |

#### Tool Trait

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    async fn execute(&self, input: Value) -> Result<String, ToolError>;
    fn required_capability(&self) -> Capability;
}
```

#### MemorySubstrate Trait

```rust
#[async_trait]
pub trait MemorySubstrate: Send + Sync {
    async fn remember(&self, key: &str, content: &str, metadata: Value) -> Result<()>;
    async fn recall(&self, query: &str, limit: usize) -> Result<Vec<Memory>>;
    async fn forget(&self, key: &str) -> Result<()>;
}
```

### 5. Safety Guards

| Guard | Mechanism | Toyota Principle |
|-------|-----------|-----------------|
| **LoopGuard** | Detects ping-pong patterns, budget exhaustion, max iterations | Jidoka |
| **Capability System** | Tools require capabilities; manifest declares allowed set | Poka-Yoke |
| **CostCircuitBreaker** | Prevents runaway spend on hybrid deployments | Muda |
| **PrivacyTier** | Sovereign blocks all remote egress | Poka-Yoke |

### 6. Builtin Tools

| Tool | Capability | Description |
|------|-----------|-------------|
| `RagTool` | `rag_search` | Wraps `oracle::rag::RagOracle` for doc retrieval |
| `ShellTool` | `shell_execute` | Execute shell commands (sandboxed) |
| `FileTool` | `file_read`, `file_write` | Read/write local files |
| `BrowserTool` | `browser_navigate` | Headless Chromium via jugar-probar |
| `McpTool` | `mcp_call` | Call external MCP tool servers at runtime |

---

## Part II: Playbook System

### 7. Overview

Batuta Playbook introduces deterministic, YAML-defined workflow execution. Playbooks declare multi-stage data pipelines as directed acyclic graphs (DAGs) with content-addressed caching, hash-based invalidation, and distributed execution.

**Core Invariants:**
- **Determinism**: Identical inputs always produce identical outputs
- **Idempotency**: Re-running skips stages whose input hashes match the lock file
- **Resumability**: Failed pipelines resume from last successful stage
- **Provenance**: Every artifact is content-addressed (BLAKE3) with full lineage in pacha
- **Shell safety**: All `cmd` fields purified through bashrs before execution

### 8. YAML Schema

```yaml
version: "1.0"
name: "pipeline-name"
description: "Pipeline description"

params:
  model: "moonshine-tiny"
  chunk_size: 512

targets:
  workstation:
    host: localhost
  server:
    host: intel
    cores: 32

stages:
  extract:
    description: "Extract data"
    cmd: "ffmpeg -i {{input}} -vn {{output}}"
    deps:
      - path: /data/input/
        type: directory
    outs:
      - path: /data/output/
        type: audio
    target: workstation
    parallel:
      strategy: per_file
      glob: "**/*.mp4"
      max_workers: 8

  process:
    description: "Process extracted data"
    cmd: "tool --input {{deps[0].path}} --output {{outs[0].path}}"
    deps:
      - path: /data/output/
    params:
      - model
    outs:
      - path: /data/processed/
    target: server
    retry:
      limit: 3
      policy: on_failure
      backoff:
        initial: 5s
        factor: 2

compliance:
  pre_flight:
    - tdg:
        path: ../component/src/
        min_grade: C
  post_flight:
    - quality_gate:
        min_grade: B

policy:
  failure: stop_on_first   # Jidoka
  validation: checksum     # BLAKE3
  lock_file: true
```

### 9. DAG Construction

DAG is implicit from deps/outs: if stage B lists path X in deps and stage A lists X in outs, then B depends on A. Explicit ordering via `after` for non-data dependencies.

**Wildcard expansion** (Snakemake pattern): Paths with `{name}` expand into per-sample DAG nodes at runtime, each with independent caching and retry.

### 10. Execution Model

| Feature | Implementation | Prior Art |
|---------|---------------|-----------|
| Content addressing | BLAKE3 hashes for all artifacts | DVC |
| Lock file | Records input/output hashes per stage | DVC |
| Parallel fan-out | per_file strategy with repartir work-stealing | Snakemake |
| Multi-machine | SSH-based remote execution via targets | Nextflow |
| Hash-bucketed work dirs | Isolate parallel stage outputs | Nextflow |
| Event log | Append-only for crash recovery and audit | Temporal |
| Resource scheduling | Per-stage resource declarations | Snakemake |
| Frozen stages | Never re-executed (pinned results) | DVC |

### 11. CLI Commands

```bash
batuta playbook run pipeline.yaml       # Execute pipeline
batuta playbook run --stage extract     # Run single stage + deps
batuta playbook status                  # Show pipeline state
batuta playbook lock pipeline.yaml      # Generate/update lock file
batuta playbook validate pipeline.yaml  # Schema + DAG validation
batuta playbook visualize pipeline.yaml # ASCII DAG visualization
```

### 12. Design Principles

| Principle | Playbook Application |
|-----------|---------------------|
| **Genchi Genbutsu** | Hash-based validation -- trust checksums, not timestamps |
| **Jidoka** | `stop_on_first` failure by default |
| **Muda** | Skip unchanged stages (DVC-style invalidation) |
| **Heijunka** | Load-balanced parallel fan-out via repartir |
| **Poka-Yoke** | JSON Schema validation of YAML before execution |
| **Kaizen** | Lock files enable incremental pipeline improvement |
