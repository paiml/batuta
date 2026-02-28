# Agent Runtime

The Batuta Agent Runtime provides autonomous agent execution using the
perceive-reason-act pattern. All inference runs locally by default
(sovereign privacy), with optional remote fallback for hybrid deployments.

## Architecture

```
AgentManifest (TOML)
  → PERCEIVE: recall memories (BM25 / substring)
  → REASON:   LlmDriver.complete() with retry+backoff
  → ACT:      Tool.execute() with capability checks
  → GUARD:    LoopGuard checks iteration/cost/ping-pong
  → repeat until Done or circuit-break
```

### Module Structure

```
src/agent/
  mod.rs          # AgentBuilder, pub exports
  runtime.rs      # run_agent_loop() — core perceive-reason-act
  phase.rs        # LoopPhase (Perceive, Reason, Act, Done, Error)
  guard.rs        # LoopGuard (Jidoka: iteration/cost/ping-pong limits)
  result.rs       # AgentLoopResult, AgentError, StopReason
  manifest.rs     # AgentManifest TOML config
  capability.rs   # Capability enum, capability_matches() (Poka-Yoke)
  signing.rs      # Ed25519 manifest signing via pacha+blake3
  contracts.rs    # Design-by-Contract YAML verification
  driver/
    mod.rs        # LlmDriver trait, CompletionRequest/Response
    realizar.rs   # RealizarDriver — sovereign local inference
    mock.rs       # MockDriver — deterministic testing
    remote.rs     # RemoteDriver — Anthropic/OpenAI HTTP
    router.rs     # RoutingDriver — local-first with fallback
  tool/
    mod.rs        # Tool trait, ToolRegistry
    rag.rs        # RagTool — wraps oracle::rag::RagOracle
    memory.rs     # MemoryTool — read/write agent state
    shell.rs      # ShellTool — sandboxed command execution
    compute.rs    # ComputeTool — parallel task execution
    browser.rs    # BrowserTool — headless Chromium (agents-browser)
  memory/
    mod.rs        # MemorySubstrate trait, MemoryFragment
    in_memory.rs  # InMemorySubstrate (ephemeral)
    trueno.rs     # TruenoMemory (SQLite + FTS5 BM25)
```

## Toyota Production System Principles

| Principle | Application |
|-----------|------------|
| **Jidoka** | `LoopGuard` stops on ping-pong, budget, max iterations |
| **Poka-Yoke** | Capability system prevents unauthorized tool access |
| **Muda** | Cost circuit breaker prevents runaway spend |
| **Heijunka** | `RoutingDriver` balances load between local and remote |
| **Genchi Genbutsu** | Default sovereign — local hardware, no proxies |

## LlmDriver Trait

The driver abstraction separates the agent loop from inference backends:

```rust
#[async_trait]
pub trait LlmDriver: Send + Sync {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, AgentError>;

    fn context_window(&self) -> usize;
    fn privacy_tier(&self) -> PrivacyTier;
}
```

### Available Drivers

| Driver | Privacy | Feature | Use Case |
|--------|---------|---------|----------|
| `RealizarDriver` | Sovereign | `inference` | Local GGUF/APR inference |
| `MockDriver` | Sovereign | `agents` | Deterministic testing |
| `RemoteDriver` | Standard | `native` | Anthropic/OpenAI APIs |
| `RoutingDriver` | Configurable | `native` | Local-first with remote fallback |

## Tool System

Tools extend agent capabilities. Each declares a required `Capability`;
the manifest must grant it (Poka-Yoke error-proofing):

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn definition(&self) -> ToolDefinition;
    async fn execute(&self, input: serde_json::Value) -> ToolResult;
    fn required_capability(&self) -> Capability;
    fn timeout(&self) -> Duration;
}
```

## Memory Substrate

Agents persist state across invocations via the `MemorySubstrate` trait:

| Implementation | Backend | Feature | Recall Strategy |
|---------------|---------|---------|----------------|
| `InMemorySubstrate` | HashMap | `agents` | Case-insensitive substring |
| `TruenoMemory` | SQLite + FTS5 | `rag` | BM25-ranked full-text search |

## Design by Contract

Formal invariants are defined in `contracts/agent-loop-v1.yaml` and
verified at test time:

| ID | Invariant | Verified By |
|----|-----------|-------------|
| INV-001 | Loop terminates within max iterations | `test_iteration_limit` |
| INV-002 | Guard budget monotonically increases | `test_counters` |
| INV-003 | Capability denied returns error | `test_capability_denied_handled` |
| INV-004 | Ping-pong detected and halted | `test_pingpong_detection` |
| INV-005 | Cost budget enforced | `test_cost_budget` |
| INV-006 | Consecutive MaxTokens circuit-breaks | `test_consecutive_max_tokens` |
| INV-007 | Conversation stored in memory | `test_conversation_stored_in_memory` |

## Feature Gates

```toml
agents = ["native"]                         # Core agent loop
agents-browser = ["agents", "jugar-probar"] # Headless browser tool
```

## CLI Commands

```bash
# Single-turn execution
batuta agent run --manifest agent.toml --prompt "Hello"

# Interactive chat
batuta agent chat --manifest agent.toml

# Validate manifest
batuta agent validate --manifest agent.toml
```

See [`batuta agent` CLI Reference](../part6/cli-agent.md) for full details.
