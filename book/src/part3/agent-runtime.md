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

    /// Estimate cost in USD for a completion's token usage.
    /// Default: 0.0 (sovereign/local inference is free).
    fn estimate_cost(&self, _usage: &TokenUsage) -> f64 { 0.0 }
}
```

### Cost Budget Enforcement (INV-005)

After each LLM completion, the runtime estimates cost via
`driver.estimate_cost(usage)` and feeds it to
`guard.record_cost(cost)`. When accumulated cost exceeds
`max_cost_usd`, the guard triggers a `CircuitBreak` (Muda
elimination — prevent runaway spend).

| Driver | Cost Model |
|--------|-----------|
| `RealizarDriver` | 0.0 (sovereign, free) |
| `MockDriver` | Configurable via `with_cost_per_token(rate)` |
| `RemoteDriver` | $3/$15 per 1M tokens (input/output) |

### Available Drivers

| Driver | Privacy | Feature | Use Case |
|--------|---------|---------|----------|
| `RealizarDriver` | Sovereign | `inference` | Local GGUF/APR inference |
| `MockDriver` | Sovereign | `agents` | Deterministic testing |
| `RemoteDriver` | Standard | `native` | Anthropic/OpenAI APIs |
| `RoutingDriver` | Configurable | `native` | Local-first with remote fallback |

### RemoteDriver

The `RemoteDriver` supports both Anthropic Messages API and OpenAI Chat
Completions API for hybrid deployments:

| Provider | Endpoint | Tool Format |
|----------|----------|-------------|
| Anthropic | `/v1/messages` | `tool_use` content blocks |
| OpenAI | `/v1/chat/completions` | `function` tool_calls |

Error mapping: HTTP 429 → RateLimited, 529/503 → Overloaded, other → Network.

### RoutingDriver

The `RoutingDriver` wraps a primary (typically local/sovereign) and fallback
(typically remote/cloud) driver with three strategies:

| Strategy | Behavior |
|----------|----------|
| `PrimaryWithFallback` | Try primary; on retryable error, spillover to fallback |
| `PrimaryOnly` | Primary only, no fallback |
| `FallbackOnly` | Fallback only, skip primary |

Privacy tier inherits the most permissive of the two drivers — if the
fallback is `Standard`, data *may* leave the machine on spillover.
Metrics track primary attempts, spillovers, and fallback success rate.

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

### Builtin Tools

| Tool | Capability | Description |
|------|-----------|-------------|
| `MemoryTool` | `Memory` | Read/write agent persistent state |
| `RagTool` | `Rag` | Search indexed documentation via BM25+vector |
| `ShellTool` | `Shell` | Sandboxed subprocess execution with allowlisting |
| `ComputeTool` | `Compute` | Parallel task execution via JoinSet |
| `BrowserTool` | `Browser` | Headless Chromium automation |

### ShellTool Security (Poka-Yoke)

The `ShellTool` executes shell commands with multi-layer protection:

1. **Allowlist**: Only commands in the `allowed_commands` list can execute
2. **Injection prevention**: Metacharacters (`;|&&||`$()`) are blocked
3. **Working directory**: Restricted to configured path
4. **Output truncation**: Capped at 8192 bytes
5. **Timeout**: Default 30 seconds, configurable

### ComputeTool

Parallel task execution for compute-intensive workflows:

- Single task execution (`run` action)
- Parallel execution (`parallel` action) via tokio `JoinSet`
- Max concurrent tasks configurable (default: 4)
- Output truncated to 16KB per task
- Configurable timeout (default: 5 minutes)

## Memory Substrate

Agents persist state across invocations via the `MemorySubstrate` trait:

| Implementation | Backend | Feature | Recall Strategy |
|---------------|---------|---------|----------------|
| `InMemorySubstrate` | HashMap | `agents` | Case-insensitive substring |
| `TruenoMemory` | SQLite + FTS5 | `rag` | BM25-ranked full-text search |

## Manifest Signing

Agent manifests can be cryptographically signed using Ed25519 via
`pacha` + BLAKE3 hashing:

```bash
# Sign a manifest
batuta agent sign --manifest agent.toml --signer "admin@paiml.com"

# Verify a signature
batuta agent verify-sig --manifest agent.toml --pubkey key.pub
```

The signing system normalizes TOML to canonical form before hashing
to ensure deterministic signatures regardless of formatting.

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

## Falsification Tests

Popperian tests that attempt to *break* invariants, per spec §13.2:

| ID | Invariant | Test |
|----|-----------|------|
| FALSIFY-AL-001 | Loop termination | Infinite ToolUse must hit max iterations |
| FALSIFY-AL-002 | Deny-by-default | Empty capabilities deny all tool calls |
| FALSIFY-AL-003 | Ping-pong detection | Same tool call 3x triggers Block |
| FALSIFY-AL-004 | Cost circuit breaker | High tokens + low budget = CircuitBreak |
| FALSIFY-AL-005 | MaxTokens circuit break | 5 consecutive MaxTokens = CircuitBreak |
| FALSIFY-AL-006 | MaxTokens reset | Interleaved ToolUse resets counter |
| FALSIFY-AL-007 | Memory storage | Conversation stored after loop completes |

## Feature Gates

```toml
agents = ["native"]                         # Core agent loop
agents-inference = ["agents", "inference"]  # Local GGUF/APR inference
agents-rag = ["agents", "rag"]              # RAG pipeline
agents-browser = ["agents", "jugar-probar"] # Headless browser tool
agents-full = ["agents-inference", "agents-rag"]  # All agent features
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
