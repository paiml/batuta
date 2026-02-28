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
  pool.rs         # AgentPool — multi-agent fan-out/fan-in
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
    mcp_server.rs # HandlerRegistry — expose tools via MCP
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
| `McpClientTool` | `Mcp` | Proxy tool calls to external MCP servers |

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

### MCP Client Tool

The `McpClientTool` wraps external MCP servers as agent tools. Each tool
discovered from an MCP server becomes a separate `McpClientTool` instance:

```rust
use batuta::agent::tool::mcp_client::{McpClientTool, McpTransport};

let tool = McpClientTool::new(
    "code-search",              // server name
    "search",                   // tool name
    "Search codebase",          // description
    serde_json::json!({ ... }), // input schema
    Box::new(transport),        // McpTransport impl
);
```

| Aspect | Detail |
|--------|--------|
| Name format | `mcp_{server}_{tool}` |
| Capability | `Mcp { server, tool }` with wildcard support |
| Privacy | Sovereign tier restricts to stdio transport only |
| Timeout | Default 30 seconds, configurable |

Capability matching supports wildcards: `Mcp { server: "code-search", tool: "*" }`
grants access to all tools on the `code-search` server.

### Tool Output Sanitization (Poka-Yoke)

All tool results are sanitized before entering the conversation history.
The `ToolResult::sanitized()` method strips known prompt injection patterns:

| Pattern | Example |
|---------|---------|
| ChatML system | `<\|system\|>`, `<\|im_start\|>system` |
| LLaMA instruction | `[INST]`, `<<SYS>>` |
| Override attempts | `IGNORE PREVIOUS INSTRUCTIONS`, `DISREGARD PREVIOUS` |
| System override | `NEW SYSTEM PROMPT:`, `OVERRIDE:` |

Matching is case-insensitive. Detected patterns are replaced with `[SANITIZED]`.
This prevents a malicious tool output from hijacking the LLM's behavior.

## Multi-Agent Pool

The `AgentPool` manages concurrent agent instances with fan-out/fan-in
patterns. Each spawned agent runs its own perceive-reason-act loop in
a separate tokio task.

```rust
use batuta::agent::pool::{AgentPool, SpawnConfig};

let mut pool = AgentPool::new(driver, 4);  // max 4 concurrent

// Fan-out: spawn multiple agents
pool.spawn(SpawnConfig {
    manifest: summarizer_manifest,
    query: "Summarize this doc".into(),
})?;
pool.spawn(SpawnConfig {
    manifest: extractor_manifest,
    query: "Extract entities".into(),
})?;

// Fan-in: collect all results
let results = pool.join_all().await;
```

| Method | Purpose |
|--------|---------|
| `spawn(config)` | Spawn a single agent, returns `AgentId` |
| `fan_out(configs)` | Spawn multiple agents at once |
| `join_all()` | Wait for all agents, return `HashMap<AgentId, Result>` |
| `join_next()` | Wait for next agent to complete |
| `abort_all()` | Cancel all running agents |

Capacity enforcement: `spawn` returns `CircuitBreak` error when the pool
is at `max_concurrent`. This prevents unbounded resource consumption (Muda).

## Tracing Instrumentation

The agent runtime emits structured tracing spans for debugging and
observability. Enable with `RUST_LOG=batuta::agent=debug`:

| Span | Fields | When |
|------|--------|------|
| `run_agent_loop` | `agent`, `query_len` | Entire agent session |
| `tool_execute` | `tool`, `id` | Each tool call |
| `call_with_retry` | — | LLM completion with retry |
| `handle_tool_calls` | `num_calls` | Processing tool batch |

Key trace events:
- `agent loop initialized` — tools and capabilities loaded
- `loop iteration start` — iteration count, total tool calls
- `tool execution complete` — tool name, is_error, output_len
- `agent loop complete` — final iterations, tool calls, stop reason
- `retryable driver error` — attempt count, error details

## MCP Server (Handler Registry)

The `HandlerRegistry` exposes agent tools as MCP server endpoints,
allowing external LLM clients to call the agent's tools over MCP:

```rust
use batuta::agent::tool::mcp_server::{HandlerRegistry, MemoryHandler};

let mut registry = HandlerRegistry::new();
registry.register(Box::new(MemoryHandler::new(memory, "agent-id")));

// MCP tools/list
let tools = registry.list_tools();

// MCP tools/call
let result = registry.dispatch("memory", params).await;
```

| Handler | Actions | Description |
|---------|---------|-------------|
| `MemoryHandler` | `store`, `recall` | Store/search agent memory fragments |

The handler pattern is forward-compatible with pforge `Handler` trait.
When pforge is added as a dependency, handlers implement the pforge
trait directly for full MCP protocol compliance.

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
verified at test time. Three core functions have compile-time
`#[contract]` bindings (via `provable-contracts-macros`, feature-gated
behind `agents-contracts`):

| Function | Contract | Equation |
|----------|----------|----------|
| `run_agent_loop` | `agent-loop-v1` | `loop_termination` |
| `capability_matches` | `agent-loop-v1` | `capability_match` |
| `LoopGuard::record_cost` | `agent-loop-v1` | `guard_budget` |

| ID | Invariant | Verified By |
|----|-----------|-------------|
| INV-001 | Loop terminates within max iterations | `test_iteration_limit` |
| INV-002 | Guard counter monotonically increases | `test_counters` |
| INV-003 | Capability denied returns error | `test_capability_denied_handled` |
| INV-004 | Ping-pong detected and halted | `test_pingpong_detection` |
| INV-005 | Cost budget enforced | `test_cost_budget` |
| INV-006 | Consecutive MaxTokens circuit-breaks | `test_consecutive_max_tokens` |
| INV-007 | Conversation stored in memory | `test_conversation_stored_in_memory` |
| INV-008 | Pool capacity enforcement | `test_pool_capacity_limit` |
| INV-009 | Fan-out count preservation | `test_pool_fan_out_fan_in` |
| INV-010 | Fan-in completeness | `test_pool_join_all` |
| INV-011 | Tool output sanitization | `test_sanitize_output_system_injection` |

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
agents-mcp = ["agents", "pmcp", "pforge-runtime"]  # MCP client+server
agents-contracts = ["agents", "provable-contracts"] # #[contract] macros
agents-viz = ["agents", "presentar"]        # WASM agent dashboards
agents-full = ["agents-inference", "agents-rag"]    # All agent features
```

### MCP Manifest Configuration

When `agents-mcp` is enabled, `AgentManifest` gains an `mcp_servers` field
for declaring external MCP server connections:

```toml
[[mcp_servers]]
name = "code-search"
transport = "stdio"
command = ["node", "server.js"]
capabilities = ["*"]
```

| Transport | Privacy | Description |
|-----------|---------|-------------|
| `stdio` | Sovereign | Subprocess via stdin/stdout |
| `sse` | Standard only | Server-Sent Events over HTTP |
| `websocket` | Standard only | WebSocket full-duplex |

Sovereign privacy tier blocks `sse` and `websocket` transports at
validation time (Poka-Yoke).

## CLI Commands

```bash
# Single-turn execution
batuta agent run --manifest agent.toml --prompt "Hello"

# Interactive chat
batuta agent chat --manifest agent.toml

# Validate manifest
batuta agent validate --manifest agent.toml

# Multi-agent fan-out
batuta agent pool \
  --manifest summarizer.toml \
  --manifest extractor.toml \
  --manifest analyzer.toml \
  --prompt "Analyze this document" \
  --concurrency 2

# Sign and verify manifests
batuta agent sign --manifest agent.toml --signer "admin"
batuta agent verify-sig --manifest agent.toml --pubkey key.pub

# Show contract invariants
batuta agent contracts

# Show manifest status
batuta agent status --manifest agent.toml
```

| Subcommand | Purpose |
|-----------|---------|
| `run` | Single-turn agent execution |
| `chat` | Interactive multi-turn session |
| `validate` | Validate manifest without running |
| `pool` | Fan-out multiple agents, fan-in results |
| `sign` | Ed25519 manifest signing |
| `verify-sig` | Verify manifest signature |
| `contracts` | Display contract invariant bindings |
| `status` | Show manifest configuration |

See [`batuta agent` CLI Reference](../part6/cli-agent.md) for full details.
