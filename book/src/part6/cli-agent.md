# `batuta agent`

Sovereign agent runtime using the perceive-reason-act pattern.

## Synopsis

```bash
batuta agent run --manifest <MANIFEST> --prompt <PROMPT> [--max-iterations <N>]
batuta agent chat --manifest <MANIFEST>
batuta agent validate --manifest <MANIFEST>
```

## Subcommands

### `run`

Execute a single agent invocation with the given prompt.

```bash
batuta agent run --manifest agent.toml --prompt "Summarize the codebase"
```

**Options:**

| Flag | Description |
|------|-------------|
| `--manifest <PATH>` | Path to agent manifest TOML file |
| `--prompt <TEXT>` | Prompt to send to the agent |
| `--max-iterations <N>` | Override max iterations from manifest |

### `chat`

Start an interactive chat session (Phase 2).

```bash
batuta agent chat --manifest agent.toml
```

### `validate`

Validate an agent manifest without running it.

```bash
batuta agent validate --manifest agent.toml
```

## Agent Manifest

The agent manifest is a TOML file that configures the runtime:

```toml
name = "code-reviewer"
version = "0.1.0"
description = "Reviews code for quality issues"

[model]
model_path = "/models/llama3-8b.gguf"
max_tokens = 4096
temperature = 0.3
system_prompt = "You are a code review assistant."

[resources]
max_iterations = 20
max_tool_calls = 50
max_cost_usd = 0.0  # 0 = unlimited (sovereign)

capabilities = ["Rag", "Memory"]
privacy = "Sovereign"
```

## Architecture

The agent uses a perceive-reason-act loop (Toyota Way: Jidoka):

```
┌─────────────────────────────────────┐
│         Perceive (Memory Recall)    │
│  Recall relevant memories, augment  │
│  system prompt with context         │
├─────────────────────────────────────┤
│         Reason (LLM Completion)     │
│  Send conversation to LlmDriver,   │
│  receive response or tool calls     │
├─────────────────────────────────────┤
│         Act (Tool Execution)        │
│  Execute tools with capability      │
│  checks (Poka-Yoke), store results  │
├─────────────────────────────────────┤
│         Guard (Jidoka)              │
│  Check iteration limits, ping-pong  │
│  detection, cost budget, truncation │
└─────────────────────────────────────┘
```

## Safety Features

- **LoopGuard**: Prevents runaway loops (max iterations, tool call limits)
- **Ping-pong detection**: FxHash-based detection of oscillatory tool calls
- **Capability filtering**: Tools only accessible if manifest grants capability
- **Cost circuit breaker**: Stops execution when cost budget exceeded
- **Privacy tier**: Sovereign (local-only), Private, or Standard

## Examples

```bash
# Validate a manifest
batuta agent validate --manifest examples/agent.toml

# Run with a prompt
batuta agent run \
  --manifest examples/agent.toml \
  --prompt "What are the main modules in this project?"

# Override iteration limit
batuta agent run \
  --manifest examples/agent.toml \
  --prompt "Find all TODO comments" \
  --max-iterations 5
```

## Builtin Tools

| Tool | Capability | Feature | Description |
|------|-----------|---------|-------------|
| `MemoryTool` | `Memory` | `agents` | Read/write agent persistent state |
| `RagTool` | `Rag` | `rag` | Search indexed documentation via BM25+vector |
| `BrowserTool` | `Browser` | `agents-browser` | Headless Chromium automation |

### BrowserTool Actions

| Action | Input | Description |
|--------|-------|-------------|
| `navigate` | `{ "url": "..." }` | Navigate to URL (Sovereign: localhost only) |
| `screenshot` | `{}` | Take page screenshot (base64 PNG) |
| `evaluate` | `{ "expression": "..." }` | Evaluate JavaScript |
| `eval_wasm` | `{ "expression": "..." }` | Evaluate WASM expression |
| `click` | `{ "selector": "..." }` | Click CSS selector |
| `wait_wasm` | `{}` | Wait for WASM runtime readiness |
| `console` | `{}` | Get console messages |

## Programmatic Usage

```rust
use batuta::agent::manifest::AgentManifest;
use batuta::agent::driver::mock::MockDriver;
use batuta::agent::memory::InMemorySubstrate;
use batuta::agent::runtime::run_agent_loop;
use batuta::agent::tool::ToolRegistry;

let manifest = AgentManifest::default();
let driver = MockDriver::single_response("Hello!");
let registry = ToolRegistry::default();
let memory = InMemorySubstrate::new();

let result = run_agent_loop(
    &manifest,
    "Say hello",
    &driver,
    &registry,
    &memory,
    None,  // Optional stream event channel
).await?;

println!("Response: {}", result.text);
```

## See Also

- [Architecture Overview](../part9/architecture-overview.md)
- [Toyota Way Principles](../part1/toyota-way.md)
- [Jidoka: Built-in Quality](../part1/jidoka.md)
