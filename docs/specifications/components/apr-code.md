# apr code Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> Runtime: [agent-and-playbook.md](agent-and-playbook.md) (batuta agent engine)
> API: [multi-provider-api.md](multi-provider-api.md) (provider routing)
> UX: [presentar-probar-integration.md](presentar-probar-integration.md) (TUI + testing)

---

## 1. Overview

`apr code` is the **agentic coding assistant** for the Sovereign AI Stack — the equivalent of Claude Code, but sovereign-first. It is a top-level subcommand of the `apr` CLI (the unified entrypoint for all stack operations) that provides an interactive, tool-using AI assistant for software engineering tasks.

**Key differentiator:** `apr code` can run fully offline using local models via realizar, with optional hybrid routing to remote providers (Anthropic, OpenAI) gated by privacy tiers. The same binary works in Sovereign mode (air-gapped), Private mode (local network), and Standard mode (cloud).

### Design Principles

| Principle | Application |
|-----------|-------------|
| **Sovereign-first** | Local inference (realizar) is the default; remote is opt-in |
| **Single primary binary** | `batuta` binary with `--features agents`. No npm, no Python, no Docker. Stack tools (pmat, renacer) invoked via shell when available |
| **Stack-native** | Deep integration with all 20+ PAIML crates (not just shell wrappers) |
| **Offline-capable** | Full functionality without internet via `--offline` flag |
| **Provably correct** | UX contracts verified by probar, behavior contracts by provable-contracts |

### What apr code IS

- An interactive AI coding assistant in the terminal
- A tool-using agent that can read/write files, run commands, search code
- A sovereign alternative to Claude Code / Cursor / Codex
- The user-facing entrypoint; batuta agent runtime is the engine underneath

### What apr code is NOT

- Not a model trainer (use `apr train` / `entrenar`)
- Not a model inspector (use `apr inspect` / `apr validate`)
- Not a deployment tool (use `apr serve` / `batuta playbook`)
- Not a transpiler (use `batuta transpile` / `depyler` / `decy`)

---

## 2. Architecture

```
User
  |
  v
apr code ("Fix the auth bug")       <-- apr-cli subcommand (aprender)
  |
  v
+----------------------------------------------+
|            apr-code Runtime                   |
|  (thin shim in apr-cli, delegates to batuta)  |
+------+--------------------+------------------+
       |                    |
       v                    v
+------+------+    +--------+---------+
| batuta      |    | batuta           |
| agent       |    | multi-provider   |
| runtime     |    | API              |
| (perceive-  |    | (Anthropic,      |
|  reason-    |    |  OpenAI, Ollama, |
|  act loop)  |    |  realizar)       |
+------+------+    +------------------+
       |
       v
+------+-----------------------------+
|   presentar-terminal TUI           |
|   (streaming, tools, cost, etc.)   |
+------------------------------------+
       |
       v (uses stack tools natively)
+------+-----------------------------+
| pmat query | renacer | depyler |   |
| cargo      | git     | shell   |   |
| file_read  | file_write | grep  |   |
+------------------------------------+
```

### Crate Boundaries

| Crate | Role in apr code |
|-------|-----------------|
| **apr-cli** (aprender) | Defines `Code` subcommand variant; thin dispatch to batuta |
| **batuta** | Agent runtime, tool execution, session management, context compaction |
| **presentar-terminal** | TUI rendering (6-panel adaptive layout) |
| **realizar** | Local LLM inference (Sovereign tier) |
| **trueno-rag** | Codebase indexing, semantic search |
| **renacer** | Syscall tracing for sandbox enforcement |
| **pmat** | Code quality queries (`pmat query` integration) |
| **probar** | UX testing, state machine validation |
| **provable-contracts** | Compile-time contract enforcement |

### Why apr, Not batuta?

`apr` is already the stack's universal CLI with 48+ subcommands for model operations (`apr run`, `apr inspect`, `apr serve`, `apr chat`). Users already know `apr`. Adding `apr code` makes the coding assistant discoverable alongside existing workflows:

```bash
apr run model.apr           # Run inference
apr chat model.apr          # Interactive chat
apr code                    # Agentic coding assistant  <-- NEW
apr serve model.apr         # Start server
apr inspect model.apr       # View model metadata
```

The alternative (`batuta agent run`) requires users to know about batuta as a separate tool. `apr code` is the user-facing surface; `batuta` is the engine.

---

## 3. User Experience

### 3.1 Launch

```bash
# Default: auto-detect best available model
apr code

# Specify model explicitly
apr code --model llama-3.2-3b.apr
apr code --model claude-sonnet-4

# Sovereign mode (offline, local only)
apr code --offline
apr code --sovereign

# With project context
apr code --project ./my-rust-project

# Resume previous session
apr code --resume
apr code --resume --session abc123
```

### 3.2 Interactive Session

```
$ apr code

  apr code v0.1.0 (realizar 0.8.3, Sovereign tier)
  Model: llama-3.2-3b.apr (local)
  Project: /home/user/my-project (142 files, Rust)

  Type a message or /help for commands.

> Fix the authentication bug in src/auth.rs

  [Perceive] Recalling project context...
  [Reason]   Reading src/auth.rs...
  [Act]      file_read src/auth.rs
  [Act]      pmat query "authentication" --include-source
  [Reason]   Found the issue: token expiry check uses <= instead of <
  [Act]      file_edit src/auth.rs (line 47)
  [Act]      shell: cargo test auth
  [Remember] Fixed: changed <= to < in token expiry comparison

  Done. 1 file modified, 12 tests pass.
  Cost: $0.00 (local inference)  |  Context: 23%  |  Turn 1/50

>
```

### 3.3 Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/model` | Switch model or provider |
| `/model list` | List available models (local + remote) |
| `/cost` | Show session cost breakdown |
| `/context` | Show context window usage |
| `/compact` | Manually trigger context compaction |
| `/session` | Show session info |
| `/sessions` | List all sessions |
| `/resume` | Resume a previous session |
| `/fork` | Fork current session |
| `/sandbox` | Show sandbox policy |
| `/providers` | List configured providers and health |
| `/theme` | Switch TUI theme |
| `/tui` | Toggle TUI panels on/off |
| `/quality` | Run pmat quality check on project |
| `/test` | Run project tests |
| `/clear` | Clear conversation history |
| `/quit` | Exit apr code |

### 3.4 CLAUDE.md / APR.md Support

`apr code` reads project-level configuration files (like Claude Code reads CLAUDE.md):

**Discovery order:**
1. `APR.md` in project root (preferred, stack-native)
2. `CLAUDE.md` in project root (compatible, for projects also using Claude Code)
3. `~/.apr/APR.md` (global user-level)
4. `~/.apr/projects/{project-hash}/APR.md` (per-project user overrides)

**APR.md format:**

```markdown
# APR.md

## Project Instructions

This project uses the Sovereign AI Stack. Always use `pmat query` for code search.

## Build Commands

- `cargo build` — debug build
- `make test` — run tests
- `make lint` — clippy

## Coding Standards

- 95% test coverage required
- Zero clippy warnings
- All functions must have doc comments

## Privacy

- Tier: Sovereign (no external API calls)
- Allowed tools: file_read, file_write, shell, pmat_query
- Blocked tools: web_fetch, web_search
```

---

## 4. Tool System

### 4.1 Builtin Tools

**Phase 1 (implemented, PMAT-103 through PMAT-106):**

| Tool | Capability | Implementation | Status |
|------|-----------|----------------|--------|
| **file_read** | `FileRead` | `agent/tool/file.rs` — line range, 128KB limit, numbered output | Done (19 tests) |
| **file_write** | `FileWrite` | `agent/tool/file.rs` — create/overwrite, parent dir creation | Done |
| **file_edit** | `FileWrite` | `agent/tool/file.rs` — unique string replacement | Done |
| **shell** | `Shell` | `agent/tool/shell.rs` — allowlist, injection blocking, timeout | Done (pre-existing) |
| **glob** | `FileRead` | `agent/tool/search.rs` — pattern match, mtime sort, 200 cap | Done (15 tests) |
| **grep** | `FileRead` | `agent/tool/search.rs` — substring match, file glob filter, binary skip | Done |
| **memory** | `Memory` | `agent/tool/memory.rs` — remember/recall via InMemorySubstrate | Done (pre-existing) |

**Phase 3 (planned — via shell fallback until dedicated tools built):**

| Tool | Capability | Current Workaround | Planned |
|------|-----------|-------------------|---------|
| **pmat_query** | Shell | `shell: pmat query "..."` | Dedicated tool with structured output |
| **cargo** | Shell | `shell: cargo test` | Dedicated tool with parsed results |
| **git** | Shell | `shell: git status` | libgit2 integration |
| **rag_search** | Rag | Not yet wired | trueno-rag index + RagTool |
| **oracle** | Shell | `shell: batuta oracle "..."` | Direct oracle API |

**Phase 4+ (planned):**

| Tool | Status |
|------|--------|
| **renacer_trace** | Requires renacer integration |
| **apr_inspect** | Requires apr-cli integration |
| **notebook_edit** | Not yet planned |

### 4.2 Stack-Native vs Shell Fallback

Where possible, tools use native Rust APIs instead of shelling out:

| Operation | Phase 1 (now) | Phase 3 (planned) |
|-----------|--------------|-------------------|
| Code search | `grep` tool (substring match) | `pmat query` (quality-annotated) |
| Build/test | `shell: cargo test` | Dedicated cargo tool with parsed output |
| Git | `shell: git status` | libgit2 integration |
| Model ops | MockDriver / RealizarDriver | Multi-provider routing |
| File search | `glob` tool (native glob crate) | Same |

Phase 1 tools use native Rust I/O (no shell) for file operations and the `glob` crate for file search. Shell tool handles everything else via subprocess. Phase 3 will add stack-native tools with richer metadata.

### 4.3 Tool Permission Model

Three layers, matching the agent-and-playbook spec:

| Layer | Mechanism | Enforcement | Status |
|-------|-----------|-------------|--------|
| **Capability** | Manifest declares allowed tools per `Capability` enum | Application-level | **Done** — `capability_matches()` in runtime.rs |
| **Allowlist** | ShellTool validates command prefix against allowlist | Application-level | **Done** — injection blocking in shell.rs |
| **Path restriction** | FileRead/FileWrite tools check `allowed_paths` via `check_prefix()` | Application-level | **Done** — symlink traversal blocked |
| **Privacy tier** | Sovereign blocks network egress in agent loop | Application-level | **Done** — runtime.rs:243-249 |
| **Hook** | Pre/post hooks intercept destructive actions | Application-level | Phase 4 |
| **Sandbox** | Landlock/Seatbelt restricts file/network access | Kernel-level | Phase 4 |

---

## 5. Sovereign Inference

### 5.1 Model Discovery

`apr code` auto-discovers local models:

```
Search order:
1. ~/.apr/models/           (apr model cache)
2. ~/.cache/huggingface/    (HF cache, converted on first use)
3. ./models/                (project-local models)
4. Ollama models            (via ollama list)
```

### 5.2 Model Requirements

Minimum model capabilities for agentic coding:

| Capability | Minimum | Recommended |
|-----------|---------|-------------|
| Context window | 8K tokens | 32K+ tokens |
| Tool use | Function calling support | Native tool_use |
| Code generation | Basic completion | Instruction-following |
| Format | APR v2, GGUF, SafeTensors | APR v2 (fastest) |

### 5.3 Recommended Models

| Model | Size | Format | Quality | Speed |
|-------|------|--------|---------|-------|
| Qwen2.5-Coder 7B | 4.5GB | APR Q4K | Good for simple tasks | Fast |
| Qwen2.5-Coder 32B | 20GB | APR Q4K | Good for complex tasks | Medium |
| DeepSeek-Coder-V2 | 16GB | APR Q4K | Excellent code quality | Medium |
| Llama 3.2 3B | 2GB | APR Q4K | Basic, fast iteration | Very fast |
| Claude Sonnet 4 | Remote | API | Best quality | Depends on network |

### 5.4 Offline Mode

`--offline` / `--sovereign` guarantees:
- Zero network syscalls (verified by renacer)
- All inference via realizar (local APR/GGUF)
- All search via local pmat/trueno-rag (no web)
- All tools restricted to local filesystem
- Session persistence to local disk only

---

## 6. Session Management

Inherits session persistence from agent-and-playbook.md section 11, with `apr code`-specific additions:

### 6.1 Session Storage

```
~/.apr/sessions/
  +-- {session_id}/
  |     +-- manifest.json        # Session metadata
  |     +-- messages.jsonl       # Conversation (append-only)
  |     +-- memory.json          # Session-local memory
  |     +-- cost.json            # Cost tracking
  |     +-- project_snapshot.json # Git state at session start
  |     +-- tools_log.jsonl      # All tool calls + results (audit)
```

### 6.2 Project Context

On session start, `apr code` captures:
- Git branch, HEAD commit, dirty files
- Project language (Rust/Python/etc.)
- APR.md / CLAUDE.md instructions
- File count, structure summary
- pmat quality baseline (if available)

### 6.3 Auto-Resume

If `apr code` detects a recent session (<24h) for the same project directory, it offers to resume:

```
$ apr code
  Found previous session (2h ago, 12 turns, $0.05)
  Resume? [Y/n]
```

---

## 7. Context Management

### 7.1 Project Indexing

On first run, `apr code` builds a trueno-rag index of the project:

```
Indexing /home/user/project...
  [P0] APR.md, CLAUDE.md, README.md
  [P1] Cargo.toml, Makefile, .github/
  [P2] src/**/*.rs (142 files)
  [P3] tests/**/*.rs (38 files)
  Index: 1,247 chunks, 892KB
  Time: 1.2s
```

Priority levels match batuta oracle's RAG indexing (P0 = project config, P1 = build config, P2 = source, P3 = tests).

### 7.2 Smart Context Selection

Before each LLM call, `apr code` selects relevant context:

1. **Always included**: System prompt + APR.md instructions
2. **RAG retrieved**: Top-K chunks relevant to current query (trueno-rag)
3. **Active files**: Files recently read/written in this session
4. **Tool results**: Recent tool outputs (subject to compaction)
5. **Git diff**: Uncommitted changes (if relevant)

### 7.3 Compaction

Two-phase compaction from agent-and-playbook.md section 7:
- Micro-compact at 60%: strip tool result bodies
- Auto-compact at 80%: summarize history, re-fetch from RAG

---

## 8. Integration with apr CLI

### 8.1 Subcommand Definition

In `apr-cli/src/commands_enum.rs`:

```rust
/// Agentic coding assistant (like Claude Code, sovereign-first)
Code {
    /// Model to use (local APR/GGUF or remote provider)
    #[arg(long)]
    model: Option<String>,

    /// Privacy tier (auto, sovereign, private, standard)
    /// auto = select based on configured providers and --offline flag
    #[arg(long, default_value = "auto")]
    tier: PrivacyTier,

    /// Resume previous session
    #[arg(long)]
    resume: bool,

    /// Session ID to resume
    #[arg(long)]
    session: Option<String>,

    /// Project directory
    #[arg(long, default_value = ".")]
    project: PathBuf,

    /// Run in offline/sovereign mode (no network)
    #[arg(long)]
    offline: bool,

    /// Agent manifest (advanced)
    #[arg(long)]
    manifest: Option<PathBuf>,

    /// Initial prompt (non-interactive mode)
    #[arg(trailing_var_arg = true)]
    prompt: Vec<String>,

    /// Print mode (non-interactive, single response)
    #[arg(long, short)]
    print: bool,

    /// Max turns before stopping
    #[arg(long, default_value = "50")]
    max_turns: u32,

    /// Session budget in USD
    #[arg(long, default_value = "5.00")]
    budget: f64,
},
```

### 8.2 Dispatch

In `apr-cli/src/dispatch.rs`:

```rust
Commands::Code { model, tier, resume, session, project, offline, manifest, prompt, print, max_turns, budget } => {
    // Delegate to batuta agent runtime
    let config = AprCodeConfig {
        model,
        tier: if offline { PrivacyTier::Sovereign } else { tier },
        resume,
        session,
        project,
        manifest,
        prompt: if prompt.is_empty() { None } else { Some(prompt.join(" ")) },
        print,
        max_turns,
        budget,
    };
    batuta::agent::run_apr_code(config).await
}
```

### 8.3 Dependency

In `apr-cli/Cargo.toml`:

```toml
[dependencies]
batuta = { version = "0.7", features = ["agent-tui"], optional = true }

[features]
code = ["dep:batuta"]  # apr code requires batuta
```

This keeps `apr` lightweight for model-only operations while enabling `apr code` when the full agent stack is needed.

---

## 9. Non-Interactive Mode

For CI/CD, scripts, and automation:

```bash
# Single prompt, print response, exit
apr code -p "Add error handling to src/parser.rs"

# Pipe input
echo "Fix the failing test" | apr code -p

# With specific model
apr code --model qwen2.5-coder-7b -p "Explain src/main.rs"

# JSON output for tooling
apr code -p --json "List all functions in src/"
```

### 9.1 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (task completed) |
| 1 | Agent error (tool failure, LLM error) |
| 2 | Budget exhausted |
| 3 | Max turns reached |
| 4 | Sandbox violation |
| 5 | No model available |

---

## 10. Configuration

### 10.1 Global Config

```toml
# ~/.apr/config.toml

[code]
default_model = "qwen2.5-coder-7b.apr"
default_tier = "sovereign"
max_turns = 50
budget_usd = 5.00
theme = "tokyo-night"
auto_resume = true
index_on_start = true

[code.providers]
# See multi-provider-api.md for full provider config
[code.providers.realizar]
priority = 1
model = "qwen2.5-coder-7b.apr"

[code.providers.ollama]
priority = 2
base_url = "http://localhost:11434"
model = "qwen2.5-coder:7b"

[code.providers.anthropic]
priority = 3
model = "claude-sonnet-4-20250514"
daily_budget_usd = 10.00

[code.tools]
# Tool allowlist (empty = all allowed)
allowed = []
# Tool blocklist
blocked = ["web_fetch", "web_search"]

[code.sandbox]
# Additional sandbox paths
read_paths = ["/usr/local/include"]
write_paths = []
```

### 10.2 Environment Variables

| Variable | Purpose |
|----------|---------|
| `APR_CODE_MODEL` | Override default model |
| `APR_CODE_TIER` | Override privacy tier |
| `APR_CODE_BUDGET` | Override session budget |
| `APR_CODE_THEME` | Override TUI theme |
| `ANTHROPIC_API_KEY` | Anthropic provider auth |
| `OPENAI_API_KEY` | OpenAI provider auth |
| `APR_CODE_OFFLINE` | Force offline mode |

---

## 11. Comparison with Claude Code

| Feature | Claude Code | apr code (Phase 1 actual) | apr code (planned) |
|---------|------------|--------------------------|-------------------|
| **Runtime** | Anthropic cloud | MockDriver dry-run (no model yet) | Local-first (realizar) + cloud |
| **Offline** | No | Yes (Sovereign tier enforced) | Yes |
| **Models** | Claude only | None yet (MockDriver) | APR, GGUF, Ollama, Claude, GPT |
| **Binary** | Node.js + npm | Single Rust binary | Same |
| **Code search** | grep/ripgrep | `grep` tool (substring match) | pmat query (Phase 3) |
| **Project config** | CLAUDE.md | Not yet | APR.md + CLAUDE.md (Phase 4) |
| **Privacy** | Standard only | Sovereign enforced in runtime | Sovereign / Private / Standard |
| **Cost** | Per-token API | Free (MockDriver) | Free (local) or API pricing |
| **TUI** | Rich streaming | Line-by-line REPL with slash commands | presentar-terminal 6-panel (Phase 2) |
| **Sandboxing** | Landlock/Seatbelt | Capability + allowlist + path restriction | + Landlock/Seatbelt (Phase 4) |
| **Session** | Yes | No (per-session only) | JSONL persistence (Phase 2) |
| **Tools** | ~15 builtin | 7 builtin (file, search, shell, memory) | 14+ (Phase 3-4) |
| **MCP** | Yes | Agent supports MCP (agents-mcp feature) | Same |

---

## 12. Implementation Phases

| Phase | Scope | Status | Refs |
|-------|-------|--------|------|
| **1** | MVP: `batuta code` subcommand, REPL with slash commands, 7 tools (file_read/write/edit, glob, grep, shell, memory), MockDriver dry-run, `-p` non-interactive mode | **DONE** | PMAT-103 through 107 |
| **1b** | Real model test: RealizarDriver with local GGUF, verify tool_use JSON generation | **Blocked** (needs GGUF model download) | PMAT-108 |
| **2** | Multi-provider: RemoteDriver (Anthropic + OpenAI), RoutingDriver failover, cost tracking, session persistence | Planned — RemoteDriver/RoutingDriver exist in code, need wiring | |
| **3** | Stack-native tools: pmat_query, cargo API, trueno-rag indexing, git integration | Planned | |
| **4** | APR.md support, hooks, Landlock/Seatbelt sandbox enforcement | Planned | |
| **5** | Probar testing, Brick UX contracts, visual regression baselines | Planned | |
| **6** | MCP server support, plugin system, multi-agent | Future | |

---

## 13. Provable Contracts

See `../provable-contracts/contracts/batuta/apr-code-v1.yaml` for the full contract. Key equations:

| Equation | Property |
|----------|----------|
| `sovereignty_guarantee` | `--offline` produces zero network syscalls (renacer verified) |
| `tool_safety` | Every tool call passes capability + hook + sandbox checks |
| `session_integrity` | resume(persist(session)) reproduces identical state |
| `apr_md_compliance` | Agent respects all APR.md instructions (blocked tools, coding standards) |
| `model_fallback` | If preferred model unavailable, graceful fallback with user notification |
| `single_binary` | `apr code` works with zero external dependencies (no npm, Python, Docker) |

---

## 14. Falsification

### 14.1 Falsifiable Claims

| Claim | Test | What Failure Means |
|-------|------|-------------------|
| **Sovereign mode has zero network calls** | renacer trace of full session under `--offline`; assert zero connect/sendto syscalls | Sovereignty is broken — air-gap unusable |
| **Local inference meets minimum quality** | Benchmark tool-use accuracy on 50 coding tasks (apr code vs Claude Code) | Local models insufficient — Sovereign tier is decorative |
| **Single binary, no external deps** | Build `apr code` on clean machine; run without npm/Python/Docker | Dependency leaked — packaging broken |
| **APR.md instructions are followed** | Block `web_fetch` in APR.md; attempt web fetch; assert blocked | Config file ignored — trust broken |
| **Session resume is lossless** | Save session at turn 20, resume, compare next 5 turns (temp=0) | Resume corrupts context — user loses work |
| **Cost tracking is accurate** | 100 Anthropic turns: compare displayed vs invoice | Users make budget decisions on wrong data |
| **Startup time < 2s** | Cold start with 1000-file project on NVMe | Too slow — users won't wait |
| **pmat query outperforms grep for code tasks** | 50 code search tasks: compare pmat query vs grep for result relevance | Stack-native tools don't justify complexity |

### 14.2 What Would Disprove This Specification

1. **Local 7B models fail >60% of coding tasks.** If Sovereign tier can't complete basic file edits, variable renames, and test fixes, `apr code` needs a larger default model or must default to Private/Standard tier. (Check: benchmark on SWE-bench-lite subset)

2. **presentar-terminal TUI adds >50ms input latency.** If the 6-panel TUI slows down interactive typing, users will disable it. The TUI must be zero-cost when no streaming is active. (Check: measure keystroke-to-echo latency with TUI on vs off)

3. **batuta dependency makes apr binary >50MB.** If the `code` feature flag bloats `apr` from ~15MB to >50MB, it should be a separate binary (`apr-code`) instead of a subcommand. (Check: measure binary size with and without `code` feature)

4. **trueno-rag indexing takes >10s for medium projects.** If initial indexing blocks the user for >10s on a 500-file Rust project, indexing must be async/incremental. (Check: benchmark on batuta, realizar, trueno repos)
