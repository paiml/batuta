# apr code Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> Runtime: [agent-and-playbook.md](agent-and-playbook.md) (batuta agent engine)
> Inference: realizar (local GGUF/APR — Sovereign only)
> UX: [presentar-probar-integration.md](presentar-probar-integration.md) (TUI + testing)

---

## 1. Overview

`apr code` is the **agentic coding assistant** for the Sovereign AI Stack — the equivalent of Claude Code, but sovereign-first. It is a top-level subcommand of the `apr` CLI (the unified entrypoint for all stack operations) that provides an interactive, tool-using AI assistant for software engineering tasks.

**Key differentiator:** `apr code` runs **exclusively on local models** via realizar. All inference stays on your machine. No API keys, no cloud, no data egress — ever. This is the Sovereign AI Stack's core promise: your code never leaves your hardware.

### Design Principles

| Principle | Application |
|-----------|-------------|
| **Sovereign-only** | All inference via realizar on local hardware. No remote APIs. |
| **Single primary binary** | `batuta` binary (agents + inference in default features). No npm, no Python, no Docker. |
| **Stack-native** | Deep integration with all 20+ PAIML crates (not just shell wrappers) |
| **Always offline** | No `--offline` flag needed — apr code is always sovereign, always local. Zero network by design. |
| **Provably correct** | UX contracts verified by probar, behavior contracts by provable-contracts |

### What apr code IS

- An interactive AI coding assistant in the terminal — **100% local, zero cloud**
- A tool-using agent that can read/write files, run commands, search code
- A sovereign alternative to Claude Code / Cursor / Codex
- Primary entrypoint: `apr code` (via `apr-cli` in the aprender workspace)
- Engine: `batuta code` (batuta agent runtime underneath)

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
+------+------+    +--------+---------+     +------------------+
| batuta      |    | AprServeDriver   |---->| apr serve run    |
| agent       |    | (PRIMARY, HTTP)  |     | (CUDA/GPU, full  |
| runtime     |    | auto-launch +    |     |  APR+GGUF, fast) |
| (perceive-  |    | localhost:port   |     +------------------+
|  reason-    |    +------------------+
|  act loop)  |    | RealizarDriver   |  <-- fallback (no apr binary)
+------+------+    | (embedded, CPU)  |
       |           +------------------+
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

**PMAT-160: Inference via `apr serve` (first-class).** `batuta code` auto-launches `apr serve run <model>` as a subprocess on a random port, connects via OpenAI-compatible HTTP API. This gives full CUDA/GPU acceleration, APR+GGUF support, and avoids feature flag issues. If `apr` is not on PATH, falls back to embedded RealizarDriver (CPU-only).

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
# Default: auto-detect from ~/.apr/models/ (prefers APR over GGUF)
apr code

# Explicit model — Qwen2.5-Coder 1.5B is the default/go-to
apr code --model ~/.apr/models/qwen2.5-coder-1.5b-q4k.apr

# Larger model for complex tasks
apr code --model ~/.apr/models/qwen2.5-coder-7b-q4k.apr

# With project context
apr code --project ./my-rust-project

# Non-interactive (single prompt)
apr code -p "Fix the auth bug in src/auth.rs"

# Resume previous session
apr code --resume
```

**Note:** `apr code` is the primary user-facing entrypoint (via `apr-cli` in the aprender workspace). `batuta code` is the engine underneath — same functionality, accessed when building batuta directly.

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

### 3.3 Multi-Turn Conversation

Each REPL session maintains a persistent `Vec<Message>` history across turns. When the user submits a prompt:

1. `run_agent_turn()` receives the full history + new query
2. The agent loop processes tool calls within the turn (inner loop)
3. On completion, all new messages (user prompt, tool calls, tool results, assistant response) are committed to history
4. Next turn sees full context from all prior turns

**Context management:**
- `/context` shows token usage (`~N / window tokens (X%)`) with warning at 80%+
- `/compact` strips tool call/result details from older turns, preserving user queries and assistant summaries
- `/clear` resets history and screen
- **Auto-compaction at 80%** context window — triggers automatically after each turn (spec §7.3, PMAT-133)
- Automatic truncation via `ContextManager` (sliding window) keeps messages within the model's context window
- Project instructions (CLAUDE.md) scaled to 25% of context budget, skipped for <4K models (PMAT-142)

**Implementation:** `run_agent_turn()` in `src/agent/runtime.rs`, `compact_history()` in `src/agent/repl_display.rs` (PMAT-115).

### 3.4 Slash Commands

**Implemented (10 commands):**

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/test` | Run `cargo test --lib` |
| `/quality` | Run clippy + test quality gate |
| `/context` | Show token usage (N / window tokens, %) |
| `/compact` | Manually trigger context compaction |
| `/session` | Show current session info (ID, turns, messages) |
| `/sessions` | List recent sessions with resume instructions |
| `/cost` | Show session cost breakdown |
| `/clear` | Clear conversation history and screen |
| `/quit` | Exit apr code |

**Planned:**

| Command | Description |
|---------|-------------|
| `/model` | Switch model mid-session (currently stub) |
| `/sandbox` | Show sandbox/capability policy |

### 3.5 CLAUDE.md / APR.md Support

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

**Phase 4a (implemented, PMAT-163):**

| Tool | Capability | Implementation | Status |
|------|-----------|----------------|--------|
| **pmat_query** | `FileRead` | `agent/tool/pmat_query.rs` — semantic/regex/literal search, TDG grade filter, complexity filter, fault patterns, exclude-tests. 7 tests. | Done (PMAT-163) |
| **rag** | `Rag` | `agent/tool/rag.rs` — trueno-rag search, starts empty, populated via `batuta oracle --rag-index` | Done (PMAT-153) |

**Available via shell fallback (dedicated tools planned):**

| Tool | Current Access | Planned Enhancement |
|------|---------------|-------------------|
| **cargo** | `shell: cargo test` / `/test` slash command | Dedicated tool with parsed results |
| **git** | `shell: git status` | libgit2 integration |
| **oracle** | `shell: batuta oracle "..."` | Direct oracle API |

**Future tools:**

| Tool | Dependency |
|------|-----------|
| **renacer_trace** | renacer integration |
| **apr_inspect** | apr-cli integration |

### 4.2 Tool-in-Prompt Architecture (Local Models)

Unlike API-based drivers (Anthropic/OpenAI) which accept tool definitions as structured parameters, local models via RealizarDriver need explicit tool definitions in the prompt text. The `build_enriched_system()` function in `chat_template.rs` (PMAT-121) appends:

1. **Tool definitions** — name, description, and compact JSON Schema for each tool
2. **Format instructions** — teaches the model to emit `<tool_call>` blocks
3. **Parsing contract** — `parse_tool_calls()` in `realizar.rs` extracts these blocks

This means the system prompt grows proportionally to the number of tools (~50 tokens per tool). With 7 tools, this adds ~350 tokens to context.

### 4.3 Stack-Native vs Shell Fallback

Where possible, tools use native Rust APIs instead of shelling out:

| Operation | Current (Phases 1-4a) | Future Enhancement |
|-----------|----------------------|-------------------|
| Code search | **`pmat_query` tool** (semantic/regex/literal, TDG grades — PMAT-163) | — |
| Build/test | `shell: cargo test` + `/test` slash command | Dedicated cargo tool |
| Git | `shell: git status` | libgit2 integration |
| Model ops | AprServeDriver (CUDA/GPU) → RealizarDriver fallback (CPU) | — |
| File ops | `file_read`/`file_write`/`file_edit` (native Rust I/O) | — |
| File search | `glob` tool (native glob crate) + `grep` tool (substring) | — |
| RAG | `rag` tool (trueno-rag, empty-index start — PMAT-153) | Auto-index on first run |

File tools use native Rust I/O. Everything else is available via `shell` tool subprocess. Slash commands (`/test`, `/quality`) provide one-keystroke shortcuts for common operations.

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

`apr code` auto-discovers local models when no `--model` flag is provided:

```
Search order:
1. ~/.apr/models/           (apr model cache — download via `apr pull`)
2. ~/.cache/huggingface/    (HF cache, converted on first use)
3. ./models/                (project-local models)
```

**Format preference:** Within each directory, `.apr` files are preferred over `.gguf` (APR is the stack's native format — faster loading, row-major layout, LZ4/ZSTD compression). Files sorted by modification time (newest first).

**PMAT-150 (Jidoka validation at discovery):** APR files are validated at discovery time — if an APR file lacks an embedded tokenizer (required for inference), it is deprioritized so valid GGUF files are tried first. This prevents the user from hitting a dead-end error when the only APR model was converted without tokenizer data. The CLI warns when GGUF fallback occurs due to an invalid APR file and suggests `apr convert` to fix it.

**Implementation:** `ModelConfig::discover_model()` in `src/agent/manifest.rs` (PMAT-116, PMAT-150). Validation via `is_valid_model_file()` in `src/agent/driver/realizar.rs`.

**Chat template auto-detection:** The prompt format is selected based on model filename:
- Qwen, DeepSeek, Yi → ChatML (`<|im_start|>role`)
- Llama → Llama 3.x format (`<|start_header_id|>role<|end_header_id|>`)
- Unknown → ChatML (most widely supported by instruct models)

### 5.2 Model Requirements

Minimum model capabilities for agentic coding:

| Capability | Minimum | Recommended |
|-----------|---------|-------------|
| Parameters | 1.5B+ (Qwen2.5-Coder) | 7B+ |
| Context window | 4K tokens | 8K+ tokens |
| Tool use | Function calling support | Native tool_use |
| Code generation | Instruction-following | Instruction-following + tool_use |
| Format | APR v2, GGUF, SafeTensors | APR v2 (fastest, preferred) |

**Default model: Qwen2.5-Coder 1.5B (APR Q4K).** This is the go-to model for development and testing — extensively validated across the Sovereign AI Stack with the most test coverage of any local model. Available in APR native format at `~/.apr/models/qwen2.5-coder-1.5b-q4k.apr`.

**Warning**: Generic chat models (e.g., TinyLlama 1.1B) cannot follow tool-use instructions and echo the system prompt. Use Qwen2.5-Coder or other code-specialized models. Output sanitization strips echoed prompts and leaked chat template markers (PMAT-146).

### 5.3 Recommended Models

| Model | Size | Format | Quality | Speed | Notes |
|-------|------|--------|---------|-------|-------|
| **Qwen2.5-Coder 1.5B** | **1.1GB** | **APR Q4K** | **Default — extensively tested** | **Very fast** | **Go-to for dev/test** |
| Qwen2.5-Coder 7B | 4.5GB | APR/GGUF Q4K | Good for complex tasks | Fast | Upgrade from 1.5B |
| Qwen2.5-Coder 32B | 20GB | APR/GGUF Q4K | Best code quality | Medium | For serious coding |
| Qwen3 8B | 5GB | APR/GGUF Q4K | Strong tool-use, multilingual | Fast | |
| DeepSeek-Coder-V2 16B | 10GB | GGUF Q4K | Excellent code quality | Medium | |

All models run locally via realizar. Prefer APR format over GGUF (native, row-major, faster loading). Download with `apr pull <model>`.

### 5.4 Always Sovereign (Enforced)

`apr code` is **always sovereign** — there is no online mode. `build_default_manifest()` unconditionally returns `PrivacyTier::Sovereign` regardless of any parameter (PMAT-117):
- Zero network syscalls (verifiable by renacer)
- All inference via realizar (local GGUF/APR)
- All search via local grep/glob (pmat query via shell)
- All tools restricted to local filesystem
- No API keys required or accepted
- The `--offline` flag is a no-op (always offline by design)

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
/// Sovereign AI coding assistant — all inference local via realizar.
Code {
    /// Path to local GGUF/APR model file (prefers .apr format)
    #[arg(long)]
    model: Option<PathBuf>,

    /// Project directory (loads APR.md/CLAUDE.md from this path)
    #[arg(long, default_value = ".")]
    project: PathBuf,

    /// Resume previous session (by ID, or most recent for cwd)
    #[arg(long)]
    resume: Option<Option<String>>,

    /// Agent manifest (advanced — overrides defaults)
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

    /// Resume previous session
    #[arg(long)]
    resume: bool,
},
```

**Note:** No `--offline` flag — `apr code` is always offline. No `--tier` flag — always Sovereign. No `--budget` flag — local inference is free.

### 8.2 Dispatch

In `apr-cli/src/dispatch.rs`:

```rust
Commands::Code { model, project, manifest, prompt, print, max_turns, resume } => {
    // Delegate to batuta agent runtime — always Sovereign
    batuta::cli::code::cmd_code(prompt, print, true /* always offline */, max_turns, 0.0, manifest)
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
apr code --model qwen2.5-coder-1.5b-q4k.apr -p "Explain src/main.rs"

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
default_model = "~/.apr/models/qwen2.5-coder-1.5b-q4k.apr"
max_turns = 50
theme = "tokyo-night"
auto_resume = true

[code.tools]
# Tool allowlist (empty = all allowed)
allowed = []
# Tool blocklist
blocked = []
```

### 10.2 Environment Variables

| Variable | Purpose |
|----------|---------|
| `APR_CODE_MODEL` | Override default model path |
| `APR_CODE_THEME` | Override TUI theme |

---

## 11. Comparison with Claude Code

| Feature | Claude Code | apr code |
|---------|------------|----------|
| **Runtime** | Anthropic cloud (requires internet) | Local only via realizar (zero network) |
| **Cost** | $3-15 per million tokens | **Free** — local inference |
| **Privacy** | Data sent to Anthropic servers | **Sovereign** — code never leaves your machine |
| **Default model** | Claude (cloud) | **Qwen2.5-Coder 1.5B** (APR, local) |
| **Model formats** | Claude only | APR (preferred), GGUF, SafeTensors |
| **Binary** | Node.js + npm | Single 18MB Rust binary |
| **Tools** | ~15 builtin | 9 tools (incl. pmat_query, rag) + 10 slash commands + shell fallback |
| **Sessions** | Cloud-synced | JSONL at `~/.apr/sessions/` with `--resume` |
| **Context mgmt** | Automatic | Auto-compact at 80%, `/context` token tracking |
| **Project config** | CLAUDE.md | APR.md (preferred) + CLAUDE.md (compatible) |
| **Sandboxing** | Landlock/Seatbelt | Capability + allowlist + path restrict |
| **Quality** | No formal verification | provable-contracts (apr_model_validity, etc.) |

**The trade-off is clear:** Claude Code has frontier model quality but requires cloud. `apr code` has local-only inference (smaller models) but guarantees sovereignty. They serve different threat models.

---

## 12. Implementation Phases

| Phase | Scope | Status | Refs |
|-------|-------|--------|------|
| **1** | MVP: `batuta code` subcommand, REPL with slash commands, 7 tools (file_read/write/edit, glob, grep, shell, memory), MockDriver dry-run, `-p` non-interactive mode | **DONE** | PMAT-103 through 107 |
| **1b** | Real model: RealizarDriver with local GGUF via `--model` flag | **DONE** — model loads, agent loop initializes (7 tools, 4 caps). CPU inference slow on debug build. | PMAT-114 |
| **2a** | Multi-turn conversation history, model discovery (APR-preferred), always-Sovereign enforcement, chat template auto-detection (ChatML/Llama3/Generic) | **DONE** | PMAT-115 through 117 |
| **2b** | Tool definitions injected into prompt for local models, enriched system prompt with `<tool_call>` format, APR.md/CLAUDE.md project instruction loading, session persistence (JSONL at `~/.apr/sessions/`) | **DONE** | PMAT-121 through 124 |
| **2c** | `--resume` and `--project` CLI flags, `/session` and `/sessions` slash commands, session resume wired end-to-end, integration tests (session roundtrip, tool injection, multi-turn) | **DONE** | PMAT-129 through 131 |
| **3a** | Auto-compaction at 80% context window (spec §7.3), token usage tracking in `/context`, `/test` and `/quality` shortcut commands | **DONE** | PMAT-133 through 135 |
| **3b** | `agents` in default features (binary ships with `code` subcommand), APR format awareness (GGUF→APR conversion tip), improved no-model UX | **DONE** | PMAT-136 through 138 |
| **3c** | Spec v2.4.0 reconciliation, project context enrichment (git, language, build system) | **DONE** | PMAT-139, PMAT-140 |
| **3d** | `inference` in default features (RealizarDriver available), context-aware prompt budgeting (scales CLAUDE.md to model context window, skips for <4K models) | **DONE** | PMAT-141, PMAT-142 |
| **3e** | **Contract: `apr_model_validity`** — APR files validated at load boundary (Jidoka). Missing tokenizer caught before REPL, not at inference. GGUF magic validated. 5 falsification tests. | **DONE** | PMAT-144, PMAT-145 |
| **3f** | Output sanitization — strip echoed system prompt from small model responses, strip leaked chat template markers, model size warning for <2K context | **DONE** | PMAT-146, PMAT-147 |
| **3g** | **Jidoka model discovery** — validate APR tokenizer at discovery time (not just at load). Invalid APR deprioritized behind valid GGUF. Tightened header scan to reject `vocab_size`-only metadata. UX warning on GGUF fallback. | **DONE** | PMAT-150 |
| **3h** | **Exit codes 2/3/4** — map `CircuitBreak`→2 (budget), `MaxIterationsReached`→3 (max turns), `CapabilityDenied`→4 (sandbox) in non-interactive mode. Constants in `exit_code` module. | **DONE** | PMAT-152 |
| **3i** | **Wire RagTool** — register `RagTool` in `build_code_tools()` with empty-index oracle. Adds `Capability::Rag` to manifest. 8 tools total. Index populated via `batuta oracle --rag-index`. | **DONE** | PMAT-153 |
| **3j** | **AprServeDriver** — auto-launch `apr serve run` as subprocess, connect via OpenAI HTTP API. Full CUDA/GPU via apr-cli. Conditional `no_gpu` for APR (wgpu bug). Lenient tool_call parser (unclosed XML + markdown code blocks). | **DONE** | PMAT-158, PMAT-160 |
| **3k** | **AprServeDriver cleanup** — remove debug `[PMAT-160]` eprintln, conditional `--gpu` flag (GGUF gets GPU, APR skips due to wgpu shader bug). Clean user-facing output. | **DONE** | PMAT-164 |
| **4a** | **Dedicated `pmat_query` tool** — replaces `shell: pmat query "..."` fallback with structured `PmatQueryTool`. Supports semantic, regex, literal search with TDG grade/complexity/fault filters. 9 tools total (10 with RAG). 7 tests. | **DONE** | PMAT-163 |
| **4b** | Stack-native tools: git integration (libgit2), auto-index on first run | Planned | |
| **5** | Hooks, Landlock/Seatbelt OS sandbox enforcement | Planned | |
| **6** | **`apr-cli` integration** — `Code` subcommand added to `commands_enum.rs` behind `code` feature flag. Dispatches to `batuta::agent::code::cmd_code()` (library-level public API). `cmd_code` refactored from binary crate to library crate for cross-crate access. Blocked by pre-existing `trueno-gpu` dep issue in aprender workspace. | **IN PROGRESS** | PMAT-162 |
| **7** | Probar testing, Brick UX contracts, visual regression baselines | Planned | |

---

## 13. Provable Contracts

See `../provable-contracts/contracts/batuta/apr-code-v1.yaml` for the full contract. Key equations:

| Equation | Property |
|----------|----------|
| `sovereignty_guarantee` | Zero network syscalls — all inference local via realizar (renacer verifiable) |
| `tool_safety` | Every tool call passes capability + allowlist + path restriction checks |
| `session_integrity` | resume(persist(session)) reproduces identical state |
| `apr_md_compliance` | Agent respects all APR.md instructions (blocked tools, coding standards) |
| `local_model_required` | If no local model found, clear error + download instructions (never silent failure) |
| `apr_model_validity` | **APR files validated at load boundary AND discovery time (Jidoka)**: embedded tokenizer required (tightened: `vocab_size` metadata alone is insufficient — PMAT-150), magic bytes checked. Broken APR deprioritized at discovery so GGUF fallback works. Actionable error with `apr convert` command. (PMAT-144, PMAT-150) |
| `single_binary` | `apr code` works with zero external dependencies (no npm, Python, Docker, no API keys) |

---

## 14. Falsification

### 14.1 Falsifiable Claims

| Claim | Test | What Failure Means |
|-------|------|-------------------|
| **Zero network calls** | renacer trace of full session; assert zero connect/sendto syscalls | Sovereignty broken — data egress possible |
| **Local inference meets minimum quality** | Benchmark tool-use accuracy on 50 coding tasks with Qwen2.5-Coder 1.5B (default) and 7B | Local models can't do tool-use — apr code is a chat-only toy |
| **Single binary, no external deps** | Build `apr code` on clean machine; run without npm/Python/Docker | Dependency leaked — packaging broken |
| **APR.md instructions are followed** | Block `web_fetch` in APR.md; attempt web fetch; assert blocked | Config file ignored — trust broken |
| **Session resume is lossless** | Save session at turn 20, resume, compare next 5 turns (temp=0) | Resume corrupts context — user loses work |
| **No model = clear error** | Run `apr code` with no models installed; assert helpful error message with download instructions | User confused, no path to fix |
| **Startup time < 2s** | Cold start with 1000-file project on NVMe | Too slow — users won't wait |
| **pmat query outperforms grep for code tasks** | 50 code search tasks: compare pmat query vs grep for result relevance | Stack-native tools don't justify complexity |

### 14.1.1 Dogfood Findings (2026-04-03)

| Finding | Resolution | Ref |
|---------|-----------|-----|
| **P0: APR Q4K converter missing tokenizer** | `save_model_tensors_q4k()` in aprender never called `insert_tokenizer_metadata()`. All other APR creation paths embedded it. Root cause: Q4K fallback path (from SafeTensors) built metadata without tokenizer. Fix: pass `gguf_tokenizer` and call `insert_f32_tokenizer_metadata()`. | PMAT-154 |
| **P0: AprTransformer reads Q4K bytes as F32** | `try_load_llama_style()` in realizar loaded quantized APR files through the F32-only `AprTransformer` path, producing garbage output. Fix: `has_quantized_tensors_apr()` check skips `AprTransformer` for Q4K/Q6K APR files, routing to `OwnedQuantizedModel` (keeps weights quantized). | PMAT-156 |
| **Conditional no_gpu: APR only** | wgpu shader panicked on `-inf` literal for APR. Initially set `no_gpu: true` globally — but this forced GGUF to CPU too (minutes per response, unusable). Fixed: `no_gpu` only for `.apr` extension. GGUF gets GPU when available. | PMAT-156, PMAT-158 |
| **Tool call parser: unclosed tags + markdown blocks** | Qwen2.5-Coder 1.5B emits tool calls as: (a) `<tool_call>` without `</tool_call>`, or (b) ` ```json ` markdown blocks. Parser now handles all three formats: XML with closing tag, XML without, and markdown code blocks. Only JSON with `"name"` field is treated as tool call. | PMAT-158 |
| **GGUF CPU-only is unusable** | Without `cuda` feature, GGUF inference dequantizes to F32 on CPU — takes minutes for a single response. Batuta doesn't enable `realizar/cuda`. **Open issue:** need either `cuda` feature flag or `apr serve` HTTP backend for interactive use. | PMAT-158 |
| **APR on crates.io still broken** | `realizar 0.8.3` on crates.io lacks PMAT-156 fix (`has_quantized_tensors_apr`). APR Q4K inference only works with local path dep. Needs `realizar 0.8.4` publish. | PMAT-157 |
| **Validation markers didn't match real APR format** | Batuta's `validate_apr_header` checked for `"\"merges\""` (literal `"merges"` with quotes) and `tokenizer_vocab`. Real APR uses `tokenizer.merges` and `tokenizer.vocabulary`. Fixed markers. | PMAT-154 |
| **Missing contract: APR tokenizer at write time** | `model-format-conversion-v1.yaml` had no equation for tokenizer embedding. Added `apr_tokenizer_embedding` equation + FALSIFY-CONV-007 test. | PMAT-154 |
| **APR tokenizer validation false positive** | `validate_apr_header` matched `"vocab_size"` as containing `"vocab"`. Tightened to require `tokenizer.merges`, `tokenizer.vocabulary`, or `tokenizer.ggml`. | PMAT-150 |
| **APR-preferred discovery hits dead end** | `discover_model()` selected broken APR over valid GGUF. Added Jidoka validation at discovery: invalid APR deprioritized so GGUF wins. | PMAT-150 |
| **No UX for GGUF fallback** | When APR is skipped, user saw no explanation. Added warning with `apr convert` instructions. | PMAT-150 |
| **`validate_model_file` read entire file** | Used `std::fs::read` on 1.1GB APR file to get 64KB header. Fixed to `File::take(65536)`. | PMAT-150 |
| **TinyLlama echoes system prompt** | Chat model (not coding model) regurgitates instructions. Output sanitization catches some markers but model lacks tool-use ability entirely. Not a bug — expected with non-coding models. | — |
| **AprServeDriver always passed `--gpu`** | Even for APR files where wgpu shader panics on `-inf`. Fixed: conditional GPU flag — GGUF gets `--gpu`, APR skips it. | PMAT-164 |
| **Debug `[PMAT-160]` output in production** | `AprServeDriver::launch()` and `wait_for_ready()` emitted debug-prefixed eprintln. Cleaned to user-facing messages. | PMAT-164 |
| **`cmd_code` in binary crate, not library** | `cli/code.rs` was `mod cli` in `main.rs` — inaccessible to apr-cli. Refactored to `agent/code.rs` in library crate. Binary `cli/code.rs` is now a thin wrapper. | PMAT-162 |
| **Shell fallback for pmat query** | Agent used `shell: pmat query "..."` instead of a dedicated tool. Structured `PmatQueryTool` now provides semantic/regex/literal search with grade, complexity, fault filters. | PMAT-163 |
| **Pre-existing clippy: `unwrap()` in tool call parser** | `parse_tool_calls()` used `parsed["name"].as_str().unwrap()` after `is_some()` check. Refactored to `if let Some(name)` pattern. | PMAT-164 |

### 14.2 What Would Disprove This Specification

1. **Qwen2.5-Coder 1.5B fails >60% of coding tasks.** If the default model can't complete basic file edits, variable renames, and test fixes, `apr code` needs a larger default (7B+) or a better prompt engineering strategy for tool-use. (Check: benchmark on SWE-bench-lite subset with Qwen2.5-Coder 1.5B and 7B)

2. **presentar-terminal TUI adds >50ms input latency.** If the 6-panel TUI slows down interactive typing, users will disable it. The TUI must be zero-cost when no streaming is active. (Check: measure keystroke-to-echo latency with TUI on vs off)

3. **batuta dependency makes apr binary >50MB.** If the `code` feature flag bloats `apr` from ~15MB to >50MB, it should be a separate binary (`apr-code`) instead of a subcommand. (Check: measure binary size with and without `code` feature)

4. **trueno-rag indexing takes >10s for medium projects.** If initial indexing blocks the user for >10s on a 500-file Rust project, indexing must be async/incremental. (Check: benchmark on batuta, realizar, trueno repos)
