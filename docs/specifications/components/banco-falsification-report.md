# Banco Comprehensive Falsification Report

> Date: 2026-03-21
> Methodology: Attempt to break every claim in banco-spec.md against source code
> Verdict: **API excellent. Everything else has critical gaps.**

---

## What Is Real (Verified)

| Component | Claim | Verified | Evidence |
|-----------|-------|----------|----------|
| HTTP API | 82 endpoints, 4 protocols | **TRUE** | `grep .route() router.rs` = 82 method bindings |
| Tests L1 | 353 passing | **TRUE** | `cargo test --lib banco` = 353 passed |
| Stack crates | 10 wired | **TRUE** | realizar, aprender, entrenar, alimentar, trueno-rag, pacha, whisper-apr, pforge/MCP, trueno (2) |
| Clippy | Zero warnings | **TRUE** | `cargo clippy -- -D warnings` clean |
| Files | All < 500 lines | **TRUE** | max 476 (inference.rs) |
| SATD | Zero markers | **TRUE** | `grep TODO/FIXME` = 0 in prod code |
| BPE decode | Fixed | **TRUE** | `decode_bpe_text()` converts U+01XX → bytes |
| Persistence | Conversations, files, experiments, audit | **TRUE** | `~/.banco/` with disk reload |
| Graceful shutdown | Ctrl+C/SIGTERM | **TRUE** | `shutdown_signal()` in server.rs |

---

## What Is Broken (Falsified)

### 1. APR Native Format — DEAD PATH

| Claim | Reality | Evidence |
|-------|---------|---------|
| "APR v2 preferred" (spec line 116) | APR files cannot run inference in banco | `extract_gguf_metadata()` returns `None` for non-GGUF |
| `batuta serve --model ./model.apr` | Loads metadata only, inference returns `None` | `model_slot.rs` never calls `OwnedQuantizedModel::from_apr()` |
| Export format `apr` in training | Export path generated but no APR writer wired | `handlers_train.rs` produces path string only |
| APR in merge output | Merge UI shows APR option but no APR serializer | `handlers_merge.rs` — format string only |

**The stack HAS APR support** (realizar `from_apr()`, aprender `apr_import/export/convert()`, apr-cli). **Banco does not wire it.** A user who converts their model to APR v2 format cannot use it with Banco.

### 2. Browser UI — CHAT BUBBLE, NOT AI STUDIO

| Claim | Reality |
|-------|---------|
| 7 screens (Chat, Arena, Models, Data, Training, Experiments, System) | 1 screen (Chat only) |
| 80+ presentar widgets | 0 presentar widgets, 41 lines inline JavaScript |
| YAML-driven layout (`banco-wasm.prs`) | No `.prs` files exist |
| Theming (`~/.banco/theme.yaml`) | Hardcoded CSS variables |
| WCAG AA accessibility | Zero ARIA roles, zero keyboard nav beyond Enter |
| TUI dashboard (`batuta banco --tui`) | Does not exist |
| presentar compiled to wasm32 | No WASM binary built |

See `banco-ux-falsification.md` for full 50+ row breakdown.

### 3. Testing — L1 ONLY, L2-L4 MISSING

| Level | Spec Claims | Exists | What's Missing |
|-------|-------------|--------|----------------|
| L1 Unit | tower::oneshot | 353 tests | None — but can't catch browser/TCP bugs |
| L2 API | Real TCP + probar LlmClient | 0 tests | Server binding, CORS, SSE parsing |
| L3 TUI | probar MockTty + FrameAssertion | 0 tests | No TUI exists to test |
| L4 Browser | probar CDP + Locators | 0 tests | Would have caught "nothing works" |
| Load | probar loadtest | 0 tests | Performance unknown |
| Fuzz | probar InputFuzzer | 0 tests | Crash robustness unknown |
| a11y | probar AccessibilityValidator | 0 tests | WCAG compliance unknown |
| Visual | probar VisualRegressionTester | 0 tests | No baselines |

### 4. Zero-JavaScript Policy — VIOLATED

| Claim | Reality |
|-------|---------|
| "All UI is Rust" | `ui.rs` has 41 lines of inline `<script>` JavaScript |
| "No .js files" | True — but inline JS in a Rust string literal is still JS |
| presentar WASM | Zero presentar integration |
| probar for browser tests | Zero probar tests written |

### 5. Tokenizer — GREEDY MISMATCH

| Claim | Reality |
|-------|---------|
| "encode_prompt" tokenizes for model | Greedy longest-match, not BPE merge rules |
| Works with Qwen2 vocabulary | Produces garbled prompts — model receives wrong token IDs |
| Chat template applied | Template applied to text, but tokenization after is wrong |

The BPE byte *decode* was fixed. The BPE *encode* (prompt → token IDs) is still greedy longest-match, not proper BPE merge. This means the model receives approximate token sequences, not exact ones. Output quality suffers.

### 6. Integration Map — Stale Claims

| Crate | Spec Says | Reality |
|-------|-----------|---------|
| aprender "Complete Phase 2b" | tokenize/detokenize use heuristic, not aprender tokenizer | Heuristic only |
| entrenar "Complete Phase 3b" | Creates LoRA config + optimizer but training is simulated | Simulated metrics |
| alimentar "Complete Phase 3b" | CSV validation only, actual Arrow parsing falls back to line parser | Validation only |
| presentar "Scaffold only" | Honest — zero presentar | Correct |
| trueno-db "Phase 4" | JSONL persistence used instead | Never wired |
| repartir "Phase 4" | Not wired | Never wired |
| forjar "Phase 4" | Not wired | Never wired |

---

## Reprioritized Work (Honest)

### P0 — Fix What's Broken (blocks real usage)

| # | Item | Why P0 |
|---|------|--------|
| 1 | **Wire APR model loading** in model_slot.rs | APR is our native format — can't use it |
| 2 | **Wire proper BPE tokenizer** from aprender | Greedy tokenizer produces wrong token IDs |
| 3 | **Write probar L2 API tests** (real TCP) | Catch server binding and streaming bugs |
| 4 | **Write probar L4 browser tests** | Catch "nothing works" before user does |
| 5 | **Fix browser UI chat** to actually work reliably | Currently broken for users |

### P1 — Build Real UI (blocks "AI studio" claim)

| # | Item | Why P1 |
|---|------|--------|
| 6 | **Replace inline JS with presentar WASM** | Zero-JS policy violation |
| 7 | **Build Training screen** (loss curve, config) | Training API exists, no UI |
| 8 | **Build Models screen** (load/unload, VRAM) | Model API exists, no UI |
| 9 | **Build Data screen** (upload, recipes, RAG) | Data API exists, no UI |
| 10 | **Build System screen** (metrics, probes) | Metrics API exists, no UI |

### P2 — Production Hardening

| # | Item | Why P2 |
|---|------|--------|
| 11 | **Wire real entrenar training** (not simulated) | Training metrics are fake |
| 12 | **Wire alimentar Arrow parsing** (not fallback) | CSV parsing uses line splitter |
| 13 | **Build TUI dashboard** via presentar-terminal | Spec promises, doesn't exist |
| 14 | **Probar load testing** | Performance unknown |
| 15 | **Probar visual regression baselines** | No visual QA |

### P3 — Remaining Stack Integration

| # | Item |
|---|------|
| 16 | trueno-db for experiment persistence (replace JSONL) |
| 17 | repartir for multi-GPU training |
| 18 | forjar for IaC deployment |
| 19 | Accessibility (ARIA, keyboard nav, WCAG AA) |
| 20 | Theming (YAML config, light/dark) |
