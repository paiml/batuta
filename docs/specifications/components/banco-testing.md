# Banco Testing Strategy: Probar Integration

> Parent: [banco-spec.md](banco-spec.md) §5
> Depends on: jugar-probar (1.0.x), probador CLI (1.0.x)

---

## Principle

Every Banco surface (API, TUI, WASM) is tested with the probar framework. Probar provides LLM-specific assertions, TUI frame verification, browser automation, visual regression, accessibility auditing, load testing, and fuzzing — all pure Rust, zero JavaScript.

```
┌──────────────────────────────────────────────────────┐
│                    Banco Test Pyramid                 │
├──────────────────────────────────────────────────────┤
│  L4: E2E Browser     probar browser + locators       │
│  L3: E2E TUI         probar tui + frame assertions   │
│  L2: API Integration  probar llm + websocket         │
│  L1: Unit             cargo test (tower::oneshot)     │
├──────────────────────────────────────────────────────┤
│  Cross-cutting: load, fuzz, a11y, visual regression  │
└──────────────────────────────────────────────────────┘
```

---

## L1: Unit Tests (cargo test, existing)

Uses `tower::ServiceExt::oneshot()` — no TCP, no probar needed.

| Suite | Count | What |
|-------|-------|------|
| BANCO_TYP | 12 | Type serde roundtrip |
| BANCO_STA | 8 | State init, health, models |
| BANCO_MID | 5 | Privacy middleware |
| BANCO_HDL | 7 | Handler routing via oneshot |
| P0/P1/P2 | 35 | Cross-cutting endpoint tests |
| CONV | 15 | Conversation CRUD + export/import |
| INF | 24 | Inference engine + integration tests |
| STOR | 12 | File storage + data endpoints |
| RECIPE | 12 | Recipe pipeline + endpoint tests |
| RAG | 12 | RAG index + search + chat integration |
| EVAL/TRAIN | 12 | Eval perplexity + training runs |
| EXP | 7 | Experiment tracking + comparison |
| BATCH | 5 | Batch inference |
| CONTRACT | 20 | Falsification tests |
| MODEL_SLOT | 8 | Model load/unload |
| **Total** | **214** | **All passing** |

These grow with each phase. 16 test modules across `*_tests.rs` files.

---

## L2: API Integration Tests (probar llm module)

### Setup

```rust
use jugar_probar::llm::{LlmClient, ChatRequest, ChatMessage, LlmAssertion};
use jugar_probar::websocket::{WebSocketConnection, WebSocketMessage};
use jugar_probar::perf::{Tracer, MetricStats};

// Start Banco server in test
let state = BancoStateInner::with_defaults();
let app = create_banco_router(state);
let listener = TcpListener::bind("127.0.0.1:0").await?;
let port = listener.local_addr()?.port();
tokio::spawn(axum::serve(listener, app));
let client = LlmClient::new(&format!("http://127.0.0.1:{port}/api/v1"));
```

### Chat Completion Tests

```rust
#[tokio::test]
async fn test_chat_response_valid() {
    let resp = client.chat(ChatRequest {
        model: "local".into(),
        messages: vec![ChatMessage::user("Hello!")],
        ..Default::default()
    }).await?;

    LlmAssertion::response_valid().assert(&resp)?;
    LlmAssertion::contains("banco").assert(&resp)?;
    LlmAssertion::latency_under(Duration::from_secs(5)).assert(&resp)?;
}
```

### SSE Streaming Tests

```rust
#[tokio::test]
async fn test_streaming_chunks() {
    let stream = client.chat_stream(ChatRequest {
        messages: vec![ChatMessage::user("Hi!")],
        ..Default::default()
    }).await?;

    let chunks: Vec<_> = stream.collect().await;
    assert!(chunks.len() >= 3); // role + content + done
    // Last chunk has finish_reason
    assert!(chunks.last().unwrap().choices[0].finish_reason.is_some());
}
```

### WebSocket Tests (Phase 4)

```rust
#[tokio::test]
async fn test_websocket_training_metrics() {
    let conn = WebSocketConnection::connect(
        &format!("ws://127.0.0.1:{port}/api/v1/ws")
    ).await?;

    conn.send(WebSocketMessage::text(
        r#"{"subscribe": "training_metrics", "run_id": "run-001"}"#,
        MessageDirection::Sent, 0
    )).await?;

    let msg = conn.receive_timeout(Duration::from_secs(5)).await?;
    let data: serde_json::Value = msg.json()?;
    assert!(data["step"].is_number());
    assert!(data["loss"].is_number());
}
```

### Network Interception

```rust
#[tokio::test]
async fn test_sovereign_blocks_external() {
    // Verify no outbound network calls in Sovereign mode
    let state = BancoStateInner::with_privacy(PrivacyTier::Sovereign);
    // ... setup server with network capture
    // Assert zero external requests made
}
```

---

## L3: TUI Integration Tests (probar tui module)

### Frame Assertions

```rust
use jugar_probar::tui::{TuiTestBackend, FrameAssertion, MockTty};

#[test]
fn test_chat_tui_renders_messages() {
    let mut tty = MockTty::new(120, 40);
    let app = BancoTuiApp::new(test_state());
    app.render(&mut tty);

    let frame = tty.capture_frame();
    FrameAssertion::new(&frame)
        .to_have_text("Chat")?
        .to_have_text("Models")?
        .to_have_text("System")?;
}

#[test]
fn test_training_dashboard_shows_loss() {
    let mut tty = MockTty::new(120, 40);
    let app = BancoTuiApp::new(state_with_training_run());
    app.render(&mut tty);

    let frame = tty.capture_frame();
    FrameAssertion::new(&frame)
        .to_have_text("Loss")?
        .to_have_text("GPU")?
        .to_have_text("ETA")?;
}
```

### TUI Load Testing

```rust
use jugar_probar::tui_load::{TuiLoadTest, TuiFrameMetrics};

#[test]
fn test_chat_tui_60fps() {
    let load = TuiLoadTest {
        item_count: 1000,       // 1000 messages in history
        frame_budget_ms: 16.6,  // 60fps
        timeout_ms: 5000,
    };

    let metrics: TuiFrameMetrics = load.run(|tty| {
        let app = BancoTuiApp::new(state_with_messages(1000));
        app.render(tty);
    })?;

    assert!(metrics.p95_frame_ms() < 16.6, "p95 frame time exceeds 60fps budget");
    assert!(metrics.p99_frame_ms() < 33.3, "p99 frame time exceeds 30fps floor");
}
```

### Playbook: State Machine Testing

```yaml
# banco-chat-playbook.yaml
version: "1.0"
machine:
  id: banco-chat
  initial: idle
  states:
    idle:
      invariants:
        - element_exists: "input-field"
        - text_contains: { selector: "status-bar", text: "ok" }
    typing:
      invariants:
        - element_exists: "send-button"
    streaming:
      invariants:
        - text_contains: { selector: "response-area", text: "" }
    complete:
      invariants:
        - element_exists: "response-area"
  transitions:
    - { from: idle, to: typing, event: key_press, assertions: [{ element_exists: "input-field" }] }
    - { from: typing, to: streaming, event: submit }
    - { from: streaming, to: complete, event: stream_done }
    - { from: complete, to: typing, event: key_press }
```

```rust
use jugar_probar::playbook::{Playbook, PlaybookExecutor};

#[test]
fn test_chat_state_machine() {
    let playbook = Playbook::from_yaml(include_str!("banco-chat-playbook.yaml"))?;
    let mut executor = PlaybookExecutor::new(playbook);
    let result = executor.run(&mut BancoTuiApp::new(test_state()))?;
    assert!(result.passed);
}
```

### Presentar Config Validation

```rust
use jugar_probar::presentar::{validate_config, generate_falsification_playbook};

#[test]
fn test_banco_tui_config_valid() {
    let config = PresentarConfig::from_yaml(include_str!("banco-tui.prs"))?;
    let result = validate_config(&config);
    assert!(result.is_valid());
}

#[test]
fn test_banco_tui_falsification_protocol() {
    let config = PresentarConfig::from_yaml(include_str!("banco-tui.prs"))?;
    let checks = generate_falsification_playbook(&config);
    assert!(checks.len() >= 100); // F001-F100 protocol
    for check in checks {
        assert!(check.run().passed);
    }
}
```

---

## L4: E2E Browser Tests (probar browser + locators)

### Setup

```rust
use jugar_probar::browser::{BrowserConfig, Page};
use jugar_probar::locator::{Selector, LocatorOptions};

async fn banco_page() -> Page {
    let config = BrowserConfig {
        headless: true,
        viewport_width: 1280,
        viewport_height: 720,
        ..Default::default()
    };
    let page = Page::new(config).await?;
    page.goto(&format!("http://localhost:{PORT}/")).await?;
    page
}
```

### Chat UI Tests

```rust
#[tokio::test]
async fn test_chat_send_message() {
    let page = banco_page().await;
    page.locator(Selector::Placeholder("Type a message...")).await?.fill("Hello!").await?;
    page.locator(Selector::Role { role: "button", name: Some("Send") }).click().await?;
    let resp = page.locator(Selector::TestId("assistant-message"))
        .with_options(LocatorOptions { timeout: Duration::from_secs(30), ..Default::default() }).await?;
    assert!(resp.text_content().await?.contains("banco"));
}
```

### Navigation: Tab through all 7 screens via `Selector::Role { role: "tab", name }`, verify each panel loads.

---

## Cross-Cutting: Accessibility (probar accessibility)

```rust
use jugar_probar::accessibility::{AccessibilityValidator, AccessibilityConfig};

#[tokio::test]
async fn test_wcag_aa_compliance() {
    let page = banco_page().await;
    let validator = AccessibilityValidator::new(AccessibilityConfig {
        check_contrast: true, check_focus: true, check_keyboard: true,
        min_contrast_text: 4.5, min_contrast_ui: 3.0, ..Default::default()
    });
    let report = validator.audit(&page).await?;
    assert!(report.passes_wcag_aa, "Failing: {:?}", report.violations);
}
```

Keyboard navigation test: Tab through 50 elements, assert chat-input, send-button, model-selector all reachable.

---

## Cross-Cutting: Visual Regression (probar visual_regression)

```rust
use jugar_probar::visual_regression::{VisualRegressionTester, VisualRegressionConfig};

#[tokio::test]
async fn test_chat_screen_visual() {
    let page = banco_page().await;
    let screenshot = page.screenshot().await?;

    let tester = VisualRegressionTester::new(VisualRegressionConfig {
        threshold: 1.0,           // 1% pixel diff allowed
        baseline_dir: "tests/baselines/".into(),
        diff_dir: "tests/diffs/".into(),
        ..Default::default()
    });

    let result = tester.compare_images(&screenshot, &baseline("chat-screen.png"))?;
    assert!(result.matches, "Diff: {:.2}% ({} pixels)",
        result.diff_percentage, result.diff_pixel_count);
}
```

Update baselines: `PROBAR_UPDATE_BASELINES=1 cargo test`

---

## Cross-Cutting: Load Testing (probar llm loadtest)

```rust
use jugar_probar::llm::loadtest::{LoadTest, LoadTestConfig};

#[tokio::test]
async fn test_banco_under_load() {
    let config = LoadTestConfig {
        concurrent_requests: 10,
        request_rate: 5.0,       // 5 req/s
        duration: Duration::from_secs(30),
        dataset: vec![
            ChatRequest { messages: vec![ChatMessage::user("Hello!")], ..Default::default() },
            ChatRequest { messages: vec![ChatMessage::user("What is 2+2?")], ..Default::default() },
        ],
    };

    let mut test = LoadTest::with_config(config);
    let result = test.run().await?;

    assert_eq!(result.failed, 0);
    assert!(result.latency_stats.p95 < 5000.0, "p95 latency > 5s");
    assert!(result.throughput_rps > 1.0, "throughput below 1 rps");
}
```

### Scorecard

```rust
use jugar_probar::llm::scorecard::{compute_scorecard, format_markdown};

#[tokio::test]
async fn test_banco_scorecard() {
    let result = run_load_test().await?;
    let card = compute_scorecard(&result);
    eprintln!("{}", format_markdown(&card));

    assert!(card.correctness.score >= 0.95);
    assert!(card.latency.score >= 0.80);
}
```

---

## Cross-Cutting: Fuzzing (probar fuzzer)

```rust
use jugar_probar::fuzzer::{InputFuzzer, FuzzerConfig, Seed};

#[test]
fn test_chat_input_fuzz() {
    let mut fuzzer = InputFuzzer::with_config(Seed(42), FuzzerConfig {
        viewport_width: 1280.0,
        viewport_height: 720.0,
        key_probability: 0.8,
        ..Default::default()
    });

    let mut app = BancoTuiApp::new(test_state());
    for _ in 0..10_000 {
        let events = fuzzer.generate_valid_inputs();
        for event in events {
            // Should never panic
            app.handle_input(event);
        }
    }
}
```

---

## Cross-Cutting: Pixel Coverage (probar pixel_coverage)

```rust
use jugar_probar::pixel_coverage::{PixelCoverageTracker, PngHeatmap};

#[tokio::test]
async fn test_ui_coverage_80pct() {
    let mut tracker = PixelCoverageTracker::new((1280, 720), (64, 36));

    // Run through all screens, record interactions
    for tab in ["chat", "arena", "models", "data", "training", "experiments", "system"] {
        navigate_to(tab).await;
        let regions = get_interactive_regions().await;
        for region in regions {
            tracker.record_region(region);
        }
    }

    let report = tracker.generate_report();
    assert!(report.overall_coverage >= 0.80, "UI coverage: {:.1}%", report.overall_coverage * 100.0);

    // Generate heatmap for review
    PngHeatmap::render(&tracker, "tests/output/banco_coverage.png")?;
}
```

---

## Cross-Cutting: Deterministic Replay (probar simulation)

Record a 300-frame TUI session with `SimulationConfig { seed: 42, fps: 30 }`, replay twice, assert identical `final_state_hash`. Catches non-determinism from threading or time-dependent logic.

---

## Phase → Test Level Matrix

| Phase | L1 Unit | L2 API (probar llm) | L3 TUI (probar tui) | L4 Browser (probar browser) |
|-------|---------|---------------------|----------------------|-----------------------------|
| Phase 1 | 32 tests | — | — | — |
| Phase 2 | +20 | Chat, SSE, models, arena | Chat, model mgmt | — |
| Phase 3 | +15 | Training, data, eval, RAG | Training dashboard, data | — |
| Phase 4 | +10 | WebSocket, tools | Agent steps | Full E2E, a11y, visual regression |

### CI Pipeline

```bash
# L1 (every commit, <30s)
cargo test --features banco --lib banco

# L2 (pre-merge, <2min)
cargo test --features banco,testing --test banco_api

# L3 (pre-merge, <2min)
cargo test --features banco-tui,testing --test banco_tui

# L4 (nightly, <10min)
cargo test --features banco-ui,testing --test banco_e2e

# Load (weekly, <5min)
cargo test --features banco,testing --test banco_load -- --ignored

# Fuzz (weekly, <10min)
cargo test --features banco-tui,testing --test banco_fuzz -- --ignored
```

---

## Feature Flags for Testing

```toml
# Cargo.toml [dev-dependencies]
jugar-probar = { version = "1.0", features = ["browser", "llm", "tui", "compute-blocks"] }
```

No probar dependency in production builds. All test code behind `#[cfg(test)]` or in `tests/` directory.
