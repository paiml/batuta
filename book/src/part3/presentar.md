# Presentar: Sovereign AI Visualization & App Framework

**Version:** 0.1.0 | **Status:** Specification Complete | **Spec:** [ui-viz-spec.md](https://github.com/paiml/presentar/docs/specifications/ui-viz-spec.md)

Presentar is a **PURE WASM** visualization and rapid application framework built entirely on Sovereign AI Stack primitives. It replaces Streamlit, Gradio, and Panel with 60fps GPU-accelerated rendering, compile-time type safety, and deterministic reproducibility.

## Position in the Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  Presentar (Visualization & Apps)           ← YOU ARE HERE     │
├─────────────────────────────────────────────────────────────────┤
│  Trueno-Viz (GPU Rendering Primitives)                         │
├─────────────────────────────────────────────────────────────────┤
│  Trueno (SIMD/GPU Compute) v0.7.3                               │
├─────────────────────────────────────────────────────────────────┤
│  Aprender (ML) | Realizar (Inference) | Alimentar (Data)       │
└─────────────────────────────────────────────────────────────────┘
```

## Core Principles

| Principle | Implementation |
|-----------|----------------|
| **80% Pure Stack** | All rendering via `trueno-viz` GPU primitives |
| **20% Minimal External** | Only `winit` (windowing) + `fontdue` (fonts) |
| **WASM-First** | Browser deployment without server dependencies |
| **YAML-Driven** | Declarative app configuration |
| **Graded Quality** | Every app receives F-A score via TDG metrics |

## Auto-Display: Convention Over Configuration

Presentar auto-generates UIs from Sovereign AI Stack file formats:

| File Type | Generated UI |
|-----------|--------------|
| `.apr` (Aprender model) | ModelCard + inference panel |
| `.ald` (Alimentar dataset) | DataCard + DataTable |
| `app.yaml` | Custom layout from YAML |
| Mixed `.apr`/`.ald` | Split-view grid |

```bash
# Point at a directory, get an app
presentar --serve ./fraud-detector/

# Bundle for deployment
presentar --bundle ./fraud-detector/ -o app.wasm
```

## YAML App Configuration

```yaml
presentar: "0.1"
name: "fraud-detection-dashboard"
version: "1.0.0"

# Data sources (Alimentar .ald files)
data:
  transactions:
    source: "pacha://datasets/transactions:latest"
    format: "ald"
    refresh: "5m"

# Model references (Aprender .apr files)
models:
  fraud_detector:
    source: "pacha://models/fraud-detector:1.2.0"
    format: "apr"

# Layout definition (12-column responsive grid)
layout:
  type: "dashboard"
  columns: 12
  sections:
    - id: "metrics"
      span: [1, 4]
      widgets:
        - type: "metric"
          label: "Fraud Rate"
          value: "{{ data.predictions | filter(fraud=true) | percentage }}"

    - id: "main-chart"
      span: [5, 12]
      widgets:
        - type: "chart"
          chart_type: "line"
          data: "{{ data.transactions }}"
          x: "timestamp"
          y: "amount"
```

## Quality Scoring

Every Presentar app receives a TDG score (0-100, F-A):

| Category | Weight | Metrics |
|----------|--------|---------|
| Structural | 25 | Widget complexity, layout depth |
| Performance | 20 | Frame time, memory, bundle size |
| Accessibility | 20 | WCAG AA, keyboard nav, ARIA |
| Data Quality | 15 | Completeness, freshness, schema |
| Documentation | 10 | Manifest, model/data cards |
| Consistency | 10 | Theme adherence, naming |

## Integration with Batuta Workflow

Presentar apps integrate with Batuta's 5-phase workflow:

```
Phase 1: Analysis    → presentar analyze app.yaml
Phase 2: Transpile   → (N/A - pure Rust)
Phase 3: Optimize    → presentar optimize --wasm-opt
Phase 4: Validate    → presentar test (zero-dep harness)
Phase 5: Deploy      → presentar --bundle → pacha publish
```

## presentar-test: Zero-Dependency E2E Testing

**Critical constraint:** No playwright, selenium, npm, or C bindings.

```rust
use presentar_test::*;

#[presentar_test]
fn inference_flow() {
    let mut h = Harness::new(include_bytes!("fixtures/app.tar"));
    h.type_text("[data-testid='input-amount']", "1500")
     .click("[data-testid='predict-btn']");
    h.assert_text_contains("[data-testid='result']", "Fraud Score:");
}

#[presentar_test]
fn visual_regression() {
    let mut h = Harness::new(include_bytes!("fixtures/app.tar"));
    Snapshot::assert_match("app-default", h.screenshot("[data-testid='app-root']"), 0.001);
}
```

**Determinism guarantees:**
- Fixed DPI: 1.0
- Font antialiasing: Grayscale only
- Fixed viewport: 1280x720
- Embedded test font (Inter)

## Trueno-Viz GPU Primitives

Presentar renders via Trueno-Viz draw commands:

```rust
pub enum DrawCommand {
    Path { points: Vec<Point>, closed: bool, style: StrokeStyle },
    Fill { path: PathRef, color: Color, rule: FillRule },
    Rect { bounds: Rect, radius: CornerRadius, style: BoxStyle },
    Text { content: String, position: Point, style: TextStyle },
    Image { tensor: TensorRef, bounds: Rect, sampling: Sampling },
}
```

**Anti-aliasing strategy:**
- Hardware MSAA (4x) for fills
- Analytical AA for lines/curves
- SDF for text rendering

## Pacha Registry Integration

```yaml
# Fetch models and datasets from Pacha
models:
  classifier:
    source: "pacha://models/mnist-cnn:1.0.0"

data:
  training:
    source: "pacha://datasets/mnist:latest"
```

Lineage tracking follows W3C PROV-DM for full provenance.

## Performance Targets

| Operation | Target | Backend |
|-----------|--------|---------|
| Path tessellation (1K points) | <1ms | Trueno SIMD |
| Fill rendering (10K triangles) | <2ms | WebGPU |
| Full frame (complex dashboard) | <16ms | 60fps |
| Bundle size | <500KB | WASM |

## Ruchy Script Integration (Future)

Embedded scripting for dynamic behavior:

```yaml
scripts:
  on_load: |
    let data = load_dataset("transactions")
    let filtered = data.filter(|row| row.amount > 100)
    set_state("filtered_data", filtered)
```

**Security:** Resource limits (1M instructions, 16MB memory, 10ms slice) prevent DoS.

## Comparison with Alternatives

| Feature | Presentar | Streamlit | Gradio |
|---------|-----------|-----------|--------|
| Runtime | WASM (no server) | Python | Python |
| Performance | 60fps GPU | ~10fps | ~10fps |
| Type Safety | Compile-time | Runtime | Runtime |
| Bundle Size | <500KB | ~50MB | ~30MB |
| Testing | Zero-dep harness | Manual | Manual |
| Reproducibility | Deterministic | Non-deterministic | Non-deterministic |

## Academic Foundation

Key references (see full spec for 30+ citations):
- Czaplicki (2012): Elm Architecture
- Haas et al. (2017): WebAssembly performance model
- Mitchell et al. (2019): Model Cards
- Ohno (1988): Toyota Production System (Jidoka)

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Trueno-Viz](./trueno-viz.md) | [Trueno](./trueno.md)
