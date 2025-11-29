# Support Tools

The Sovereign AI Stack includes essential support tools for scripting, quality analysis, and system tracing. These tools integrate with Batuta's orchestration workflow.

## Tool Overview

| Tool | Purpose | Integration Point |
|------|---------|-------------------|
| **Ruchy** | Rust scripting language | Embedded scripting, automation |
| **PMAT** | Quality analysis (TDG scoring) | Phase 1: Analysis, CI/CD gates |
| **Renacer** | Syscall tracing | Phase 4: Validation |

## Ruchy: Rust Scripting

Ruchy provides a scripting language that compiles to Rust, enabling:

- **Automation scripts:** Build, deployment, data processing
- **Embedded scripting:** In Presentar apps (Section 8)
- **REPL development:** Interactive exploration

```ruchy
// Ruchy script for data processing
let data = load_dataset("transactions")
let filtered = data.filter(|row| row.amount > 100)
let aggregated = filtered.group_by("category").sum("amount")
save_dataset(aggregated, "output.ald")
```

**Security (in Presentar):**
- Max 1M instructions per script
- Max 16MB memory allocation
- 10ms time slices (cooperative yielding)

## PMAT: Quality Analysis

PMAT computes Technical Debt Grade (TDG) scores for projects:

- **0-100 scale:** F, D, C-, C, C+, B-, B, B+, A-, A, A+
- **Multi-language:** Rust, Python, C/C++, Shell
- **Metrics:** Complexity, coverage, duplication, dependencies

```bash
# Analyze a project
pmat analyze ./myproject --output report.json

# CI gate (fail if below B+)
pmat gate ./myproject --min-grade B+
```

**Integration with Batuta:**
- Phase 1 (Analysis): Initial TDG assessment
- Phase 4 (Validation): Post-transpilation quality check
- CI/CD: Gate enforcement

## Renacer: Syscall Tracing

Renacer captures system call traces for validation:

- **Deterministic replay:** Ensures transpiled code matches original behavior
- **Golden trace comparison:** Baseline vs current
- **Cross-platform:** Linux, macOS, Windows

```bash
# Capture baseline trace
renacer capture ./original_binary -- args > baseline.trace

# Compare against transpiled
renacer compare baseline.trace ./transpiled_binary -- args
```

**Integration with Batuta:**
- Phase 4 (Validation): Behavioral equivalence testing

## Additional Support Tools

### Trueno-RAG (v0.1.0)

Retrieval-Augmented Generation pipeline built on Trueno:

- Vector similarity search
- Document chunking
- Embedding generation

### Trueno-Graph

Graph data structures and algorithms:

- Property graphs
- Traversal operations
- Connected component analysis

### Trueno-DB

Embedded database with Trueno compute:

- Column-store backend
- SQL-like query interface
- ACID transactions

## Tool Ecosystem Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    Batuta (Orchestration)                       │
├─────────────────────────────────────────────────────────────────┤
│  Transpilers          │  Support Tools      │  Data/ML         │
│  ├── Depyler          │  ├── Ruchy          │  ├── Alimentar   │
│  ├── Decy             │  ├── PMAT           │  ├── Aprender    │
│  └── Bashrs           │  └── Renacer        │  └── Realizar    │
├─────────────────────────────────────────────────────────────────┤
│  Visualization        │  Extensions         │  Registry        │
│  ├── Trueno-Viz       │  ├── Trueno-RAG     │  └── Pacha       │
│  └── Presentar        │  ├── Trueno-Graph   │                  │
│                       │  └── Trueno-DB      │                  │
└─────────────────────────────────────────────────────────────────┘
```

## Further Reading

- [Ruchy: Rust Scripting](./ruchy.md)
- [PMAT: Quality Analysis](./pmat.md)
- [Renacer: Syscall Tracing](./renacer.md)

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Foundation Libraries](./foundation-libs.md)
