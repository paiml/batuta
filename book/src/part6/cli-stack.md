# `batuta stack`

PAIML Stack dependency orchestration commands.

## Synopsis

```bash
batuta stack <COMMAND>
```

## Commands

| Command | Description |
|---------|-------------|
| `check` | Check dependency health across the PAIML stack |
| `gate` | Enforce A- quality threshold for all components |
| `quality` | Analyze quality metrics across the PAIML stack |
| `release` | Coordinate releases across the PAIML stack |
| `status` | Show stack health status dashboard |
| `sync` | Synchronize dependencies across the stack |
| `tree` | Display hierarchical tree of PAIML stack components |

---

## `batuta stack tree`

Display a visual hierarchical tree of all 21 PAIML stack components.

### Usage

```bash
batuta stack tree [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--format <FORMAT>` | Output format: `ascii` (default), `json`, `dot` |
| `--health` | Show health status and version information |
| `--filter <LAYER>` | Filter by layer name |

### Layers

| Layer | Components |
|-------|------------|
| `core` | trueno, trueno-viz, trueno-db, trueno-graph, trueno-rag |
| `ml` | aprender, aprender-shell, aprender-tsp |
| `inference` | realizar, renacer, alimentar, entrenar |
| `orchestration` | batuta, certeza, presentar, pacha |
| `distributed` | repartir |
| `transpilation` | ruchy, decy, depyler |
| `docs` | sovereign-ai-stack-book |

### Examples

```bash
# ASCII tree (default)
batuta stack tree

# Output:
# PAIML Stack (21 crates)
# ├── core
# │   ├── trueno
# │   ├── trueno-viz
# │   └── ...
# ├── ml
# │   └── ...

# JSON output for tooling
batuta stack tree --format json

# Graphviz DOT for visualization
batuta stack tree --format dot | dot -Tpng -o stack.png

# Filter to specific layer
batuta stack tree --filter core

# Show health status
batuta stack tree --health
```

---

## `batuta stack check`

Analyze dependency health across the PAIML ecosystem.

### Usage

```bash
batuta stack check [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--project <NAME>` | Specific project to check (default: all) |
| `--format <FORMAT>` | Output format: `text`, `json`, `markdown` |
| `--strict` | Fail on any warnings |
| `--verify-published` | Verify crates.io versions exist |
| `--workspace <PATH>` | Path to workspace root |

### Examples

```bash
# Check all projects
batuta stack check

# Check specific project with strict mode
batuta stack check --project trueno --strict

# JSON output for CI
batuta stack check --format json --verify-published
```

---

## `batuta stack gate`

Enforce A- quality threshold across all PAIML stack components. This command is designed for CI/CD pipelines and pre-commit hooks to block releases or commits when any component falls below the quality threshold.

### Usage

```bash
batuta stack gate [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--workspace <PATH>` | Path to workspace root (default: parent of current directory) |
| `--quiet, -q` | Quiet mode - only output on failure |

### Quality Threshold

The quality gate enforces an **A- minimum** (SQI ≥ 85) for all stack components. Components below this threshold are **blocked** and will cause the gate to fail.

| Grade | SQI Range | Gate Status |
|-------|-----------|-------------|
| A+ | 95-100% | PASS |
| A | 90-94% | PASS |
| A- | 85-89% | PASS |
| B+ | 80-84% | BLOCKED |
| B | 70-79% | BLOCKED |
| C | 60-69% | BLOCKED |
| D | 50-59% | BLOCKED |
| F | 0-49% | BLOCKED |

### Enforcement Points

The quality gate is enforced at multiple points in the development workflow:

| Point | Trigger | Action |
|-------|---------|--------|
| Pre-commit | `git push` | Blocks push if any component < A- |
| Release | `batuta stack release` | Blocks release by default (use `--no-verify` to skip) |
| CI Pipeline | Pull request | Blocks PR merge if quality gate fails |
| Manual | `make stack-gate` | Returns exit code 1 if failed |

### Examples

```bash
# Run quality gate check
batuta stack gate

# Output:
# ╔════════════════════════════════════════════════════╗
# ║  Stack Quality Gate - A- Enforcement               ║
# ╚════════════════════════════════════════════════════╝
#
# trueno           SQI: 95.9  Grade: A+  ✅ PASS
# aprender         SQI: 96.2  Grade: A+  ✅ PASS
# batuta           SQI: 94.1  Grade: A   ✅ PASS
# ...
#
# ✅ All 21 components meet A- quality threshold

# Quiet mode for CI (only outputs on failure)
batuta stack gate --quiet

# Check specific workspace
batuta stack gate --workspace /path/to/paiml
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All components pass the quality gate |
| 1 | One or more components are below A- threshold |

### Pre-commit Hook Configuration

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: stack-quality-gate
      name: Stack Quality Gate (A- enforcement)
      entry: cargo run --quiet -- stack gate
      language: system
      pass_filenames: false
      stages: [push]
```

### Makefile Targets

```makefile
stack-gate:  ## Quality gate enforcement
	@cargo run --quiet -- stack gate

stack-quality:  ## Show detailed quality matrix
	@cargo run --quiet -- stack quality
```

---

## `batuta stack quality`

Analyze quality metrics across the PAIML stack using PMAT integration.

This command evaluates each stack component against the Stack Quality Matrix, which includes:
- **Rust Project Score** (0-114): Code quality, testing, documentation
- **Repository Score** (0-110): CI/CD, security, community health
- **README Score** (0-20): Documentation completeness
- **Hero Image**: Visual branding presence

### Usage

```bash
batuta stack quality [OPTIONS] [COMPONENT]
```

### Options

| Option | Description |
|--------|-------------|
| `--strict` | Require A+ grade for all components |
| `--format <FORMAT>` | Output format: `text` (default), `json` |
| `--verify-hero` | Verify hero image exists and meets requirements |
| `--verbose` | Show detailed scoring breakdown |
| `--workspace <PATH>` | Path to workspace root |

### Quality Grades

| Grade | SQI Range | Description |
|-------|-----------|-------------|
| A+ | 95-100% | Exceptional quality |
| A | 90-94% | Excellent quality |
| A- | 85-89% | Very good quality |
| B+ | 80-84% | Good quality |
| B | 70-79% | Acceptable quality |
| C | 60-69% | Needs improvement |
| D | 50-59% | Poor quality |
| F | 0-49% | Failing quality |

### Stack Quality Index (SQI)

The SQI is calculated as a weighted composite:

```
SQI = 0.40 × Rust Score + 0.30 × Repo Score + 0.20 × README Score + 0.10 × Hero Score
```

### Examples

```bash
# Check quality of all stack components
batuta stack quality

# Output:
# Stack Quality Report
# ====================
#
# trueno          A+  (SQI: 97.2%)
# aprender        A   (SQI: 92.1%)
# batuta          A+  (SQI: 96.8%)
# ...
#
# Summary: 18/25 components at A+ grade
# Overall Stack Grade: A

# Check specific component with verbose output
batuta stack quality trueno --verbose

# Strict mode for CI (fails if any component below A+)
batuta stack quality --strict

# JSON output for tooling
batuta stack quality --format json

# Verify hero images exist
batuta stack quality --verify-hero
```

### Hero Image Requirements

A hero image is required for A+ grade and must be:
- Located at `docs/hero.svg` (preferred) or `docs/hero.png`
- Can also be referenced as first image in README.md
- SVG format preferred for scalability and crisp rendering
- If using PNG: minimum dimensions 1280x640 pixels

---

## `batuta stack release`

Coordinate releases with automatic dependency ordering.

### Usage

```bash
batuta stack release [OPTIONS] [CRATE_NAME]
```

### Options

| Option | Description |
|--------|-------------|
| `--all` | Release all crates with changes |
| `--dry-run` | Show what would be released |
| `--bump <TYPE>` | Version bump: `patch`, `minor`, `major` |
| `--no-verify` | Skip quality gate verification |
| `--yes` | Skip interactive confirmation |
| `--publish` | Publish to crates.io |

### Examples

```bash
# Dry run to see release plan
batuta stack release --all --dry-run

# Release specific crate (and its dependencies)
batuta stack release trueno --bump patch

# Full release with publish
batuta stack release --all --bump minor --publish --yes
```

---

## `batuta stack status`

Show health dashboard for the entire stack.

### Usage

```bash
batuta stack status [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--simple` | Simple text output (no TUI) |
| `--format <FORMAT>` | Output format: `text`, `json`, `markdown` |
| `--tree` | Show dependency tree |

---

## `batuta stack sync`

Synchronize dependency versions across the stack.

### Usage

```bash
batuta stack sync [OPTIONS] [CRATE_NAME]
```

### Options

| Option | Description |
|--------|-------------|
| `--all` | Sync all crates |
| `--dry-run` | Show what would change |
| `--align <DEP=VER>` | Align specific dependency version |

### Examples

```bash
# Sync all crates
batuta stack sync --all --dry-run

# Align arrow version across stack
batuta stack sync --all --align "arrow=54.0"
```

---

## Toyota Way Principles

The stack commands embody Toyota Way principles:

| Principle | Implementation |
|-----------|----------------|
| **Jidoka** | Pre-flight checks stop broken releases |
| **Just-in-Time** | Pull-based release ordering |
| **Heijunka** | Version alignment across stack |
| **Genchi Genbutsu** | Real-time crates.io verification |
| **Visual Management** | Tree view with health indicators |
