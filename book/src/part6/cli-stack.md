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
