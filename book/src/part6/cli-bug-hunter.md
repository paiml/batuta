# `batuta bug-hunter`

The `bug-hunter` command provides proactive bug hunting using multiple falsification-driven strategies. It implements Section 11 of the Popperian Falsification Checklist (BH-01 to BH-15).

## Philosophy

> "A theory that explains everything, explains nothing." — Karl Popper

Bug hunting operationalizes falsification: we systematically attempt to break code, not merely verify it works. Each mode represents a different strategy for falsifying the implicit claim "this code is correct."

## Usage

```bash
# LLM-augmented static analysis
batuta bug-hunter analyze .

# SBFL fault localization from coverage data
batuta bug-hunter hunt .

# Mutation-based invariant falsification
batuta bug-hunter falsify .

# Targeted unsafe Rust fuzzing
batuta bug-hunter fuzz .

# Hybrid concolic + SBFL deep analysis
batuta bug-hunter deep-hunt .

# Run all modes and combine results
batuta bug-hunter ensemble .
```

## Modes

### `analyze` - LLM-Augmented Static Analysis (LLIFT Pattern)

Combines traditional static analysis with pattern matching for common defect categories.

```bash
batuta bug-hunter analyze /path/to/project
batuta bug-hunter analyze . --format json
batuta bug-hunter analyze . --min-suspiciousness 0.7
```

### `hunt` - SBFL Without Failing Tests (SBEST Pattern)

Uses Spectrum-Based Fault Localization on coverage data to identify suspicious code regions.

```bash
# Basic hunt with default Ochiai formula
batuta bug-hunter hunt .

# Specify coverage file location
batuta bug-hunter hunt . --coverage ./lcov.info

# Use different SBFL formula
batuta bug-hunter hunt . --formula tarantula
batuta bug-hunter hunt . --formula dstar
```

Coverage file detection searches:
- `./lcov.info` (project root)
- `./target/coverage/lcov.info`
- `./target/llvm-cov/lcov.info`
- `$CARGO_TARGET_DIR/coverage/lcov.info`

### `falsify` - Mutation Testing (FDV Pattern)

Identifies mutation testing targets and weak test coverage.

```bash
batuta bug-hunter falsify .
batuta bug-hunter falsify . --timeout 60
```

### `fuzz` - Targeted Unsafe Fuzzing (FourFuzz Pattern)

Inventories unsafe blocks and identifies fuzzing targets.

```bash
batuta bug-hunter fuzz .
batuta bug-hunter fuzz . --duration 120
```

**Note**: For crates with `#![forbid(unsafe_code)]`, fuzz mode returns `BH-FUZZ-SKIPPED` (Info) instead of `BH-FUZZ-NOTARGETS` (Medium), since there's no unsafe code to fuzz.

### `deep-hunt` - Hybrid Analysis (COTTONTAIL Pattern)

Combines concolic execution analysis with SBFL for complex conditionals.

```bash
batuta bug-hunter deep-hunt .
batuta bug-hunter deep-hunt . --coverage ./lcov.info
```

### `ensemble` - Combined Results

Runs all modes and combines results with weighted scoring.

```bash
batuta bug-hunter ensemble .
batuta bug-hunter ensemble . --min-suspiciousness 0.5
```

## Advanced Features (BH-11 to BH-15)

### Spec-Driven Bug Hunting (BH-11)

Hunt bugs guided by specification files:

```bash
batuta bug-hunter spec . --spec docs/spec.md
batuta bug-hunter spec . --spec docs/spec.md --section "Authentication"
batuta bug-hunter spec . --spec docs/spec.md --update-spec
```

### Ticket-Scoped Hunting (BH-12)

Focus on areas defined by work tickets:

```bash
batuta bug-hunter ticket . --ticket GH-42
batuta bug-hunter ticket . --ticket PERF-001
```

## Output Formats

```bash
# Text output (default)
batuta bug-hunter analyze .

# JSON output
batuta bug-hunter analyze . --format json

# Markdown output
batuta bug-hunter analyze . --format markdown
```

## Finding Categories

| Category | Description |
|----------|-------------|
| MemorySafety | Pointer issues, buffer overflows |
| LogicErrors | Off-by-one, boundary conditions |
| Concurrency | Race conditions, deadlocks |
| ResourceLeaks | Unclosed handles, memory leaks |
| ConfigurationErrors | Missing configs, wrong settings |
| TypeErrors | Type mismatches, invalid casts |

## Severity Levels

| Severity | Suspiciousness | Action Required |
|----------|----------------|-----------------|
| Critical | 0.9+ | Immediate fix |
| High | 0.7-0.9 | Fix before release |
| Medium | 0.5-0.7 | Review and address |
| Low | 0.3-0.5 | Consider fixing |
| Info | 0.0-0.3 | Informational |

## Example Output

```
Bug Hunter Report
═══════════════════════════════════════════════════════════

Project: /home/user/project
Mode: Analyze | Duration: 1234ms | Findings: 5

Top Findings:

BH-CLIP-0001  High   LogicErrors  0.75  src/parser.rs:42
              Potential off-by-one in loop boundary

BH-CLIP-0002  Medium MemorySafety 0.60  src/buffer.rs:128
              Unchecked slice indexing

═══════════════════════════════════════════════════════════
```

## Integration with CI

```yaml
- name: Bug Hunter Analysis
  run: |
    batuta bug-hunter ensemble . --format json > findings.json
    # Fail if critical findings exist
    jq -e '[.findings[] | select(.severity == "Critical")] | length == 0' findings.json
```

## Related Commands

- [`batuta falsify`](./cli-falsify.md) - Full Popperian Falsification Checklist
- [`batuta analyze`](./cli-analyze.md) - Project analysis
- [`batuta stack quality`](./cli-stack.md) - Stack-wide quality metrics
