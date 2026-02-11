# Tool Detection System

Batuta discovers external transpilers (depyler, decy, bashrs) and analysis tools (pmat, renacer) at runtime through PATH-based lookup.

## Detection Process

1. Search PATH for the binary name
2. Run `<tool> --version` to get the version
3. Compare against minimum required version
4. Cache the result in `.batuta/cache/tool_versions.json`

## Tool Registry

| Tool | Binary | Min Version | Purpose |
|------|--------|-------------|---------|
| depyler | `depyler` | 0.5.0 | Python to Rust |
| decy | `decy` | 0.3.0 | C/C++ to Rust |
| bashrs | `bashrs` | 0.2.0 | Shell to Rust |
| pmat | `pmat` | 0.8.0 | Static analysis, TDG |
| renacer | `renacer` | 0.7.0 | Syscall tracing |

## Checking Tools

```bash
batuta analyze --check-tools
```

Output:

```
Tool Detection Report:
  depyler  v0.5.2  ~/.cargo/bin/depyler  [OK]
  decy     v0.3.1  ~/.cargo/bin/decy      [OK]
  bashrs   v0.2.0  ~/.cargo/bin/bashrs    [OK]
  pmat     v0.8.3  ~/.cargo/bin/pmat       [OK]
  renacer  v0.7.1  ~/.cargo/bin/renacer    [OK]
```

## Version Mismatch Handling

| Condition | Behavior |
|-----------|----------|
| Tool found, version OK | Proceed normally |
| Tool found, version old | Error with upgrade instructions |
| Tool not found | Error with install instructions |

## Fallback Behavior

Configure in `batuta.toml`:

```toml
[pipeline]
# strict: fail if any tool missing (default)
# lenient: skip unsupported languages, warn only
missing_tool_policy = "strict"
```

## Cache Behavior

Tool detection results are cached to avoid repeated PATH lookups. The cache is invalidated when:

- The PATH environment variable changes
- A tool binary is newer than the cache entry
- The cache is older than 24 hours

Force re-detection:

```bash
rm .batuta/cache/tool_versions.json
batuta analyze --check-tools
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
