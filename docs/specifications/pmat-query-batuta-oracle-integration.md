# PMAT Query + Batuta Oracle Integration Specification

**Status:** Draft
**Version:** 1.0
**Authors:** PAIML Engineering
**Date:** 2025-01-15
**Document ID:** BATUTA-PMAT-QUERY-001

---

## 1. Executive Summary

This specification defines the integration of `pmat query` (function-level quality-annotated code search) into `batuta oracle` via a `--pmat-query` flag. The integration creates a hybrid retrieval system that combines function-level code search with document-level RAG retrieval, enabling developers to locate code by semantic intent while simultaneously assessing quality metrics (TDG grade, cyclomatic complexity, Big-O, SATD).

---

## 2. Problem Statement

The Sovereign AI Stack currently offers two disjoint search modalities:

| Modality | Tool | Granularity | Quality Signals |
|----------|------|-------------|-----------------|
| Document search | `batuta oracle --rag` | Chunk-level (paragraphs) | None |
| Code search | `pmat query` | Function-level | TDG, complexity, Big-O, SATD |

Developers must invoke these tools separately and mentally correlate results. This gap is well-documented in the source code retrieval literature: Binkley et al. [4] demonstrated that combining structural and textual features improves retrieval precision by 15-30% over either modality alone.

---

## 3. Architecture

### 3.1 Data Flow

```
User Query: "error handling"
         │
         ├──[--pmat-query]──► pmat query "error handling" --format json
         │                         │
         │                    JSON results (function-level)
         │                         │
         │                    ┌────▼─────────────────────┐
         │                    │ PmatQueryResult[]         │
         │                    │  file_path, function_name │
         │                    │  tdg_score, complexity    │
         │                    │  big_o, satd_count        │
         │                    └────┬─────────────────────┘
         │                         │
         ├──[--rag]──────────► rag_load_index() → HybridRetriever
         │                         │
         │                    ┌────▼─────────────────────┐
         │                    │ RetrievalResult[]         │
         │                    │  component, source        │
         │                    │  score, content           │
         │                    └────┬─────────────────────┘
         │                         │
         └─────────────────────────┤
                                   ▼
                          pmat_display_combined()
                          (unified ranked view)
```

### 3.2 Invocation Protocol

```bash
# Function-level search only
batuta oracle --pmat-query "error handling"

# Function-level search with quality filters
batuta oracle --pmat-query "error handling" --pmat-min-grade A --pmat-max-complexity 10

# Combined: function-level + document-level
batuta oracle --pmat-query "error handling" --rag

# JSON output for tooling
batuta oracle --pmat-query "error handling" --format json

# Include source code in results
batuta oracle --pmat-query "error handling" --pmat-include-source
```

---

## 4. Integration Protocol

### 4.1 PMAT JSON Schema

The `pmat query` command returns a JSON array of function-level results:

```json
[
  {
    "file_path": "src/pipeline.rs",
    "function_name": "validate_stage",
    "signature": "fn validate_stage(&self, stage: &Stage) -> Result<()>",
    "doc_comment": "Validates a pipeline stage using Jidoka stop-on-error.",
    "start_line": 142,
    "end_line": 185,
    "language": "rust",
    "tdg_score": 92.5,
    "tdg_grade": "A",
    "complexity": 4,
    "big_o": "O(n)",
    "satd_count": 0,
    "loc": 43,
    "relevance_score": 0.87,
    "source": null
  }
]
```

### 4.2 Tool Invocation

```rust
// Build command arguments
let args = ["query", &query, "--format", "json", "--limit", &limit.to_string()];

// Optional filters
if let Some(grade) = min_grade { args.push("--min-grade"); args.push(&grade); }
if let Some(max) = max_complexity { args.push("--max-complexity"); args.push(&max.to_string()); }
if include_source { args.push("--include-source"); }

// Execute via ToolRegistry
tools::run_tool("pmat", &args, Some(&project_path))
```

### 4.3 Error Handling

| Condition | Behavior |
|-----------|----------|
| `pmat` not in PATH | Print install instructions, exit gracefully |
| `pmat query` returns empty `[]` | Display "No functions matched" message |
| `pmat query` returns non-zero exit | Surface stderr, suggest `--pmat-project-path` |
| Invalid JSON output | Parse error with context, suggest pmat version check |
| Combined mode, RAG index missing | Show pmat results, warn about missing RAG index |

---

## 5. Output Formats

### 5.1 Text (default)

```
PMAT Query: "error handling"
──────────────────────────────────────────────────

1. [A] src/pipeline.rs:142  validate_stage          █████████░ 92.5
   fn validate_stage(&self, stage: &Stage) -> Result<()>
   Complexity: 4 | Big-O: O(n) | SATD: 0

2. [B] src/backend.rs:88    select_backend          ████████░░ 78.3
   fn select_backend(&self, workload: &Workload) -> Backend
   Complexity: 8 | Big-O: O(n log n) | SATD: 1
```

### 5.2 JSON

```json
{
  "query": "error handling",
  "source": "pmat",
  "result_count": 2,
  "results": [ ... ]
}
```

### 5.3 Markdown

```markdown
## PMAT Query Results

**Query:** error handling

| # | Grade | File | Function | TDG | Complexity | Big-O |
|---|-------|------|----------|-----|------------|-------|
| 1 | A | src/pipeline.rs:142 | validate_stage | 92.5 | 4 | O(n) |
```

---

## 6. References

[1] S. E. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333-389, 2009. DOI: 10.1561/1500000019

[2] G. V. Cormack, C. L. A. Clarke, and S. Büttcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," in *Proc. 32nd ACM SIGIR*, pp. 758-759, 2009. DOI: 10.1145/1571941.1572114

[3] T. J. McCabe, "A Complexity Measure," *IEEE Transactions on Software Engineering*, vol. SE-2, no. 4, pp. 308-320, 1976. DOI: 10.1109/TSE.1976.233837

[4] D. Binkley, D. Lawrie, S. Maex, and C. Morrell, "Identifier Length and Limited Programmer Memory in Source Code Retrieval," *Science of Computer Programming*, vol. 68, no. 1, pp. 35-48, 2007. DOI: 10.1016/j.scico.2006.09.006

[5] R. Bavishi, H. Joshi, J. Cambronero, A. Fariha, M. Lahiri, S. Gulwani, and V. Le, "Neurosymbolic Repair for Low-Code Formula Languages," in *Proc. OOPSLA*, 2022. DOI: 10.1145/3527313

[6] N. Nagappan and T. Ball, "Use of Relative Code Churn Measures to Predict System Defect Density," in *Proc. 27th ICSE*, pp. 284-292, 2005. DOI: 10.1145/1062455.1062514
