# Banco Phase 3: Training, Data Recipes, and Experiment Tracking

> Parent: [banco-spec.md](banco-spec.md) §5
> Ticket: —
> Status: **Planned**
> Depends on: Phase 2

---

## Scope

Add the training loop: upload documents, transform them into datasets via data recipes, fine-tune models with LoRA/QLoRA, track experiments with metrics, and export trained adapters. This is the phase that turns Banco from a chat server into a workbench.

## Stack Crates Used

| Crate | Version | Role |
|-------|---------|------|
| `entrenar` | 0.7.x | Autograd, LoRA/QLoRA, quantization, model merge |
| `alimentar` | 0.2.x | Zero-copy Parquet/Arrow data loading |
| `aprender` | 0.27.x | ML algorithms, APR v2 format, tokenizer |
| `repartir` | 2.0.x | Multi-GPU distributed training |
| `trueno` | 0.16.x | SIMD/GPU tensor ops underlying everything |

Feature flag: `banco-train = ["banco", "ml", "distributed"]`

## New Endpoints

### Data Management

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/v1/data/upload` | Upload PDF/CSV/JSON/DOCX/TXT files |
| GET | `/api/v1/data/files` | List uploaded files |
| DELETE | `/api/v1/data/files/{id}` | Delete uploaded file |
| POST | `/api/v1/data/recipes` | Create a data recipe (transform docs → dataset) |
| GET | `/api/v1/data/recipes` | List recipes |
| GET | `/api/v1/data/recipes/{id}` | Recipe status + preview |
| POST | `/api/v1/data/recipes/{id}/run` | Execute recipe |
| GET | `/api/v1/data/datasets` | List generated datasets |
| GET | `/api/v1/data/datasets/{id}/preview` | Preview rows |

### Training

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/v1/train/start` | Start training run |
| GET | `/api/v1/train/runs` | List all training runs |
| GET | `/api/v1/train/runs/{id}` | Run status, config, metrics |
| GET | `/api/v1/train/runs/{id}/metrics` | Real-time metrics stream (SSE) |
| POST | `/api/v1/train/runs/{id}/stop` | Stop a running training |
| DELETE | `/api/v1/train/runs/{id}` | Delete run artifacts |
| POST | `/api/v1/train/runs/{id}/export` | Export adapter/merged model |

### Experiment Tracking

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/api/v1/experiments` | List experiments (groups of runs) |
| POST | `/api/v1/experiments` | Create experiment |
| GET | `/api/v1/experiments/{id}/compare` | Compare runs within experiment |

---

## Data Recipes

### What They Are

A data recipe is a declarative pipeline that transforms raw documents into a fine-tuning dataset. It replaces the manual process of: open PDF → copy text → write prompts → format as JSONL.

### Recipe Format

```json
{
  "name": "support-docs-to-qa",
  "source_files": ["file-id-1", "file-id-2"],
  "steps": [
    {"type": "extract_text", "config": {}},
    {"type": "chunk", "config": {"max_tokens": 512, "overlap": 64}},
    {"type": "generate_qa", "config": {
      "style": "instruction",
      "system_prompt": "You are a technical support assistant.",
      "pairs_per_chunk": 3
    }},
    {"type": "format", "config": {"template": "chatml"}}
  ],
  "output_format": "jsonl"
}
```

### Built-in Recipe Steps

| Step | Input | Output | Description |
|------|-------|--------|-------------|
| `extract_text` | PDF/DOCX/TXT | Plain text | OCR-free text extraction |
| `parse_csv` | CSV | Structured rows | Column mapping to fields |
| `parse_json` | JSON | Structured records | JSONPath extraction |
| `chunk` | Text | Text chunks | Token-aware splitting with overlap |
| `generate_qa` | Chunks | Q&A pairs | Self-instruct from chunks (requires loaded model) |
| `generate_summary` | Chunks | Summaries | Summarize each chunk |
| `classify` | Text | Labeled text | Auto-label via loaded model |
| `filter` | Records | Filtered records | Quality/length/dedup filtering |
| `format` | Records | Chat format | Apply chat template (ChatML, Alpaca, etc.) |
| `deduplicate` | Records | Unique records | MinHash near-duplicate removal |

### Self-Instruct Loop

The `generate_qa` and `generate_summary` steps use the currently loaded model (Phase 2) to synthesize training data from raw documents. This creates a powerful loop:

```
Upload docs → Recipe extracts + chunks → Model generates Q&A pairs
→ Train on Q&A → Better model → Better Q&A generation → Iterate
```

This is Banco's equivalent of Unsloth's "Data Recipes powered by Nemo Data Designer."

---

## Training

### Configuration

```json
{"base_model": "primary", "dataset": "dataset-id-123", "method": "lora",
 "config": {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
   "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
   "learning_rate": 2e-4, "epochs": 3, "batch_size": 4,
   "gradient_accumulation_steps": 4, "max_seq_length": 2048,
   "warmup_steps": 100, "weight_decay": 0.01,
   "optimizer": "adamw", "scheduler": "cosine", "fp16": true}}
```

### Training Methods

| Method | Crate | VRAM | Quality | Speed |
|--------|-------|------|---------|-------|
| **LoRA** | entrenar | Low (~4GB for 7B) | Good | Fast |
| **QLoRA** | entrenar | Very low (~2GB) | Good | Medium |
| **Full fine-tune** | entrenar | High (full model) | Best | Slow |
| **CITL** (entrenar specialty) | entrenar | Medium | Good | Fast |

### Presets

```bash
# Quick fine-tune (sane defaults)
curl -X POST http://localhost:8090/api/v1/train/start \
  -d '{"dataset": "ds-123", "preset": "quick-lora"}'

# Custom YAML config (Unsloth-compatible format)
curl -X POST http://localhost:8090/api/v1/train/start \
  -d '{"dataset": "ds-123", "config_yaml": "..."}'
```

| Preset | Method | Epochs | LR | LoRA R | Target |
|--------|--------|--------|-----|--------|--------|
| `quick-lora` | LoRA | 1 | 2e-4 | 8 | q_proj, v_proj |
| `standard-lora` | LoRA | 3 | 2e-4 | 16 | q/k/v/o_proj |
| `deep-lora` | LoRA | 5 | 1e-4 | 32 | all linear |
| `qlora-low-vram` | QLoRA | 3 | 2e-4 | 16 | q/k/v/o_proj |
| `full-finetune` | FFT | 3 | 5e-5 | — | all params |

### Multi-GPU

When multiple GPUs are available, training uses `repartir` for data-parallel distribution:

```json
{
  "config": {
    "distributed": true,
    "gpu_ids": [0, 1],
    "strategy": "data_parallel"
  }
}
```

---

## Training Metrics (SSE Stream)

```
GET /api/v1/train/runs/{id}/metrics
Accept: text/event-stream

data: {"step": 1, "loss": 2.34, "grad_norm": 1.2, "lr": 0.0002, "gpu_util": 0.87, "vram_mb": 3200, "tokens_per_sec": 1200, "eta_secs": 3600}
data: {"step": 2, "loss": 2.31, "grad_norm": 1.1, "lr": 0.0002, ...}
...
data: {"step": 500, "loss": 0.45, "grad_norm": 0.3, "lr": 0.00005, "status": "complete"}
data: [DONE]
```

Metrics emitted per training step:

| Metric | Type | Description |
|--------|------|-------------|
| `step` | u64 | Global step counter |
| `loss` | f32 | Training loss |
| `grad_norm` | f32 | Gradient L2 norm |
| `lr` | f32 | Current learning rate |
| `gpu_util` | f32 | GPU utilization (0.0-1.0) |
| `vram_mb` | u64 | VRAM usage |
| `tokens_per_sec` | u64 | Training throughput |
| `eta_secs` | u64 | Estimated time remaining |
| `eval_loss` | f32 | Validation loss (when eval step) |

---

## Export

After training, export the adapter or merged model:

```bash
# Export LoRA adapter only (small, fast)
curl -X POST http://localhost:8090/api/v1/train/runs/{id}/export \
  -d '{"format": "safetensors", "merge": false}'

# Export merged model (base + adapter baked in)
curl -X POST http://localhost:8090/api/v1/train/runs/{id}/export \
  -d '{"format": "gguf", "merge": true, "quantization": "Q4_K_M"}'
```

### Export Formats

| Format | Use With | Size | Notes |
|--------|----------|------|-------|
| SafeTensors | HuggingFace, vLLM, TGI | Full | Standard interchange |
| GGUF | llama.cpp, Ollama, LM Studio | Quantized | For local inference |
| APR v2 | realizar, Banco | Compressed | Native stack format |

### Post-Export Workflow

Exported models can be immediately loaded into the primary or arena slot:

```bash
# Export + load into arena for comparison
curl -X POST .../export -d '{"format": "gguf", "merge": true, "load_slot": "arena"}'
```

This enables the compare loop: train → export → load into arena → compare against base → iterate.

---

## Experiment Tracking

### Run History

Every training run is persisted to disk (SQLite via trueno-db):

```
~/.banco/
  +-- runs/
  |     +-- run-20260319-001/
  |     |     +-- config.json
  |     |     +-- metrics.jsonl
  |     |     +-- adapter/      (LoRA weights)
  |     |     +-- merged/       (optional merged export)
  |     +-- run-20260319-002/
  +-- datasets/
  |     +-- ds-abc123.jsonl
  +-- uploads/
  +-- banco.db              (SQLite index)
```

### Compare Runs

```
GET /api/v1/experiments/{id}/compare?runs=run-001,run-002

{
  "runs": [
    {"id": "run-001", "method": "lora", "final_loss": 0.45, "duration_secs": 3600},
    {"id": "run-002", "method": "qlora", "final_loss": 0.52, "duration_secs": 1800}
  ],
  "metrics_comparison": {
    "loss_curves": [[2.3, 1.8, 0.9, 0.45], [2.3, 1.9, 1.1, 0.52]],
    "best_run": "run-001"
  }
}
```

---

## File Upload

### Supported Formats

| Format | Max Size | Processing |
|--------|----------|------------|
| PDF | 100MB | Text extraction (no OCR in Phase 3) |
| CSV | 500MB | Column detection, type inference |
| JSON/JSONL | 500MB | Schema detection, flattening |
| DOCX | 50MB | Text + heading extraction |
| TXT | 500MB | Raw text, line splitting |

### Upload API

```bash
curl -X POST http://localhost:8090/api/v1/data/upload \
  -F "file=@training-docs.pdf" \
  -F "file=@examples.jsonl"
```

Files stored in `~/.banco/uploads/` with content-hash dedup.

---

## Model Evaluation

Aprender provides `eval` with perplexity (PPL) on wikitext-2, lambada, and custom text. Banco exposes this to measure model quality before and after fine-tuning.

### Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/v1/eval/perplexity` | Run PPL evaluation on a dataset |
| POST | `/api/v1/eval/benchmark` | Run standard benchmarks |
| GET | `/api/v1/eval/runs` | List eval runs |
| GET | `/api/v1/eval/runs/{id}` | Eval results |

### Perplexity Evaluation

```bash
curl -X POST http://localhost:8090/api/v1/eval/perplexity \
  -d '{"dataset": "wikitext-2", "max_samples": 500}'

# Or on a custom dataset
curl -X POST http://localhost:8090/api/v1/eval/perplexity \
  -d '{"dataset_id": "ds-123", "text_field": "content"}'
```

Response:
```json
{
  "eval_id": "eval-001",
  "model": "phi-3-mini-Q4_K_M",
  "perplexity": 8.42,
  "tokens_evaluated": 245000,
  "duration_secs": 120
}
```

### Pre/Post Fine-Tune Comparison

The experiment comparison endpoint (§Experiment Tracking) includes eval results when available:

```json
{
  "runs": [
    {"id": "run-001", "final_loss": 0.45, "eval_ppl_before": 12.3, "eval_ppl_after": 8.4},
    {"id": "run-002", "final_loss": 0.52, "eval_ppl_before": 12.3, "eval_ppl_after": 9.1}
  ]
}
```

---

## Model Merging

Aprender and whisper-apr support merge strategies: average, weighted, TIES, DARE, SLERP. Banco exposes this for creating merged models from fine-tuned adapters or combining multiple models.

### Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/v1/models/merge` | Merge two or more models |
| GET | `/api/v1/models/merge/{id}/progress` | SSE merge progress |

### Usage

```bash
# Merge base + adapter (apply LoRA weights)
curl -X POST http://localhost:8090/api/v1/models/merge \
  -d '{
    "models": ["primary", "run-001"],
    "strategy": "weighted",
    "weights": [0.7, 0.3],
    "output_format": "gguf",
    "quantization": "Q4_K_M"
  }'

# TIES merge of multiple fine-tunes
curl -X POST http://localhost:8090/api/v1/models/merge \
  -d '{
    "models": ["./model-a.gguf", "./model-b.gguf", "./model-c.gguf"],
    "strategy": "ties",
    "output_format": "safetensors"
  }'
```

### Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `average` | Element-wise mean | Simple combination |
| `weighted` | Weighted average with per-model weights | Controlled blending |
| `ties` | Trim, Elect, Sign merge (Yadav et al., 2023) | Multi-task merging |
| `dare` | Drop And REscale (Yu et al., 2023) | Reduce interference |
| `slerp` | Spherical linear interpolation | Smooth interpolation |

Merged output can be loaded directly into primary or arena slot via `"load_slot": "primary"`.

---

## Built-in RAG Pipeline

Trueno-rag provides BM25+vector search with RRF fusion. Banco integrates this so uploaded documents are automatically searchable in chat.

### How It Works

1. Upload documents via `/api/v1/data/upload` (Phase 3 §File Upload)
2. Documents are chunked and indexed into trueno-rag automatically
3. Chat with `rag: true` retrieves relevant chunks before generating

```json
POST /api/v1/chat/completions
{
  "messages": [{"role": "User", "content": "What does our policy say about refunds?"}],
  "rag": true,
  "rag_config": {
    "top_k": 5,
    "min_score": 0.3
  }
}
```

Response includes sources:
```json
{
  "choices": [{
    "message": {
      "content": "According to the policy document, refunds are available within 30 days...",
      "sources": [
        {"file": "refund-policy.pdf", "chunk": 3, "score": 0.92, "text": "..."},
        {"file": "faq.txt", "chunk": 12, "score": 0.78, "text": "..."}
      ]
    }
  }]
}
```

### Index Management

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/v1/rag/index` | Force re-index all uploaded documents |
| GET | `/api/v1/rag/status` | Index stats (doc count, chunk count, size) |
| DELETE | `/api/v1/rag/index` | Clear RAG index |

Documents are auto-indexed on upload. Re-indexing is only needed after config changes (chunk size, overlap).

---

## Architecture Changes

```
src/serve/banco/
  +-- mod.rs              (updated: register new submodules)
  +-- types.rs            (updated: training/data/eval types)
  +-- state.rs            (updated: add run store, dataset store, RAG index)
  +-- handlers.rs         (Phase 1+2 handlers, unchanged)
  +-- data_handlers.rs    (new: upload, recipes, datasets)
  +-- train_handlers.rs   (new: start, stop, metrics, export)
  +-- eval_handlers.rs    (new: perplexity, benchmarks)
  +-- merge_handlers.rs   (new: model merge strategies)
  +-- rag_handlers.rs     (new: RAG index management, chat integration)
  +-- experiment.rs       (new: experiment tracking)
  +-- recipes.rs          (new: recipe execution engine)
  +-- storage.rs          (new: ~/.banco/ persistence)
```

## Open Questions

1. **Self-instruct dependency**: `generate_qa` requires a loaded model. Should we mandate a model before recipes, or support a "manual Q&A" mode?
2. **GPU contention**: Training and inference on the same GPU. Pause inference during training, or use separate CUDA streams?
3. **Eval during training**: Run eval on a held-out split? How often? Configurable?
4. **Export quantization**: Quantize in Banco (via aprender/realizar), or shell out to llama.cpp's `quantize`?
5. **RAG chunking strategy**: Fixed-size token chunks, or semantic paragraph splitting? Configurable?
6. **Merge VRAM**: TIES/DARE need all models in memory simultaneously. Limit to 2-model merge on low-VRAM machines?

## Test Strategy

| Test | Type | What |
|------|------|------|
| Upload PDF, list files | Integration | File storage + listing |
| Create recipe, run on CSV | Integration | Recipe execution pipeline |
| Start LoRA training on tiny dataset | Integration | entrenar integration |
| Stream metrics during training | Integration | SSE metrics format |
| Export adapter as safetensors | Integration | File output + format |
| Compare two runs | Unit | Metrics comparison logic |
| Preset expansion | Unit | Preset → full config |
| No-model recipe step → error | Unit | Self-instruct guard |
| PPL eval on wikitext-2 sample | Integration | aprender eval pipeline |
| Weighted merge of two models | Integration | Merge output loadable |
| TIES merge produces valid model | Integration | Merge strategy correctness |
| RAG chat returns sources | Integration | trueno-rag + chat pipeline |
| RAG auto-index on upload | Integration | Index triggered by upload |
| Batch inference on JSONL | Integration | Multi-prompt processing |
