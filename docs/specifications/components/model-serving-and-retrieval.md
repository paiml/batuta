# Model Serving and Retrieval Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> Sources: model-serving-ecosystem-spec, hugging-face-integration-query-publish-spec, hugging-face-crud-spec, retriever-spec

---

## 1. Model Serving Ecosystem

### Local Serving Landscape

| Tool | Language | Format | Key Feature | PAIML Integration |
|------|----------|--------|-------------|-------------------|
| **realizar** | Rust | GGUF/SafeTensors/APR | MoE, circuit breakers | Native |
| Ollama | Go | GGUF | Docker-like UX | API compatible |
| llamafile | C | GGUF | Single executable | Format compatible |
| llama.cpp | C++ | GGUF | Reference impl | Format compatible |
| vLLM | Python | PT/SafeTensors | PagedAttention | API bridgeable |
| TGI | Rust/Python | SafeTensors | HF official | Pattern reference |

### Remote Serving Landscape

| Service | Provider | Characteristics |
|---------|----------|----------------|
| HF Inference API | HuggingFace | Serverless, pay-per-token |
| HF Endpoints | HuggingFace | Dedicated, auto-scaling |
| Together.ai | Together | Fast inference, open models |
| AWS SageMaker | AWS | Managed, GPU instances |
| Replicate | Replicate | Docker-based, pay-per-second |

### Batuta Serve Architecture

```
batuta serve/
  +-- backends.rs      # PrivacyTier (Sovereign/Private/Standard)
  +-- router.rs        # SpilloverRouter (local-first, remote fallback)
  +-- failover.rs      # FailoverManager (streaming recovery)
  +-- circuit_breaker.rs  # CostCircuitBreaker (budget enforcement)
  +-- context.rs       # ContextManager (token counting, truncation)
  +-- templates.rs     # ChatTemplateEngine (ChatML/Llama/Mistral)
```

### Privacy Tiers

| Tier | Data Location | Allowed Backends | Use Case |
|------|--------------|------------------|----------|
| **Sovereign** | Local only | realizar | GDPR, classified data |
| **Private** | VPC/dedicated | realizar, HF Endpoints | Enterprise |
| **Standard** | Any | All backends | General use |

### Failover Strategy

```
Request -> Primary Backend (realizar local)
              |
              +-- Success -> Return
              |
              +-- Failure -> Circuit Breaker check
                               |
                               +-- Budget OK -> Spillover to remote
                               |
                               +-- Budget exceeded -> Reject
```

---

## 2. HuggingFace Integration

### CLI Interface

```bash
# Search
batuta hf search models "llama 7b" --task text-generation
batuta hf search datasets "common voice" --language en --size ">1GB"
batuta hf search spaces "gradio" --sdk gradio

# Info
batuta hf info model "meta-llama/Llama-2-7b-hf"
batuta hf info dataset "mozilla-foundation/common_voice_13_0"

# Pull
batuta hf pull model "TheBloke/Llama-2-7B-GGUF" --quantization Q4_K_M
batuta hf pull model "mistralai/Mistral-7B-v0.1" --format gguf --output ./models/
batuta hf pull dataset "squad" --split train --output ./data/

# Push
batuta hf push model ./my-model --repo "myorg/my-classifier" --format safetensors
batuta hf push dataset ./data/processed --repo "myorg/my-dataset"

# Ecosystem view
batuta hf tree                    # Full ecosystem
batuta hf tree --integration      # PAIML integration mapping
batuta hf tree --platform models  # Filter by category
```

### Ecosystem Taxonomy (50+ Components)

| Category | Examples | PAIML Mapping |
|----------|---------|---------------|
| Libraries | transformers, datasets, tokenizers | aprender, alimentar |
| Inference | TGI, Inference API, Optimum | realizar |
| Training | Accelerate, PEFT, TRL | entrenar |
| Formats | SafeTensors, GGUF | APR v2, GGUF |
| Tools | Hub, Evaluate, Gradio | pacha, pmat, presentar |

### PAIML Integration Mapping

| HuggingFace | PAIML Equivalent | Notes |
|-------------|-----------------|-------|
| `transformers` | `aprender` + `realizar` | Training + inference split |
| `datasets` | `alimentar` | Zero-copy Parquet/Arrow |
| `tokenizers` | `aprender::tokenizer` | Rust-native |
| `safetensors` | `aprender::apr` | APR v2 preferred |
| `accelerate` | `repartir` | Distributed compute |
| `peft` (LoRA) | `entrenar::lora` | LoRA/QLoRA |
| `evaluate` | `pmat` | Quality metrics |
| `gradio` | `presentar` | WASM-first UI |

### Coursera Specialization Support

The HF integration supports "Next-Gen AI Development with Hugging Face" (5 courses, 60 hours):

| Course | HF Components | Batuta Commands |
|--------|--------------|-----------------|
| Foundations | Hub, Transformers | `batuta hf tree`, `batuta hf info` |
| NLP Deep Dive | Tokenizers, Datasets | `batuta hf search`, `batuta hf pull` |
| Computer Vision | ViT, CLIP | `batuta hf pull model` |
| Audio/Multimodal | Whisper, BLIP | `batuta hf pull model --format gguf` |
| Production | TGI, Endpoints | `batuta hf push`, `batuta serve` |

---

## 3. Int8 Rescoring Retriever

### Design

Two-stage scalar int8 rescoring retriever achieving **99% accuracy retention** with **3.66x speedup** and **4x memory reduction**.

### Architecture

```
Stage 1: Fast Retrieval (Int8)
  Query -> Int8 Quantize -> ANN Search (HNSW) -> Candidate Pool (k * multiplier)

Stage 2: Precise Rescoring (Mixed Precision)
  f32 Query -> Dot Product (f32 x i8) -> Rerank Top-K -> Final Results
```

### Mathematical Foundation

**Stage 1 (Int8 Approximate):**
- Quantize: `Q(x) = round(x / scale) + zero_point`
- Scale: `(max(x) - min(x)) / 255`
- Compute approximate scores: `s_approx(q, d) = q_i8 . d_i8`
- Retrieve top-m candidates where m = k * multiplier

**Stage 2 (Float32 Rescoring):**
- Precise scores: `s_precise(q, c) = q . dequant(c_i8)`
- Return top-k from candidates

### Performance Targets

| Metric | Target | Method |
|--------|--------|--------|
| Accuracy retention | >= 99% vs full f32 | NDCG@10 comparison |
| Speed improvement | >= 3x | Throughput benchmark |
| Memory reduction | >= 4x (f32 -> i8) | RSS measurement |
| Quantization error | < 2% MSE | Error bound validation |

### SIMD Implementation

Int8 dot product uses SIMD acceleration via trueno:
- **AVX-512 VNNI**: `_mm512_dpbusd_epi32` (64 int8 ops per cycle)
- **AVX2**: `_mm256_maddubs_epi16` fallback
- **NEON**: `vdotq_s32` on ARM

### Calibration

Per-vector symmetric quantization with calibration dataset:
1. Compute per-dimension statistics on representative sample
2. Select scale/zero_point to minimize quantization error
3. Validate error bounds (Poka-Yoke: reject if MSE > threshold)
4. Monitor runtime accuracy drift (Kaizen: recalibrate if NDCG drops)

### Key Citations

| Topic | Reference |
|-------|-----------|
| Quantization theory | Jacob et al. (2018), CVPR |
| Embedding quantization | Xiao et al. (2023), SmoothQuant, ICML |
| Two-stage retrieval | Nogueira & Cho (2019), BERT reranking |
| HNSW index | Malkov & Yashunin (2020), IEEE TPAMI |
| RRF fusion | Cormack et al. (2009), SIGIR |
| BM25 | Robertson & Zaragoza (2009) |

---

## 4. Serving CLI

```bash
# Start local model server
batuta serve --model ./model.gguf --port 8080
batuta serve --model ./model.apr --privacy sovereign

# Health check
batuta serve health

# Benchmark
batuta serve bench --tokens 100 --batch-size 32
```
