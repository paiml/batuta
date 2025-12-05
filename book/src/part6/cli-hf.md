# `batuta hf`

HuggingFace Hub integration commands.

## Synopsis

```bash
batuta hf <COMMAND>
```

## Commands

| Command | Description |
|---------|-------------|
| `tree` | Display HuggingFace ecosystem tree |
| `search` | Search models, datasets, spaces |
| `info` | Get info about a Hub asset |
| `pull` | Download from HuggingFace Hub |
| `push` | Upload to HuggingFace Hub |

---

## `batuta hf tree`

Display hierarchical view of HuggingFace ecosystem or PAIML integration map.

### Usage

```bash
batuta hf tree [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--integration` | Show PAIML↔HuggingFace integration map |
| `--format <FORMAT>` | Output format: `ascii` (default), `json` |

### Examples

```bash
# HuggingFace ecosystem tree
batuta hf tree

# Output:
# HuggingFace Ecosystem (6 categories)
# ├── hub
# │   ├── models         (700K+ models)
# │   ├── datasets       (100K+ datasets)
# │   └── spaces         (300K+ spaces)
# ├── libraries
# │   ├── transformers   (Model architectures)
# │   └── ...

# PAIML-HuggingFace integration map
batuta hf tree --integration

# Output shows:
# ✓ COMPATIBLE  - Interoperates with HF format/API
# ⚡ ALTERNATIVE - PAIML native replacement (pure Rust)
# 🔄 ORCHESTRATES - PAIML wraps/orchestrates HF
# 📦 USES        - PAIML uses HF library directly
```

---

## `batuta hf search`

Search HuggingFace Hub for models, datasets, or spaces.

### Usage

```bash
batuta hf search <ASSET_TYPE> <QUERY> [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<ASSET_TYPE>` | Type: `model`, `dataset`, `space` |
| `<QUERY>` | Search query string |

### Options

| Option | Description |
|--------|-------------|
| `--task <TASK>` | Filter by task (for models) |
| `--limit <N>` | Limit results (default: 10) |

### Examples

```bash
# Search for Llama models
batuta hf search model "llama 7b" --task text-generation

# Search for speech datasets
batuta hf search dataset "common voice" --limit 5

# Search for Gradio spaces
batuta hf search space "image classifier"
```

---

## `batuta hf info`

Get detailed information about a HuggingFace asset.

### Usage

```bash
batuta hf info <ASSET_TYPE> <REPO_ID>
```

### Examples

```bash
# Get model info
batuta hf info model "meta-llama/Llama-2-7b-hf"

# Get dataset info
batuta hf info dataset "mozilla-foundation/common_voice_13_0"

# Get space info
batuta hf info space "gradio/chatbot"
```

---

## `batuta hf pull`

Download models, datasets, or spaces from HuggingFace Hub.

### Usage

```bash
batuta hf pull <ASSET_TYPE> <REPO_ID> [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `-o, --output <PATH>` | Output directory |
| `--quantization <Q>` | Model quantization (Q4_K_M, Q5_K_M, etc.) |

### Examples

```bash
# Pull GGUF model with quantization
batuta hf pull model "TheBloke/Llama-2-7B-GGUF" --quantization Q4_K_M

# Pull to specific directory
batuta hf pull model "mistralai/Mistral-7B-v0.1" -o ./models/

# Pull dataset
batuta hf pull dataset "squad" -o ./data/
```

---

## `batuta hf push`

Upload models, datasets, or spaces to HuggingFace Hub.

### Usage

```bash
batuta hf push <ASSET_TYPE> <PATH> --repo <REPO_ID> [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--repo <REPO_ID>` | Target repository (required) |
| `--message <MSG>` | Commit message |

### Examples

```bash
# Push trained model
batuta hf push model ./my-model --repo "myorg/my-classifier"

# Push dataset
batuta hf push dataset ./data/processed --repo "myorg/my-dataset"

# Push Presentar app as Space
batuta hf push space ./my-app --repo "myorg/demo" --message "Initial release"
```

---

## PAIML-HuggingFace Integration

The integration map shows how PAIML stack components relate to HuggingFace (28 mappings):

| Category | PAIML | HuggingFace | Type |
|----------|-------|-------------|------|
| **Formats** | `.apr` | pickle/.joblib, safetensors, gguf | ⚡ Alternative |
| | realizar/gguf | gguf | ✓ Compatible |
| | realizar/safetensors | safetensors | ✓ Compatible |
| **Data Formats** | `.ald` | parquet/arrow, json/csv | ⚡ Alternative |
| **Hub Access** | aprender/hf_hub | huggingface_hub | 📦 Uses |
| | batuta/hf | huggingface_hub | 🔄 Orchestrates |
| **Registry** | pacha | HF Hub registry, MLflow/W&B | ⚡ Alternative |
| **Inference** | realizar | transformers, TGI | ⚡ Alternative |
| | realizar/moe | optimum | ⚡ Alternative |
| **Classical ML** | aprender | sklearn, xgboost/lightgbm | ⚡ Alternative |
| **Deep Learning** | entrenar | PyTorch training | ⚡ Alternative |
| | alimentar | datasets | ⚡ Alternative |
| **Compute** | trueno | NumPy/PyTorch tensors | ⚡ Alternative |
| | repartir | accelerate | ⚡ Alternative |
| **Tokenization** | realizar/tokenizer | tokenizers | ✓ Compatible |
| | trueno-rag | tokenizers | ✓ Compatible |
| **Apps** | presentar | gradio | ⚡ Alternative |
| | trueno-viz | visualization | ⚡ Alternative |
| **Quality** | certeza | evaluate | ⚡ Alternative |
| **MCP Tooling** | pforge | LangChain Tools | ⚡ Alternative |
| | pmat | code analysis tools | ⚡ Alternative |
| | pmcp | mcp-sdk | ⚡ Alternative |

**Legend:**
- ✓ COMPATIBLE - Interoperates with HF format/API
- ⚡ ALTERNATIVE - PAIML native replacement (pure Rust)
- 🔄 ORCHESTRATES - PAIML wraps/orchestrates HF
- 📦 USES - PAIML uses HF library directly

### Compatible Formats

PAIML can load and save HuggingFace formats:

```rust
// Load GGUF model (realizar)
let model = GGUFModel::from_file("model.gguf")?;

// Load SafeTensors (aprender)
let weights = SafeTensors::load("model.safetensors")?;

// Load HF tokenizer (realizar)
let tokenizer = Tokenizer::from_pretrained("meta-llama/Llama-2-7b-hf")?;
```

### Security Features (v1.1.0)

### SafeTensors Enforcement

By default, `batuta hf pull` blocks unsafe pickle-based formats:

```bash
# Default: blocks .bin, .pkl, .pt files
batuta hf pull model "repo/model"

# Explicit override for unsafe formats
batuta hf pull model "repo/model" --allow-unsafe
```

| Extension | Safety | Notes |
|-----------|--------|-------|
| `.safetensors` | ✓ Safe | Recommended |
| `.gguf` | ✓ Safe | Quantized |
| `.json` | ✓ Safe | Config |
| `.bin` | ✗ Unsafe | Pickle-based |
| `.pkl` | ✗ Unsafe | Pickle |
| `.pt` | ✗ Unsafe | PyTorch |

### Secret Scanning

Automatic scan before push blocks accidental credential exposure:

```bash
# Blocked if secrets detected
batuta hf push model ./my-model --repo "org/model"

# Detected patterns:
# - .env files
# - Private keys (.pem, id_rsa)
# - Credential files
```

### Rate Limit Handling

Automatic exponential backoff for API rate limits (429):

- Initial: 1s → 2s → 4s → 8s → 16s
- Max backoff: 60s
- Max retries: 5
- Respects `Retry-After` header

### Model Card Auto-Generation

```bash
# Auto-generates README.md if missing
batuta hf push model ./my-model --repo "org/model"
```

Generated card includes:
- YAML frontmatter (license, tags)
- Training metrics from certeza
- PAIML stack attribution

### Differential Uploads

Only uploads changed files using content-addressable hashing:

```bash
# Only uploads modified files
batuta hf push model ./my-model --repo "org/model"
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token |
| `HF_HOME` | Cache directory |
| `HF_HUB_OFFLINE` | Offline mode |
