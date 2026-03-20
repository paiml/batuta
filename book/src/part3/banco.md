# Banco: Local-First AI Workbench

Banco is a self-contained AI studio shipped as `batuta serve --banco`. One command, one binary, zero cloud dependencies.

## Quick Start

```bash
# Build with banco feature
cargo build --features banco

# Start the workbench
batuta serve --banco --port 8090

# Chat (OpenAI-compatible)
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

## Architecture

```
batuta serve --banco
  │
  ├── Endpoints (19 total)
  │   ├── /health                     Health + circuit breaker state
  │   ├── /api/v1/models              Recommended backends as models
  │   ├── /api/v1/chat/completions    Chat (sync + SSE streaming)
  │   ├── /api/v1/system              Privacy tier, GPU, version, telemetry=false
  │   ├── /api/v1/tokenize            Token count estimation
  │   ├── /api/v1/detokenize          Approximate text from tokens
  │   ├── /api/v1/embeddings          Text embeddings (128-dim heuristic)
  │   ├── /api/v1/models/load|unload|status  Model management
  │   ├── /api/v1/chat/parameters     Inference parameter tuning
  │   ├── /api/v1/conversations       CRUD + auto-title
  │   ├── /api/v1/prompts             System prompt presets
  │   ├── /v1/*                       OpenAI SDK aliases
  │   └── /api/*                      Ollama protocol compat
  │
  ├── Middleware (3 layers)
  │   ├── Audit logging               Every request logged with latency
  │   ├── Authentication              API key for LAN mode
  │   └── Privacy + CORS              X-Privacy-Tier header, CORS headers
  │
  └── State (Arc<BancoStateInner>)
      ├── BackendSelector              Privacy-aware backend recommendation
      ├── SpilloverRouter              Heijunka load leveling (atomics)
      ├── CostCircuitBreaker           Muda budget enforcement (atomics)
      ├── ContextManager               Token counting + truncation
      ├── ChatTemplateEngine           6 template formats
      ├── ConversationStore            In-memory + optional disk JSONL
      ├── PromptStore                  Built-in + custom presets
      └── AuthStore                    Local (no auth) or ApiKey mode
```

## Compatibility

Banco speaks three protocols from the same port:

| Protocol | Routes | Works With |
|----------|--------|------------|
| **Banco native** | `/api/v1/*` | curl, custom clients |
| **OpenAI** | `/v1/*` | OpenAI Python SDK, LangChain, LlamaIndex |
| **Ollama** | `/api/chat`, `/api/tags`, `/api/show` | Open WebUI, Continue.dev, Aider |

### OpenAI SDK Example

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/api/v1", api_key="unused")
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(r.choices[0].message.content)
```

## Model Loading

```bash
# Load at startup
batuta serve --banco --model ./tinyllama.gguf --port 8090

# Load at runtime via API
curl -X POST http://localhost:8090/api/v1/models/load \
  -d '{"model": "./phi-3-mini.gguf"}'

# Check status (shows architecture, vocab, layers when inference feature enabled)
curl http://localhost:8090/api/v1/models/status
# {"loaded":true,"model":{"model_id":"phi-3-mini","format":"gguf",
#   "architecture":"phi3","vocab_size":32064,"hidden_dim":3072,
#   "num_layers":32,"context_length":4096,"tensor_count":195}}

# Unload
curl -X POST http://localhost:8090/api/v1/models/unload
```

Build with `--features banco,inference` for GGUF metadata extraction via realizar.

## Conversations

Messages are persisted server-side. First user message auto-generates a title.

```bash
# Create
curl -X POST http://localhost:8090/api/v1/conversations -d '{}'
# {"id":"conv-1234-0","title":"New conversation"}

# Chat within conversation
curl -X POST http://localhost:8090/api/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Explain gravity"}],
       "conversation_id":"conv-1234-0"}'

# List
curl http://localhost:8090/api/v1/conversations

# Get full history
curl http://localhost:8090/api/v1/conversations/conv-1234-0
```

## System Prompt Presets

Three built-in presets, plus custom ones. Referenced via `@preset:name`:

```bash
# List presets
curl http://localhost:8090/api/v1/prompts
# Built-in: coding, concise, tutor

# Use in chat
curl -X POST http://localhost:8090/api/v1/chat/completions \
  -d '{"messages":[
    {"role":"system","content":"@preset:coding"},
    {"role":"user","content":"Write fizzbuzz in Rust"}
  ]}'

# Create custom
curl -X POST http://localhost:8090/api/v1/prompts \
  -d '{"name":"Pirate","content":"You are a pirate. Arr!"}'
```

## Privacy Tiers

Every response includes `X-Privacy-Tier`. Sovereign mode blocks all external backends.

| Tier | Data Location | Remote Allowed |
|------|--------------|----------------|
| **Sovereign** | Local only | No |
| **Private** | VPC/dedicated | Enterprise only |
| **Standard** | Anywhere | Yes |

## Authentication

```bash
# Local (127.0.0.1) — no auth
batuta serve --banco

# LAN (0.0.0.0) — API key recommended
batuta serve --banco --host 0.0.0.0

# Client auth
curl -H "Authorization: Bearer bk_..." http://192.168.1.5:8090/api/v1/models
```

## Configuration

`~/.banco/config.toml`:

```toml
[server]
host = "127.0.0.1"
port = 8090
privacy_tier = "standard"

[inference]
temperature = 0.7
top_p = 1.0
max_tokens = 256

[budget]
daily_limit_usd = 10.0
max_request_usd = 1.0
```

## Provable Contracts

5 YAML contracts in `contracts/banco/` formalize Banco's critical invariants:

| Contract | Property |
|----------|----------|
| privacy-enforcement | Sovereign blocks all remote backends |
| budget-conservation | Spending never exceeds daily budget |
| routing-determinism | Same state → same routing decision |
| context-enforcement | Truncated messages always fit window |
| template-correctness | User content preserved across formats |

Each has Kani harnesses and 20 falsification tests.

## Feature Flag

```toml
[dependencies]
batuta = { version = "0.7", features = ["banco"] }
```

Adds: axum, tower, async-stream, tokio-stream. Default build unaffected.

## Inference

When built with `--features banco,inference`, Banco performs real token generation using realizar's `forward_single_with_cache()` autoregressive loop:

```
POST /api/v1/chat/completions → tokenize prompt → prefill cache → decode tokens → response
```

- **Prefill**: Processes all prompt tokens through the model to populate KV cache
- **Decode**: Autoregressively generates tokens using greedy or top-k sampling
- **Streaming**: Each generated token is yielded as an SSE chunk in real time
- **Fallback**: Without inference feature or without a loaded model, returns echo/dry-run response

Sampling parameters (temperature, top_k, max_tokens) can be set per-request or via `PUT /api/v1/chat/parameters`.

## Phase Roadmap

| Phase | Status | What |
|-------|--------|------|
| **1** | **Complete** | HTTP API skeleton, 24 endpoints, 121 tests |
| **2a** | **Complete** | Model slot, load/unload/status, inference params, GGUF metadata, structured output types |
| **2b** | **Complete** | Inference loop: `forward_single_with_cache()`, greedy/top-k sampling, SSE streaming, 148 tests |
| 3 | Planned | Training, data recipes, eval, RAG |
| 4 | Planned | Browser UI, code sandbox, agents |

See [banco-spec.md](../../docs/specifications/components/banco-spec.md) for full specification.
