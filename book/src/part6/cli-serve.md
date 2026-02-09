# `batuta serve`

Serve ML models via Realizar inference server with optional OpenAI-compatible API.

## Synopsis

```bash
batuta serve [OPTIONS] [MODEL]
```

## Description

The serve command launches a local inference server for ML models. It supports multiple model sources (Pacha registry, HuggingFace, local files) and can expose an OpenAI-compatible REST API for drop-in integration with existing toolchains.

## Arguments

| Argument | Description |
|----------|-------------|
| `[MODEL]` | Model reference: `pacha://name:version`, `hf://org/model`, or local path |

## Options

| Option | Description |
|--------|-------------|
| `-H, --host <HOST>` | Host to bind to (default: `127.0.0.1`) |
| `-p, --port <PORT>` | Port to bind to (default: `8080`) |
| `--openai-api` | Enable OpenAI-compatible API at `/v1/*` |
| `--watch` | Enable hot-reload on model changes |
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Print help |

## Examples

### Serve a Local Model

```bash
$ batuta serve ./model.gguf --port 8080
```

### Serve from Pacha Registry

```bash
$ batuta serve pacha://llama3:8b
```

### OpenAI-Compatible API

```bash
$ batuta serve pacha://llama3:8b --openai-api

# Then use standard OpenAI clients:
# curl http://localhost:8080/v1/chat/completions ...
```

### Hot-Reload During Development

```bash
$ batuta serve ./model.apr --watch
```

## See Also

- [Model Serving Ecosystem](../part3/model-serving.md)
- [`batuta deploy`](./cli-deploy.md) - Production deployment

---

**Previous:** [`batuta mcp`](./cli-mcp.md)
**Next:** [`batuta deploy`](./cli-deploy.md)
