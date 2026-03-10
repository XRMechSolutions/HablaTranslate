# LM Studio Manager

`services/lmstudio_manager.py` (728L) — Windows-only process lifecycle management for LM Studio, including model discovery, loading, health monitoring, and settings persistence.

## Why LM Studio?

LM Studio provides a local LLM server with an OpenAI-compatible API. It's the default provider for development and personal use because it has a GUI for browsing and downloading models, handles GGUF quantization natively, and supports hot-swapping models without server restarts.

LM Studio is Windows-only for this project (the target hardware is an MSI laptop running Windows). On Linux/macOS, the server falls back to Ollama automatically.

## Process Management

### Startup

`ensure_running()` checks if LM Studio is already running (via HTTP health check to `/v1/models`). If not:

1. Locate `lms` CLI executable (searches PATH, common cache dirs)
2. Launch LM Studio via subprocess
3. Drain stdout/stderr in daemon threads (prevent pipe blocking)
4. Wait for server to become responsive (health check loop)
5. Discover currently loaded models

### Shutdown

`stop()` terminates the LM Studio process tree using `taskkill /T` on Windows (kills the subprocess and all children).

### Restart

`restart()` stops, waits, starts, and reloads the previously active model.

## Model Management

### Discovery

Three strategies for listing available models, tried in order:

1. **CLI**: `lms ls --json` — authoritative list of all downloaded models
2. **API**: `GET /v1/models` — returns loaded models only
3. **File scan**: Search configured model paths for `.gguf` files

### Loading

Three strategies for loading a model, tried in order:

1. **CLI**: `lms load <stem>` — preferred; CLI resolves GGUF path from stem
2. **API prompt**: `POST /v1/completions` with minimal prompt — triggers lazy loading
3. **Alt endpoint**: `POST /v1/models/load` with full path — undocumented fallback

### Model Stem Normalization

`_model_stem()` normalizes model identifiers by stripping:
- `:N` instance suffixes
- `.gguf` extension
- Quantization suffixes (`-Q3_K_M`, `-Q4_K_S`, etc.)
- Case normalization to lowercase

This allows flexible matching: `Qwen3-4B-Q3_K_M.gguf`, `qwen3-4b`, and `qwen3-4b:1` all resolve to the same model.

### Switching

`switch_model()` unloads the current model (if different), loads the new one, and updates persisted settings.

## Health Monitoring

`start_monitor()` launches a background task that runs every 60 seconds:

- Check if LM Studio is still responding
- Compare loaded models against expected models
- Log warnings if models diverge
- On failure: attempt `ensure_running()` to auto-restart
- 20-second startup delay before first check

## Settings Persistence

Provider selection, model choices, API keys, and quick-model preferences are persisted to `data/llm_settings.json`. Restored on startup so the last-used configuration survives server restarts.

## Platform Guard

`main.py` checks `sys.platform` before initializing LM Studio:

```python
if provider == "lmstudio" and sys.platform != "win32":
    # Fall back to Ollama, log warning
```

The `lms.exe` path lookup also guards: on non-Windows platforms, it falls through to PATH lookup immediately instead of searching Windows-specific cache directories.

## Design Rationale

**Why process management instead of assuming LM Studio is running?** In development, the developer frequently restarts LM Studio or switches models. Having the server manage the lifecycle means it can auto-recover from crashes and ensure the right model is loaded.

**Why three model-loading strategies?** LM Studio's CLI and API have evolved across versions. The cascade ensures compatibility: if one method fails (version mismatch, undocumented API change), the next one tries.

**Why stem normalization?** Model filenames include quantization info (`Q3_K_M`, `Q4_K_S`) and instance suffixes that don't matter for selection. Normalizing lets users specify `qwen3-4b` without worrying about the exact filename.

**Why Windows-only?** LM Studio's CLI (`lms.exe`) and process model are Windows-specific. Ollama serves the same purpose on Linux with less management overhead (it runs as a service). Supporting both platforms with LM Studio would double the process management code for no real benefit.

## Planned Direction

- **LM Studio auto-restart on health failure** (3.7): Already wired — the health monitor calls `ensure_running()` on failure. Needs testing under real crash conditions.
- **Cloud deployment** (Phase 10): LM Studio is irrelevant for cloud. The Ollama and cloud provider paths handle that.
