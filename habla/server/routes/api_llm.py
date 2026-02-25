"""LLM provider management and LM Studio control routes."""

import asyncio

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import server.routes._state as _state


# --- LLM Provider Routes ---

llm_router = APIRouter(prefix="/api/llm", tags=["llm"])


@llm_router.get("/current")
async def llm_current():
    """Return current LLM provider, model, and metrics.

    Returns 200 {status: "not initialized"} when pipeline is None.
    Returns 200 with provider, model, quick_model, metrics (+ costs if openai).
    """
    if not _state._pipeline:
        return {"status": "not initialized"}
    t = _state._pipeline.translator
    result = {
        "provider": t.config.provider,
        "model": t.config.model,
        "quick_model": t.config.quick_model,
        "metrics": t.metrics,
    }
    if t.config.provider == "openai":
        result["costs"] = t.costs
    return result


@llm_router.get("/providers")
async def llm_providers():
    """Probe all configured providers and return connection status + models.

    Returns 200 {providers: []} when pipeline is None.
    Returns 200 with providers list (ollama, lmstudio, openai) and active provider.
    Each provider has name, url, status (ok/down/error/no_api_key), and models list.
    """
    if not _state._pipeline:
        return {"providers": []}

    cfg = _state._pipeline.translator.config
    providers = []

    # Ollama
    ollama_info = {"name": "ollama", "url": cfg.ollama_url, "status": "unknown", "models": []}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{cfg.ollama_url}/api/tags")
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            ollama_info["status"] = "ok"
            ollama_info["models"] = models
    except httpx.ConnectError:
        ollama_info["status"] = "down"
    except Exception:
        ollama_info["status"] = "error"
    providers.append(ollama_info)

    # LM Studio
    lms_info = {"name": "lmstudio", "url": cfg.lmstudio_url, "status": "unknown", "models": []}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{cfg.lmstudio_url}/v1/models")
            resp.raise_for_status()
            models = [m.get("id", "") for m in resp.json().get("data", [])]
            lms_info["status"] = "ok"
            lms_info["models"] = models
    except httpx.ConnectError:
        lms_info["status"] = "down"
    except Exception:
        lms_info["status"] = "error"
    providers.append(lms_info)

    # OpenAI
    openai_info = {"name": "openai", "status": "unknown", "models": []}
    if cfg.openai_api_key:
        openai_info["status"] = "ok"
        openai_info["models"] = ["gpt-5", "gpt-5-mini", "gpt-4o-mini", "gpt-5-nano"]
    else:
        openai_info["status"] = "no_api_key"
    providers.append(openai_info)

    return {"providers": providers, "active": cfg.provider}


@llm_router.get("/models")
async def llm_models(provider: str):
    """List available models for a specific provider.

    Returns 200 {models: [...]}.
    Returns 400 for unknown provider or missing OpenAI key.
    Returns 502 if provider unreachable. Returns 503 if pipeline not ready.
    """
    if not _state._pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    cfg = _state._pipeline.translator.config

    if provider == "ollama":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{cfg.ollama_url}/api/tags")
                resp.raise_for_status()
                return {"models": [m.get("name", "") for m in resp.json().get("models", [])]}
        except Exception as e:
            raise HTTPException(502, f"Cannot reach Ollama: {e}")

    elif provider == "lmstudio":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{cfg.lmstudio_url}/v1/models")
                resp.raise_for_status()
                return {"models": [m.get("id", "") for m in resp.json().get("data", [])]}
        except Exception as e:
            raise HTTPException(502, f"Cannot reach LM Studio: {e}")

    elif provider == "openai":
        if not cfg.openai_api_key:
            raise HTTPException(400, "OPENAI_API_KEY not configured")
        return {"models": ["gpt-5", "gpt-5-mini", "gpt-4o-mini", "gpt-5-nano"]}

    else:
        raise HTTPException(400, f"Unknown provider: {provider}")


class LLMSelectRequest(BaseModel):
    provider: str
    model: str = ""
    url: str = ""
    quick_model: str = ""


@llm_router.post("/select")
async def llm_select(req: LLMSelectRequest):
    """Switch LLM provider and model at runtime.

    Returns 200 with new provider/model/quick_model.
    Returns 400 for unknown provider, missing OpenAI key, or model not found.
    Returns 502 if cannot reach provider to verify model. Returns 503 if pipeline not ready.
    Side effects: mutates translator config and resets provider connection.
    """
    if not _state._pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    if req.provider not in ("ollama", "lmstudio", "openai"):
        raise HTTPException(400, "Provider must be 'ollama', 'lmstudio', or 'openai'")
    if req.provider == "openai" and not _state._pipeline.translator.config.openai_api_key:
        raise HTTPException(400, "OPENAI_API_KEY not configured")
    if req.model:
        cfg = _state._pipeline.translator.config
        if req.provider == "ollama":
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{cfg.ollama_url}/api/tags")
                    resp.raise_for_status()
                    models = [m.get("name", "") for m in resp.json().get("models", [])]
                if req.model not in models:
                    raise HTTPException(400, f"Ollama model not found: {req.model}")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(502, f"Cannot verify Ollama model: {e}")
        if req.provider == "lmstudio":
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{cfg.lmstudio_url}/v1/models")
                    resp.raise_for_status()
                    models = [m.get("id", "") for m in resp.json().get("data", [])]
                if req.model not in models:
                    raise HTTPException(400, f"LM Studio model not found: {req.model}")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(502, f"Cannot verify LM Studio model: {e}")

    if req.quick_model:
        _state._pipeline.translator.config.quick_model = req.quick_model
    _state._pipeline.translator.switch_provider(req.provider, req.model, req.url)
    return {
        "provider": _state._pipeline.translator.config.provider,
        "model": _state._pipeline.translator.config.model,
        "quick_model": _state._pipeline.translator.config.quick_model,
    }


@llm_router.get("/costs")
async def llm_costs():
    """Get OpenAI API cost tracking data.

    Returns 200 {costs: null} when pipeline not initialized.
    Returns 200 {cost_tracking: "Free (local)"} for non-OpenAI providers.
    Returns 200 with provider, model, costs dict for OpenAI.
    """
    if not _state._pipeline:
        return {"costs": None}
    t = _state._pipeline.translator
    if t.config.provider != "openai":
        return {"provider": t.config.provider, "cost_tracking": "Free (local)"}
    return {
        "provider": "openai",
        "model": t.config.openai_model,
        "costs": t.costs,
    }


# --- LM Studio management routes ---

lmstudio_router = APIRouter(prefix="/api/lmstudio", tags=["lmstudio"])


@lmstudio_router.get("/status")
async def lmstudio_status():
    """Return LM Studio running state and loaded model names.

    Returns 200 {running: false, note: "..."} when manager not active.
    Returns 200 {running: bool, models: [...]} when manager is set.
    """
    if _state._lmstudio_manager is None:
        return {"running": False, "models": [], "note": "lmstudio provider not active"}
    running = await _state._lmstudio_manager.is_running()
    return {
        "running": running,
        "models": _state._lmstudio_manager.get_loaded_models(),
    }


@lmstudio_router.post("/restart")
async def lmstudio_restart():
    """Restart LM Studio and reload configured models.

    Returns 200 {status: "restarting"}. Returns 503 if manager not active.
    Side effects: spawns async restart task (non-blocking).
    """
    if _state._lmstudio_manager is None:
        raise HTTPException(503, "LM Studio manager not active")
    asyncio.create_task(_state._lmstudio_manager.restart())
    return {"status": "restarting"}


@lmstudio_router.get("/models")
async def lmstudio_models():
    """Proxy /v1/models from LM Studio (shows all currently loaded models).

    Returns 200 with LM Studio's model list JSON. Returns 502 if unreachable.
    Returns 503 if manager not active.
    """
    if _state._lmstudio_manager is None:
        raise HTTPException(503, "LM Studio manager not active")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"{_state._pipeline.translator.config.lmstudio_url}/v1/models"
                if _state._pipeline else "http://localhost:1234/v1/models"
            )
        if resp.is_success:
            return resp.json()
        raise HTTPException(502, f"LM Studio returned HTTP {resp.status_code}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Cannot reach LM Studio: {exc}")
