"""Startup and runtime health checks for Habla services."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import httpx

logger = logging.getLogger("habla.health")


class ComponentStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
class HealthCheck:
    component: str
    status: ComponentStatus
    message: str = ""
    latency_ms: float = 0.0


@dataclass
class SystemHealth:
    checks: list[HealthCheck] = field(default_factory=list)
    overall: ComponentStatus = ComponentStatus.OK

    def add(self, check: HealthCheck):
        self.checks.append(check)
        if check.status == ComponentStatus.DOWN:
            self.overall = ComponentStatus.DOWN
        elif check.status == ComponentStatus.DEGRADED and self.overall != ComponentStatus.DOWN:
            self.overall = ComponentStatus.DEGRADED

    def to_dict(self) -> dict:
        return {
            "status": self.overall.value,
            "components": {
                c.component: {
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": round(c.latency_ms, 1),
                }
                for c in self.checks
            },
        }


async def check_ollama(url: str, model: str) -> HealthCheck:
    """Check Ollama is reachable and has the required model loaded."""
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Check if Ollama is reachable
            resp = await client.get(f"{url}/api/tags")
            resp.raise_for_status()
            latency = (time.monotonic() - start) * 1000

            # Check if the required model is available
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            # Ollama model names can have :latest suffix
            found = any(
                model in name or name.startswith(model.split(":")[0])
                for name in model_names
            )

            if found:
                return HealthCheck("ollama", ComponentStatus.OK,
                                   f"Model {model} available", latency)
            else:
                return HealthCheck("ollama", ComponentStatus.DEGRADED,
                                   f"Ollama reachable but model '{model}' not found. "
                                   f"Available: {model_names}. Run: ollama pull {model}",
                                   latency)
    except httpx.ConnectError:
        return HealthCheck("ollama", ComponentStatus.DOWN,
                           f"Cannot connect to Ollama at {url}. Is it running?")
    except Exception as e:
        return HealthCheck("ollama", ComponentStatus.DOWN, str(e))


async def check_ffmpeg() -> HealthCheck:
    """Check ffmpeg is installed and reachable."""
    import asyncio
    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        latency = (time.monotonic() - start) * 1000

        if proc.returncode == 0:
            # Extract version from first line
            version_line = stdout.decode().split("\n")[0] if stdout else "unknown"
            return HealthCheck("ffmpeg", ComponentStatus.OK, version_line, latency)
        else:
            return HealthCheck("ffmpeg", ComponentStatus.DOWN, "ffmpeg returned non-zero exit code")
    except FileNotFoundError:
        return HealthCheck("ffmpeg", ComponentStatus.DOWN,
                           "ffmpeg not found in PATH. Install: apt install ffmpeg")
    except Exception as e:
        return HealthCheck("ffmpeg", ComponentStatus.DOWN, str(e))


async def check_database(db_path) -> HealthCheck:
    """Check database is accessible and tables exist."""
    start = time.monotonic()
    try:
        from server.db.database import get_db
        db = await get_db()
        # Quick integrity check - just verify we can query
        cursor = await db.execute("SELECT COUNT(*) FROM vocab")
        count = (await cursor.fetchone())[0]
        latency = (time.monotonic() - start) * 1000
        return HealthCheck("database", ComponentStatus.OK,
                           f"OK, {count} vocab items", latency)
    except RuntimeError as e:
        return HealthCheck("database", ComponentStatus.DOWN, f"Not initialized: {e}")
    except Exception as e:
        return HealthCheck("database", ComponentStatus.DOWN, str(e))


def check_whisperx(model) -> HealthCheck:
    """Check if WhisperX model is loaded."""
    if model is not None:
        return HealthCheck("whisperx", ComponentStatus.OK, "Model loaded")
    return HealthCheck("whisperx", ComponentStatus.DEGRADED,
                       "Not loaded - text-only mode active")


def check_diarization(pipeline) -> HealthCheck:
    """Check if Pyannote diarization pipeline is loaded."""
    if pipeline is not None:
        return HealthCheck("diarization", ComponentStatus.OK, "Pipeline loaded")
    return HealthCheck("diarization", ComponentStatus.DEGRADED,
                       "Not loaded - speaker attribution disabled")


async def check_lmstudio(url: str) -> HealthCheck:
    """Check LM Studio is reachable and list loaded models."""
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{url}/v1/models")
            resp.raise_for_status()
            latency = (time.monotonic() - start) * 1000
            models = [m.get("id", "") for m in resp.json().get("data", [])]
            if models:
                return HealthCheck("lmstudio", ComponentStatus.OK,
                                   f"{len(models)} model(s) loaded", latency)
            return HealthCheck("lmstudio", ComponentStatus.DEGRADED,
                               "LM Studio reachable but no models loaded", latency)
    except httpx.ConnectError:
        return HealthCheck("lmstudio", ComponentStatus.DOWN,
                           f"Cannot connect to LM Studio at {url}")
    except Exception as e:
        return HealthCheck("lmstudio", ComponentStatus.DOWN, str(e))


def check_openai_key(api_key: str) -> HealthCheck:
    """Check if OpenAI API key is configured."""
    if api_key:
        return HealthCheck("openai", ComponentStatus.OK, "Configured")
    return HealthCheck("openai", ComponentStatus.DEGRADED,
                       "OPENAI_API_KEY not set - cloud translation unavailable")


async def check_hf_token(token: str) -> HealthCheck:
    """Check if HuggingFace token is set."""
    if token:
        return HealthCheck("hf_token", ComponentStatus.OK, "Set")
    return HealthCheck("hf_token", ComponentStatus.DEGRADED,
                       "HF_TOKEN not set - diarization will be disabled")


async def _check_active_llm(translator_config) -> HealthCheck:
    """Check health of the currently active LLM provider."""
    provider = translator_config.provider
    if provider == "ollama":
        return await check_ollama(translator_config.ollama_url, translator_config.ollama_model)
    elif provider == "lmstudio":
        return await check_lmstudio(translator_config.lmstudio_url)
    elif provider == "openai":
        return check_openai_key(translator_config.openai_api_key)
    return HealthCheck("llm", ComponentStatus.DOWN, f"Unknown provider: {provider}")


async def run_startup_checks(config) -> SystemHealth:
    """Run all startup health checks. Logs results and returns health status."""
    health = SystemHealth()

    logger.info("Running startup health checks...")

    # Check active LLM provider
    llm = await _check_active_llm(config.translator)
    health.add(llm)

    ffmpeg = await check_ffmpeg()
    health.add(ffmpeg)

    db = await check_database(config.db_path)
    health.add(db)

    hf = await check_hf_token(config.diarization.hf_token)
    health.add(hf)

    # Log results
    for check in health.checks:
        if check.status == ComponentStatus.OK:
            logger.info(f"  [{check.status.value}] {check.component}: {check.message}")
        elif check.status == ComponentStatus.DEGRADED:
            logger.warning(f"  [{check.status.value}] {check.component}: {check.message}")
        else:
            logger.error(f"  [{check.status.value}] {check.component}: {check.message}")

    logger.info(f"Startup health: {health.overall.value} (LLM: {config.translator.provider}/{config.translator.model})")
    return health


_runtime_cache: dict = {"result": None, "timestamp": 0.0}
_RUNTIME_CACHE_TTL = 15.0  # seconds


async def run_runtime_checks(pipeline) -> SystemHealth:
    """Run health checks against the live pipeline (for /health endpoint).

    Expensive checks (ffmpeg subprocess, LLM network calls) are cached for
    15 seconds to avoid per-request overhead under frequent probing.
    """
    now = time.monotonic()
    if _runtime_cache["result"] and (now - _runtime_cache["timestamp"]) < _RUNTIME_CACHE_TTL:
        return _runtime_cache["result"]

    health = SystemHealth()

    llm = await _check_active_llm(pipeline.config.translator)
    health.add(llm)

    ffmpeg = await check_ffmpeg()
    health.add(ffmpeg)

    db = await check_database(pipeline.config.db_path)
    health.add(db)

    health.add(check_whisperx(pipeline._whisperx_model))
    health.add(check_diarization(pipeline._diarize_pipeline))

    _runtime_cache["result"] = health
    _runtime_cache["timestamp"] = now

    return health


async def run_llm_health_monitor(
    get_config,
    get_session_fn,
    interval: float = 60.0,
):
    """Background task: periodically check active LLM provider health.

    Sends WebSocket error to connected client when provider goes down.
    Logs recovery when provider comes back.
    """
    last_status = ComponentStatus.OK

    while True:
        await asyncio.sleep(interval)
        try:
            config = get_config()
            check = await _check_active_llm(config)

            if check.status != ComponentStatus.OK and last_status == ComponentStatus.OK:
                logger.warning(
                    f"LLM provider {config.provider} is {check.status.value}: "
                    f"{check.message}"
                )
                session = get_session_fn()
                if session:
                    await session._send({
                        "type": "error",
                        "message": f"LLM provider ({config.provider}) is unavailable: {check.message}",
                    })

            elif check.status == ComponentStatus.OK and last_status != ComponentStatus.OK:
                logger.info(
                    f"LLM provider {config.provider} recovered "
                    f"(was {last_status.value})"
                )
                session = get_session_fn()
                if session:
                    await session._send({
                        "type": "status",
                        "message": f"LLM provider ({config.provider}) is back online",
                    })

            last_status = check.status

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"LLM health monitor error: {e}")
