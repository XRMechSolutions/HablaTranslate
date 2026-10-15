"""Multi-provider LLM translator service (Ollama, LM Studio, OpenAI, Claude)."""

import asyncio
import contextlib
import json
import logging
import time
import httpx
import re
from server import shared_assets
from server.config import TranslatorConfig, parse_direction, lang_name
from server.models.prompts import (
    get_translator_system_prompt,
    build_translator_user_prompt,
    get_passthrough_retry_prompt,
)
from server.models.schemas import TranslationResult, FlaggedPhrase, CorrectionDetail

logger = logging.getLogger("habla.translator")

# Errors that should NOT be retried (permanent failures)
_PERMANENT_ERRORS = ("model not found", "invalid model", "404", "invalid_api_key")

# OpenAI 429 error codes that mean "out of money", not "slow down"
_QUOTA_ERROR_CODES = {
    "insufficient_quota", "billing_hard_limit_reached", "credit_balance_exhausted",
    "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
}


def _retry_after_seconds(error: Exception) -> float | None:
    """Server-requested wait from retry-after-ms / retry-after, or None."""
    if not isinstance(error, httpx.HTTPStatusError):
        return None
    headers = error.response.headers
    try:
        if "retry-after-ms" in headers:
            value = float(headers["retry-after-ms"]) / 1000.0
        elif "retry-after" in headers:
            value = float(headers["retry-after"])
        else:
            return None
    except ValueError:
        return None
    return value if value >= 0 else None


def _skips_model_fallback(error: Exception) -> bool:
    """Errors a cheaper OpenAI model can't fix: no/invalid key, exhausted quota, no network."""
    if isinstance(error, ValueError) and "invalid_api_key" in str(error):
        return True
    if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 401:
            return True
        if status == 429 and _openai_error_code(error.response) in _QUOTA_ERROR_CODES:
            return True
    return False


def _openai_error_code(response: httpx.Response) -> str:
    """Return error.code (or error.type) from an OpenAI error body, or ''."""
    try:
        err = response.json().get("error") or {}
    except (ValueError, AttributeError):
        return ""
    return str(err.get("code") or err.get("type") or "")


def _openai_request_body(
    model: str, system_prompt: str, user_prompt: str, max_tokens: int, json_mode: bool,
) -> dict:
    """Build a Responses API body using only parameters this model accepts.

    Accepted reasoning efforts and verbosity support come from shared/models.json;
    an unknown model gets no optional parameters at all.
    """
    info = _OPENAI_MODELS.get(model, {})
    body: dict = {
        "model": model,
        "input": user_prompt,
        "max_output_tokens": max_tokens,
        "store": False,
    }
    if system_prompt:
        body["instructions"] = system_prompt
    effort = info.get("effort")
    if effort and effort in info.get("efforts", []):
        body["reasoning"] = {"effort": effort}
    text: dict = {}
    if info.get("verbosity"):
        text["verbosity"] = "low"
    if json_mode:
        text["format"] = {"type": "json_object"}
    if text:
        body["text"] = text
    return body


# Cloud vs local provider classification for fallback logic
_CLOUD_PROVIDERS = {"openai", "claude"}
_LOCAL_PROVIDERS = ("ollama", "lmstudio")  # ordered by preference for fallback

# OpenAI model table (prices, accepted reasoning efforts) shared with the
# Android app: shared/models.json.
_OPENAI_MODELS: dict[str, dict] = shared_assets.models()["openai"]["models"]

# OpenAI pricing per 1M tokens (input, output)
_OPENAI_PRICING = {m: (v["input"], v["output"]) for m, v in _OPENAI_MODELS.items()}

# Claude pricing per 1M tokens (input, output) — Mar 2026
_CLAUDE_PRICING = {
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# OpenAI model fallback order: descend from most capable toward cheapest
# before falling back to local providers
_OPENAI_MODEL_ORDER: list[str] = shared_assets.models()["openai"]["fallback_order"]

# Claude model fallback order: most capable to cheapest
_CLAUDE_MODEL_ORDER = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"]


def _extract_field(text: str, field: str) -> str:
    # Best-effort extraction from malformed JSON blocks
    pattern = rf'"{re.escape(field)}"\s*:\s*"(.*?)"'
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return ""
    value = m.group(1)
    return value.replace("\\n", " ").strip()


def _extract_raw_translation(text: str) -> str:
    """Extract a translation from non-JSON LLM output.

    Handles patterns like:
      Translated: "some text"
      Translation: some text
      "some text"
    """
    # Pattern: Translated/Translation: "text" or Translated/Translation: text
    # Use MULTILINE so $ matches end-of-line (not just end-of-string with DOTALL)
    m = re.search(r'(?:translat(?:ed|ion))\s*[:=]\s*"?(.+?)"?\s*$', text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip().strip('"')
    # Pattern: just a quoted string (entire response is the translation)
    m = re.match(r'^\s*"(.+)"\s*$', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _is_passthrough(source: str, translated: str, direction: str = "") -> bool:
    """Detect when the LLM returned source language instead of translating.

    Two checks:
    1. Word overlap: if > 70% of source words appear in translation.
    2. Script heuristic: if translating *to* English and the output still has
       significant Spanish-specific characters (accents, ¿, ¡, ñ), flag it.
       This catches partial passthrough that word overlap misses.
    """
    if not source or not translated:
        return False
    # Normalize: lowercase, strip punctuation
    def _normalize(t: str) -> set[str]:
        return set(re.sub(r'[^\w\s]', '', t.lower()).split())

    src_words = _normalize(source)
    tgt_words = _normalize(translated)
    if not src_words:
        return False
    overlap = len(src_words & tgt_words) / len(src_words)
    if overlap > 0.7:
        return True

    # Partial passthrough heuristic: Spanish chars in English output
    if direction.endswith("_to_en"):
        # Count chars that are rare in English but common in Spanish
        spanish_chars = len(re.findall(r'[áéíóúüñ¿¡]', translated.lower()))
        total_alpha = max(len(re.findall(r'[a-záéíóúüñ]', translated.lower())), 1)
        if spanish_chars / total_alpha > 0.02 and spanish_chars > 3:
            return True

    return False


def _is_retryable(error: Exception) -> bool:
    """Determine if an error is transient and worth retrying."""
    if isinstance(error, (httpx.TimeoutException, httpx.ConnectError,
                          httpx.RemoteProtocolError)):
        return True
    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code >= 500:
        return True
    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429:
        # Rate limits clear on their own; an exhausted quota or billing limit does not.
        return _openai_error_code(error.response) not in _QUOTA_ERROR_CODES
    msg = str(error).lower()
    if any(p in msg for p in _PERMANENT_ERRORS):
        return False
    if isinstance(error, httpx.HTTPStatusError):
        return False
    return isinstance(error, (OSError, ConnectionError))


class Translator:
    """Translates text using a configurable LLM provider with retry logic.

    Supports four providers: Ollama (local), LM Studio (local), OpenAI (cloud), Claude (cloud).
    Provider can be switched at runtime via switch_provider().

    Retry and fallback strategy:
    - Local providers: 1 retry (retrying rarely helps). Falls back to other local provider.
    - Cloud (OpenAI, Claude): 3 retries with exponential backoff. Falls back through model
      tiers then to local providers.
    - After DEGRADED_THRESHOLD consecutive timeouts, enters degraded mode (shorter timeout).

    Error contract:
    - _call_llm raises on permanent errors (model not found, invalid key).
    - translate() returns TranslationResult with empty translation on total failure (non-fatal).
    - switch_provider raises ValueError for unknown provider names.
    - Tracks metrics (call counts, errors, latency) and cloud costs.
    """

    MAX_RETRIES_LOCAL = 1   # Local LLMs: if they don't respond, retrying won't help
    MAX_RETRIES_CLOUD = 3   # Cloud: transient network issues are common
    INITIAL_BACKOFF = 1.0
    BACKOFF_FACTOR = 2.0
    MAX_BACKOFF = 15.0

    # Health tracking: after this many consecutive timeouts, enter degraded mode
    DEGRADED_THRESHOLD = 3
    # In degraded mode, use this shorter timeout to fail fast
    DEGRADED_TIMEOUT = 8.0
    # A final cut off at max_output_tokens is re-asked once with this budget
    INCOMPLETE_RETRY_TOKENS = 2048
    # Normal timeout for local providers — generous enough for 9B models on mid-range
    # GPUs where prompt processing + generation can take 10-15s per segment.
    LOCAL_TIMEOUT = 30.0

    def __init__(self, config: TranslatorConfig):
        self.config = config
        # Use a shorter read timeout for local providers
        read_timeout = (
            self.LOCAL_TIMEOUT
            if config.provider in _LOCAL_PROVIDERS
            else config.timeout_seconds
        )
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,
                read=read_timeout,
                write=5.0,
                pool=5.0,
            )
        )

        # Metrics
        self._metrics = {
            "requests": 0,
            "successes": 0,
            "retries": 0,
            "failures": 0,
            "timeouts": 0,
            "total_latency_ms": 0.0,
            "last_error": None,
            "last_error_time": None,
            "rate_limited": 0,
        }

        # Health tracking: consecutive timeout counter
        self._consecutive_timeouts = 0
        self._degraded = False

        # Rate limiting (minimum interval between requests)
        self._last_request_time: float = 0.0
        self._rate_limit_lock = asyncio.Lock()

        # Serialize requests to local LLM backends (LM Studio, Ollama).
        # These process requests sequentially on one GPU — concurrent requests
        # just queue internally and cause cascading timeouts when httpx gives up
        # waiting and disconnects, aborting in-progress generation.
        self._local_semaphore = asyncio.Semaphore(1)

        # Cloud cost tracking (OpenAI + Claude)
        self._costs = {
            "session_input_tokens": 0,
            "session_output_tokens": 0,
            "session_cost_usd": 0.0,
            "all_time_input_tokens": 0,
            "all_time_output_tokens": 0,
            "all_time_cost_usd": 0.0,
        }

    async def auto_detect_model(self):
        """Auto-detect model for the active provider if none configured."""
        if self.config.provider == "lmstudio" and not self.config.lmstudio_model:
            try:
                resp = await self.client.get(f"{self.config.lmstudio_url}/v1/models")
                resp.raise_for_status()
                models = [m.get("id", "") for m in resp.json().get("data", []) if m.get("id")]
                if models:
                    self.config.lmstudio_model = models[0]
                    logger.info(f"Auto-detected LM Studio model: {self.config.lmstudio_model}")
                else:
                    logger.warning("LM Studio running but no models loaded")
            except Exception as e:
                logger.warning(f"Could not auto-detect LM Studio model: {e}")

    @property
    def metrics(self) -> dict:
        m = dict(self._metrics)
        m["provider"] = self.config.provider
        m["model"] = self.config.model
        m["degraded"] = self._degraded
        m["consecutive_timeouts"] = self._consecutive_timeouts
        return m

    @property
    def costs(self) -> dict:
        return dict(self._costs)

    def get_session_costs(self) -> dict:
        """Return current session cost snapshot for persistence."""
        return {
            "session_input_tokens": self._costs["session_input_tokens"],
            "session_output_tokens": self._costs["session_output_tokens"],
            "session_cost_usd": self._costs["session_cost_usd"],
        }

    async def load_all_time_costs(self):
        """Load cumulative cost totals from all past sessions in the DB."""
        from server.db.database import get_db
        try:
            db = await get_db()
            rows = await db.execute_fetchall(
                """SELECT COALESCE(SUM(llm_input_tokens), 0) as total_input,
                          COALESCE(SUM(llm_output_tokens), 0) as total_output,
                          COALESCE(SUM(llm_cost_usd), 0.0) as total_cost
                   FROM sessions
                   WHERE llm_provider IN ('openai', 'claude')"""
            )
            row = rows[0] if rows else None
            if row:
                self._costs["all_time_input_tokens"] = row["total_input"]
                self._costs["all_time_output_tokens"] = row["total_output"]
                self._costs["all_time_cost_usd"] = row["total_cost"]
                if row["total_cost"] > 0:
                    logger.info(
                        f"Loaded all-time cloud costs: ${row['total_cost']:.4f} "
                        f"({row['total_input'] + row['total_output']} tokens)"
                    )
        except Exception as e:
            logger.warning(f"Could not load all-time costs from DB: {e}")

    def switch_provider(self, provider: str, model: str = "", url: str = ""):
        """Switch LLM provider and model at runtime."""
        if provider not in ("ollama", "lmstudio", "openai", "claude"):
            raise ValueError(f"Unknown provider: {provider}")
        self.config.provider = provider
        if provider == "ollama" and model:
            self.config.ollama_model = model
        elif provider == "lmstudio" and model:
            self.config.lmstudio_model = model
        elif provider == "openai" and model:
            self.config.openai_model = model
        elif provider == "claude" and model:
            self.config.anthropic_model = model
        if url:
            if provider == "ollama":
                self.config.ollama_url = url
            elif provider == "lmstudio":
                self.config.lmstudio_url = url
        # Reset session costs on provider switch
        self._costs["session_input_tokens"] = 0
        self._costs["session_output_tokens"] = 0
        self._costs["session_cost_usd"] = 0.0
        logger.info(f"Switched to {provider}/{self.config.model}")

    async def _call_llm(
        self, system_prompt: str, user_prompt: str, retries: int | None = None,
        max_tokens: int = 1024, temperature: float | None = None, json_mode: bool = True,
        bypass_local_sem: bool = False,
    ) -> str:
        """Route LLM call to the active provider with fallback.

        Fallback rules:
        - User-selected provider is always tried first (with retries).
        - If selected provider is cloud (OpenAI) and fails:
          1. Try the other OpenAI models in shared/models.json fallback_order
          2. Then try local providers (Ollama, then LM Studio)
        - If selected provider is local and fails, NEVER fall back to cloud
          (to avoid unexpected costs).
        """
        provider = self.config.provider
        temp = temperature if temperature is not None else self.config.temperature

        # Try the user-selected provider first
        try:
            return await self._call_provider(
                provider, system_prompt, user_prompt, retries, max_tokens, temp, json_mode,
                bypass_local_sem=bypass_local_sem,
            )
        except Exception as primary_error:
            # Only fall back from cloud → local, never local → cloud
            if provider not in _CLOUD_PROVIDERS or not self.config.fallback_to_local:
                raise

            # Cloud fallback: try cheaper models for this provider, then local.
            # A missing/invalid key, exhausted quota or no network fails every
            # model the same way, so go straight to local providers.
            skip_models = _skips_model_fallback(primary_error)
            if provider == "claude":
                model_order = _CLAUDE_MODEL_ORDER
                selected_model = self.config.anthropic_model
                model_kwarg = "claude_model"
            else:
                model_order = _OPENAI_MODEL_ORDER
                selected_model = self.config.openai_model
                model_kwarg = "openai_model"

            try:
                idx = model_order.index(selected_model)
            except ValueError:
                idx = -1

            # Try each cheaper model (those after the selected one in the order)
            for model in ([] if skip_models else model_order[idx + 1:]):
                logger.warning(
                    f"{provider} {selected_model} failed, trying {model}: {primary_error}"
                )
                try:
                    text = await self._call_provider(
                        provider, system_prompt, user_prompt,
                        retries=0, max_tokens=max_tokens,
                        temperature=temp, json_mode=json_mode,
                        **{model_kwarg: model},
                    )
                    logger.info(f"{provider} fallback to {model} succeeded")
                    return text
                except Exception as model_err:
                    logger.warning(f"{provider} {model} also failed: {model_err}")
                    continue

            # All cloud models failed — try local providers (single attempt each)
            for fallback in _LOCAL_PROVIDERS:
                logger.warning(
                    f"All {provider} models failed, falling back to {fallback}: {primary_error}"
                )
                try:
                    text = await self._call_provider(
                        fallback, system_prompt, user_prompt,
                        retries=0, max_tokens=max_tokens,
                        temperature=temp, json_mode=json_mode,
                    )
                    logger.info(f"Fallback to {fallback} succeeded")
                    return text
                except Exception as fb_err:
                    logger.warning(f"Fallback {fallback} also failed: {fb_err}")
                    continue

            # All fallbacks exhausted — raise the original error
            raise primary_error

    async def _call_provider(
        self, provider: str, system_prompt: str, user_prompt: str,
        retries: int | None = None, max_tokens: int = 1024,
        temperature: float = 0.3, json_mode: bool = True,
        openai_model: str = "",
        lmstudio_model: str = "",
        claude_model: str = "",
        bypass_local_sem: bool = False,
    ) -> str:
        """Call a specific provider with retry logic. Returns raw text response."""
        # Enforce minimum interval between requests
        async with self._rate_limit_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            min_interval = self.config.rate_limit_interval
            if elapsed < min_interval:
                delay = min_interval - elapsed
                self._metrics["rate_limited"] += 1
                logger.debug(f"Rate limiter: waiting {delay:.2f}s before LLM call")
                await asyncio.sleep(delay)
            self._last_request_time = time.monotonic()

        if retries is not None:
            max_retries = retries
        elif self._degraded:
            # In degraded mode, don't retry — fail fast
            max_retries = 0
        elif provider in _CLOUD_PROVIDERS:
            max_retries = self.MAX_RETRIES_CLOUD
        else:
            max_retries = self.MAX_RETRIES_LOCAL
        last_error = None

        # Serialize local provider requests — they process one at a time on the GPU.
        # Without this, concurrent requests queue inside LM Studio and the httpx read
        # timeout fires while waiting behind earlier requests, causing the connection
        # to close ("Client disconnected") and aborting in-progress generation.
        # bypass_local_sem allows low-priority calls (partials, topic summaries) to
        # skip the queue — they'll either get served quickly or timeout harmlessly.
        use_sem = provider in _LOCAL_PROVIDERS and not bypass_local_sem
        sem = self._local_semaphore if use_sem else contextlib.nullcontext()

        async with sem:
            for attempt in range(max_retries + 1):
                self._metrics["requests"] += 1
                start = time.monotonic()

                try:
                    if provider == "ollama":
                        coro = self._call_ollama(system_prompt, user_prompt, temperature, max_tokens, json_mode)
                    elif provider == "lmstudio":
                        coro = self._call_lmstudio(system_prompt, user_prompt, temperature, max_tokens, json_mode, model_override=lmstudio_model)
                    elif provider == "openai":
                        coro = self._call_openai(system_prompt, user_prompt, temperature, max_tokens, json_mode, model_override=openai_model)
                    elif provider == "claude":
                        coro = self._call_claude(system_prompt, user_prompt, temperature, max_tokens, model_override=claude_model)
                    else:
                        raise ValueError(f"Unknown provider: {provider}")

                    # In degraded mode, use a shorter timeout to fail fast
                    if self._degraded:
                        try:
                            text = await asyncio.wait_for(coro, timeout=self.DEGRADED_TIMEOUT)
                        except asyncio.TimeoutError:
                            raise httpx.ReadTimeout(
                                f"Degraded mode timeout ({self.DEGRADED_TIMEOUT}s)"
                            )
                    else:
                        text = await coro

                    latency = (time.monotonic() - start) * 1000
                    self._metrics["successes"] += 1
                    self._metrics["total_latency_ms"] += latency

                    # Success clears timeout tracking
                    if self._consecutive_timeouts > 0:
                        if self._degraded:
                            logger.info(
                                f"Provider recovered after {self._consecutive_timeouts} "
                                f"consecutive timeouts — exiting degraded mode"
                            )
                        self._consecutive_timeouts = 0
                        self._degraded = False

                    return text

                except Exception as e:
                    last_error = e
                    latency = (time.monotonic() - start) * 1000
                    self._metrics["total_latency_ms"] += latency

                    if isinstance(e, httpx.TimeoutException):
                        self._metrics["timeouts"] += 1
                        self._consecutive_timeouts += 1
                        if (self._consecutive_timeouts >= self.DEGRADED_THRESHOLD
                                and not self._degraded):
                            self._degraded = True
                            logger.warning(
                                f"Provider {provider} entered degraded mode after "
                                f"{self._consecutive_timeouts} consecutive timeouts — "
                                f"skipping retries, using {self.DEGRADED_TIMEOUT}s timeout"
                            )

                    if not _is_retryable(e):
                        logger.error(f"Permanent {provider} error (no retry): {e}")
                        self._metrics["failures"] += 1
                        self._metrics["last_error"] = str(e)
                        self._metrics["last_error_time"] = time.time()
                        raise

                    if attempt < max_retries:
                        delay = min(
                            self.INITIAL_BACKOFF * (self.BACKOFF_FACTOR ** attempt),
                            self.MAX_BACKOFF,
                        )
                        retry_after = _retry_after_seconds(e)
                        if retry_after is not None:
                            if retry_after > self.MAX_BACKOFF:
                                # A translation that waits this long is stale; let the
                                # caller's model fallback take over instead.
                                self._metrics["failures"] += 1
                                self._metrics["last_error"] = str(e)
                                self._metrics["last_error_time"] = time.time()
                                raise
                            delay = max(delay, retry_after)
                        self._metrics["retries"] += 1
                        logger.warning(
                            f"{provider} request failed (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {delay:.1f}s: {e}"
                        )
                        await asyncio.sleep(delay)

        self._metrics["failures"] += 1
        self._metrics["last_error"] = str(last_error)
        self._metrics["last_error_time"] = time.time()
        raise last_error

    async def _call_ollama(
        self, system_prompt: str, user_prompt: str,
        temperature: float, max_tokens: int, json_mode: bool,
    ) -> str:
        """Ollama: POST /api/generate with prompt + system fields."""
        body = {
            "model": self.config.ollama_model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            body["format"] = "json"

        response = await self.client.post(
            f"{self.config.ollama_url}/api/generate", json=body,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama error {response.status_code}: {response.text}")
            raise
        return response.json().get("response", "{}")

    async def _call_lmstudio(
        self, system_prompt: str, user_prompt: str,
        temperature: float, max_tokens: int, json_mode: bool,
        model_override: str = "",
    ) -> str:
        """LM Studio: POST /v1/chat/completions (OpenAI-compatible)."""
        model_name = model_override or self.config.lmstudio_model
        model_lower = model_name.lower()

        # TranslateGemma uses a special structured content format — not a
        # normal chat model.  Only use it for translation-like requests.
        # Non-translation calls (topic summary, study summary) fall through
        # to the standard chat path — they'll fail on TranslateGemma but
        # that's expected; TranslateGemma should only be a quick model.
        is_translategemma = "translategemma" in model_lower
        if is_translategemma:
            # Detect if this looks like a translation request
            combined = (system_prompt + " " + user_prompt).lower()
            is_translation = (
                "translat" in combined
                or "transcript" in combined
                or "spanish" in combined
                or "english" in combined
            )
            if is_translation:
                return await self._call_lmstudio_translategemma(
                    model_name, system_prompt, user_prompt, temperature, max_tokens,
                )
            else:
                # Skip TranslateGemma for non-translation tasks
                logger.warning(
                    f"Skipping TranslateGemma for non-translation task "
                    f"(prompt: '{user_prompt[:50]}...')"
                )
                raise ValueError(
                    "TranslateGemma cannot handle non-translation tasks. "
                    "Use a general-purpose LLM as the main model."
                )

        # Disable thinking/reasoning mode for Qwen3+ models — translation
        # doesn't benefit from chain-of-thought and it burns 10-20s of latency.
        if "qwen3" in model_lower:
            system_prompt = (
                system_prompt
                + "\nDo NOT output any thinking, reasoning, or analysis. "
                "Respond ONLY with the JSON object."
            )
            user_prompt = user_prompt + " /no_think"

        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        response = await self.client.post(
            f"{self.config.lmstudio_url}/v1/chat/completions", json=body,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"LM Studio error {response.status_code}: {response.text}")
            raise
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def _call_lmstudio_translategemma(
        self, model_name: str, system_prompt: str, user_prompt: str,
        temperature: float, max_tokens: int,
    ) -> str:
        """TranslateGemma requires structured content with language codes."""
        # Infer language codes from system prompt or user prompt
        src_code, tgt_code = "es", "en"  # defaults
        combined = (system_prompt + " " + user_prompt).lower()
        if "spanish to english" in combined or "spanish-to-english" in combined:
            src_code, tgt_code = "es", "en"
        elif "english to spanish" in combined or "english-to-spanish" in combined:
            src_code, tgt_code = "en", "es"

        # Extract the actual text to translate from various prompt formats:
        #   Full prompt: ... Transcript: "some text" ...
        #   Quick prompt: Translate this Spanish to ...\n\nsome text
        transcript = user_prompt
        if 'Transcript: "' in transcript:
            start = transcript.index('Transcript: "') + len('Transcript: "')
            end = transcript.index('"', start)
            transcript = transcript[start:end]
        elif "\n\n" in transcript:
            # Quick translate format: instructions\n\nraw text
            transcript = transcript.split("\n\n", 1)[-1].strip()

        body = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "source_lang_code": src_code,
                        "target_lang_code": tgt_code,
                        "text": transcript,
                        "image": None,
                    }],
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        response = await self.client.post(
            f"{self.config.lmstudio_url}/v1/chat/completions", json=body,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"LM Studio error {response.status_code}: {response.text}")
            raise
        data = response.json()
        translated = data["choices"][0]["message"]["content"]

        # If called from _quick_translate (no system prompt), return plain text
        if not system_prompt:
            return translated

        # Full translation path — wrap in our JSON format
        import json as _json
        return _json.dumps({
            "corrected": transcript,
            "translated": translated,
            "flagged_phrases": [],
            "confidence": 0.85,
            "speaker_hint": None,
            "is_correction": False,
            "correction_detail": None,
        })

    async def _call_openai(
        self, system_prompt: str, user_prompt: str,
        temperature: float, max_tokens: int, json_mode: bool,
        model_override: str = "",
    ) -> str:
        """OpenAI: POST /v1/responses (Responses API, not Chat Completions)."""
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key not set (invalid_api_key)")
        model = model_override or self.config.openai_model
        body = _openai_request_body(model, system_prompt, user_prompt, max_tokens, json_mode)

        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json",
        }

        response = await self.client.post(
            "https://api.openai.com/v1/responses",
            json=body, headers=headers,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            logger.error(f"OpenAI error {response.status_code}: {response.text}")
            raise
        data = response.json()

        # Track costs from usage
        usage = data.get("usage", {})
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cached_tokens = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
            self._track_cloud_cost(input_tokens, output_tokens, model, "openai", cached_tokens)

        if data.get("status") == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason", "unknown")
            logger.warning(f"OpenAI {model} response incomplete ({reason})")
            if reason == "max_output_tokens" and json_mode and max_tokens < self.INCOMPLETE_RETRY_TOKENS:
                # Truncated JSON is useless; ask once more with room to finish.
                return await self._call_openai(
                    system_prompt, user_prompt, temperature, self.INCOMPLETE_RETRY_TOKENS,
                    json_mode, model_override=model,
                )

        # Extract text: find output item with type "message", then content with type "output_text"
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        return content.get("text", "{}")

        logger.warning(f"OpenAI: no output_text found in response")
        return "{}"

    async def _call_claude(
        self, system_prompt: str, user_prompt: str,
        temperature: float, max_tokens: int,
        model_override: str = "",
    ) -> str:
        """Claude: POST https://api.anthropic.com/v1/messages (Messages API)."""
        model = model_override or self.config.anthropic_model
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }

        headers = {
            "x-api-key": self.config.anthropic_api_key,
            "anthropic-version": "2024-10-22",
            "Content-Type": "application/json",
        }

        response = await self.client.post(
            "https://api.anthropic.com/v1/messages",
            json=body, headers=headers,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            logger.error(f"Claude error {response.status_code}: {response.text}")
            raise
        data = response.json()

        # Track costs from usage
        usage = data.get("usage", {})
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            self._track_cloud_cost(input_tokens, output_tokens, model, "claude")

        # Extract text from content blocks
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block.get("text", "{}")

        logger.warning("Claude: no text block found in response")
        return "{}"

    def _track_cloud_cost(
        self, input_tokens: int, output_tokens: int, model: str = "", provider: str = "openai",
        cached_tokens: int = 0,
    ):
        """Update running cost totals for cloud provider usage.

        cached_tokens is the part of input_tokens served from the prompt cache
        (OpenAI usage.input_tokens_details.cached_tokens), billed at the cached rate.
        """
        if provider == "claude":
            pricing = _CLAUDE_PRICING.get(model or self.config.anthropic_model, (3.00, 15.00))
            cost = (input_tokens / 1_000_000 * pricing[0]) + (output_tokens / 1_000_000 * pricing[1])
        else:
            model = model or self.config.openai_model
            info = _OPENAI_MODELS.get(model)
            if info is None:
                logger.warning(f"No price for OpenAI model {model}; cost not counted")
                cost = 0.0
            else:
                cached = min(cached_tokens, input_tokens)
                cost = (
                    (input_tokens - cached) / 1_000_000 * info["input"]
                    + cached / 1_000_000 * info["cached_input"]
                    + output_tokens / 1_000_000 * info["output"]
                )

        self._costs["session_input_tokens"] += input_tokens
        self._costs["session_output_tokens"] += output_tokens
        self._costs["session_cost_usd"] += cost
        self._costs["all_time_input_tokens"] += input_tokens
        self._costs["all_time_output_tokens"] += output_tokens
        self._costs["all_time_cost_usd"] += cost

    async def translate(
        self,
        transcript: str,
        speaker_label: str,
        direction: str,
        mode: str,
        context_exchanges: list[dict],
        topic_summary: str = "",
    ) -> TranslationResult:
        """Send transcript through LLM for translation + correction + idiom detection."""
        system_prompt = get_translator_system_prompt(direction, mode)
        user_prompt = build_translator_user_prompt(
            transcript=transcript,
            speaker_label=speaker_label,
            context_exchanges=context_exchanges,
            topic_summary=topic_summary,
            max_context=self.config.max_context_exchanges,
        )

        try:
            raw_text = await self._call_llm(
                system_prompt, user_prompt,
                max_tokens=1024, json_mode=True,
            )
            result = self._parse_response(raw_text, transcript)

            # Detect passthrough: LLM returned source language as translation
            if _is_passthrough(transcript, result.translated, direction):
                logger.warning(
                    f"LLM passed source through untranslated: '{transcript[:60]}' "
                    f"-> '{result.translated[:60]}'"
                )
                # Retry with a forceful prompt
                retry_result = await self._retry_passthrough(
                    transcript, direction, system_prompt,
                )
                if retry_result:
                    return retry_result
                # Retry also failed — degrade confidence to flag this
                result.confidence = min(result.confidence, 0.2)

            return result

        except httpx.TimeoutException:
            logger.warning(f"{self.config.provider} timeout — returning raw transcript")
            return TranslationResult(
                corrected=transcript,
                translated=f"[translation timeout] {transcript}",
                confidence=0.0,
            )
        except Exception as e:
            logger.error(f"Translation failed ({self.config.provider}): {e}")
            return TranslationResult(
                corrected=transcript,
                translated=f"[translation error] {transcript}",
                confidence=0.0,
            )

    async def _retry_passthrough(
        self, transcript: str, direction: str, system_prompt: str,
    ) -> TranslationResult | None:
        """Retry translation with a forceful prompt when passthrough is detected."""
        src_code, tgt_code = parse_direction(direction)
        forceful_prompt = get_passthrough_retry_prompt(
            transcript, lang_name(src_code), lang_name(tgt_code),
        )

        try:
            raw_text = await self._call_llm(
                system_prompt, forceful_prompt,
                retries=0, max_tokens=512, json_mode=True,
            )
            result = self._parse_response(raw_text, transcript)
            if not _is_passthrough(transcript, result.translated, direction):
                logger.info(f"Passthrough retry succeeded: '{result.translated[:60]}'")
                result.confidence = min(result.confidence, 0.5)  # cap confidence on retry
                return result
            logger.warning("Passthrough retry also failed — LLM cannot translate this text")
        except Exception as e:
            logger.warning(f"Passthrough retry error: {e}")

        return None

    def _parse_response(self, raw: str, original: str) -> TranslationResult:
        """Parse the JSON response from the LLM."""
        cleaned = raw.strip()
        # Strip chain-of-thought blocks if present (tagged or plain text)
        if "<think>" in cleaned and "</think>" in cleaned:
            cleaned = cleaned.split("</think>", 1)[-1].strip()
        # Qwen3 models sometimes output "Thinking Process:\n..." as plain text
        # before the JSON. Strip everything before the first '{'.
        if cleaned.lower().startswith("thinking"):
            brace = cleaned.find("{")
            if brace != -1:
                cleaned = cleaned[brace:]
        if "```" in cleaned:
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        def _extract_first_json(text: str) -> str:
            start = text.find("{")
            if start == -1:
                return text
            in_str = False
            escape = False
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == "\"":
                        in_str = False
                else:
                    if ch == "\"":
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return text[start:i + 1]
            # If we never closed, return the slice from start (best effort)
            return text[start:]

        candidate = _extract_first_json(cleaned)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            # Fallback: attempt to salvage corrected/translated via regex
            corrected = _extract_field(cleaned, "corrected")
            translated = _extract_field(cleaned, "translated")
            if corrected or translated:
                return TranslationResult(
                    corrected=corrected or original,
                    translated=translated or cleaned.strip(),
                    confidence=0.3,
                )
            # Try to extract translation from common non-JSON patterns:
            #   "Translated: ..." / "Translation: ..." / just a plain sentence
            extracted = _extract_raw_translation(cleaned)
            if extracted:
                logger.warning(f"Extracted translation from non-JSON LLM output: {cleaned[:100]}")
                return TranslationResult(
                    corrected=original,
                    translated=extracted,
                    confidence=0.3,
                )
            logger.warning(f"Failed to parse LLM JSON: {cleaned[:200]}")
            return TranslationResult(
                corrected=original,
                translated=cleaned.strip(),
                confidence=0.1,
            )
        if not isinstance(data, dict):
            logger.warning("LLM JSON parsed to non-object; using raw text translation")
            return TranslationResult(
                corrected=original,
                translated=cleaned.strip(),
                confidence=0.1,
            )

        flagged = []
        for fp in data.get("flagged_phrases", []):
            flagged.append(FlaggedPhrase(
                phrase=fp.get("phrase", ""),
                literal=fp.get("literal"),
                meaning=fp.get("meaning", ""),
                type=fp.get("type", "idiom"),
                save_worthy=fp.get("save_worthy", True),
                source="llm",
            ))

        correction = None
        if data.get("correction_detail"):
            cd = data["correction_detail"]
            if isinstance(cd, str):
                correction = CorrectionDetail(
                    wrong="",
                    right="",
                    explanation=cd,
                )
            else:
                correction = CorrectionDetail(
                    wrong=cd.get("wrong", ""),
                    right=cd.get("right", ""),
                    explanation=cd.get("explanation", ""),
                )

        return TranslationResult(
            corrected=data.get("corrected") or original,
            translated=data.get("translated") or original,
            flagged_phrases=flagged,
            confidence=data.get("confidence", 0.5),
            speaker_hint=data.get("speaker_hint"),
            is_correction=data.get("is_correction", False),
            correction_detail=correction,
        )

    async def update_topic_summary(
        self,
        previous_summary: str,
        latest_source: str,
        latest_translation: str,
        speaker_label: str,
    ) -> str:
        """Update the rolling topic summary (uses same LLM, cheap call)."""
        from server.models.prompts import get_topic_summary_prompt

        prompt = get_topic_summary_prompt(
            previous_summary, latest_source, latest_translation, speaker_label
        )

        try:
            text = await self._call_llm(
                "", prompt,
                retries=0, max_tokens=200, temperature=0.2, json_mode=False,
                bypass_local_sem=True,
            )
            return text.strip() or previous_summary
        except Exception as e:
            logger.debug(f"Topic summary update skipped: {e}")
            return previous_summary

    async def close(self):
        await self.client.aclose()
