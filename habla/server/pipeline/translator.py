"""Multi-provider LLM translator service (Ollama, LM Studio, OpenAI)."""

import asyncio
import json
import logging
import time
import httpx
import re
from server.config import TranslatorConfig
from server.models.prompts import (
    get_translator_system_prompt,
    build_translator_user_prompt,
)
from server.models.schemas import TranslationResult, FlaggedPhrase, CorrectionDetail

logger = logging.getLogger("habla.translator")

# Errors that should NOT be retried (permanent failures)
_PERMANENT_ERRORS = ("model not found", "invalid model", "404", "invalid_api_key")

# Cloud vs local provider classification for fallback logic
_CLOUD_PROVIDERS = {"openai"}
_LOCAL_PROVIDERS = ("ollama", "lmstudio")  # ordered by preference for fallback

# OpenAI pricing per 1M tokens (input, output) — Feb 2026
_OPENAI_PRICING = {
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5": (1.25, 10.00),
}

# OpenAI model fallback order: descend from selected model toward cheapest
# before falling back to local providers
_OPENAI_MODEL_ORDER = ["gpt-5", "gpt-5-mini", "gpt-4o-mini", "gpt-5-nano"]


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


def _is_passthrough(source: str, translated: str) -> bool:
    """Detect when the LLM returned the source language instead of translating.

    Compares normalized texts. If > 70% of words overlap, it's likely a passthrough.
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
    return overlap > 0.7


def _is_retryable(error: Exception) -> bool:
    """Determine if an error is transient and worth retrying."""
    if isinstance(error, (httpx.TimeoutException, httpx.ConnectError,
                          httpx.RemoteProtocolError)):
        return True
    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code >= 500:
        return True
    msg = str(error).lower()
    if any(p in msg for p in _PERMANENT_ERRORS):
        return False
    if isinstance(error, httpx.HTTPStatusError):
        return False
    return isinstance(error, (OSError, ConnectionError))


class Translator:
    """Translates text using a configurable LLM provider with retry logic.

    Supports three providers: Ollama (local), LM Studio (local), OpenAI (cloud).
    Provider can be switched at runtime via switch_provider().

    Retry and fallback strategy:
    - Local providers: 1 retry (retrying rarely helps). Falls back to other local provider.
    - Cloud (OpenAI): 3 retries with exponential backoff. Falls back through model tiers
      (gpt-5 -> gpt-5-mini -> gpt-4o-mini -> gpt-5-nano) then to local providers.
    - After DEGRADED_THRESHOLD consecutive timeouts, enters degraded mode (shorter timeout).

    Error contract:
    - _call_llm raises on permanent errors (model not found, invalid key).
    - translate() returns TranslationResult with empty translation on total failure (non-fatal).
    - switch_provider raises ValueError for unknown provider names.
    - Tracks metrics (call counts, errors, latency) and OpenAI costs.
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
    # Normal timeout for local providers (shorter than the 30s config default)
    LOCAL_TIMEOUT = 15.0

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
        }

        # Health tracking: consecutive timeout counter
        self._consecutive_timeouts = 0
        self._degraded = False

        # OpenAI cost tracking
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

    def switch_provider(self, provider: str, model: str = "", url: str = ""):
        """Switch LLM provider and model at runtime."""
        if provider not in ("ollama", "lmstudio", "openai"):
            raise ValueError(f"Unknown provider: {provider}")
        self.config.provider = provider
        if provider == "ollama" and model:
            self.config.ollama_model = model
        elif provider == "lmstudio" and model:
            self.config.lmstudio_model = model
        elif provider == "openai" and model:
            self.config.openai_model = model
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
    ) -> str:
        """Route LLM call to the active provider with fallback.

        Fallback rules:
        - User-selected provider is always tried first (with retries).
        - If selected provider is cloud (OpenAI) and fails:
          1. Try cheaper OpenAI models (descending: gpt-5 → gpt-5-mini → gpt-4o-mini → gpt-5-nano)
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
            )
        except Exception as primary_error:
            # Only fall back from cloud → local, never local → cloud
            if provider not in _CLOUD_PROVIDERS or not self.config.fallback_to_local:
                raise

            # Cloud fallback: try cheaper OpenAI models first, then local
            selected_model = self.config.openai_model
            try:
                idx = _OPENAI_MODEL_ORDER.index(selected_model)
            except ValueError:
                idx = -1

            # Try each cheaper OpenAI model (those after the selected one in the order)
            for model in _OPENAI_MODEL_ORDER[idx + 1:]:
                logger.warning(
                    f"OpenAI {selected_model} failed, trying {model}: {primary_error}"
                )
                try:
                    text = await self._call_provider(
                        "openai", system_prompt, user_prompt,
                        retries=0, max_tokens=max_tokens,
                        temperature=temp, json_mode=json_mode,
                        openai_model=model,
                    )
                    logger.info(f"OpenAI fallback to {model} succeeded")
                    return text
                except Exception as model_err:
                    logger.warning(f"OpenAI {model} also failed: {model_err}")
                    continue

            # All OpenAI models failed — try local providers (single attempt each)
            for fallback in _LOCAL_PROVIDERS:
                logger.warning(
                    f"All OpenAI models failed, falling back to {fallback}: {primary_error}"
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
    ) -> str:
        """Call a specific provider with retry logic. Returns raw text response."""
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
        body = {
            "model": model_override or self.config.lmstudio_model,
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

    async def _call_openai(
        self, system_prompt: str, user_prompt: str,
        temperature: float, max_tokens: int, json_mode: bool,
        model_override: str = "",
    ) -> str:
        """OpenAI: POST /v1/responses (Responses API, not Chat Completions)."""
        model = model_override or self.config.openai_model
        body = {
            "model": model,
            "input": f"{system_prompt}\n\n{user_prompt}",
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": "minimal"},
            "text": {"verbosity": "low"},
        }
        if json_mode:
            body["text"]["format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json",
        }

        response = await self.client.post(
            "https://api.openai.com/v1/responses",
            json=body, headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        # Track costs from usage
        usage = data.get("usage", {})
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            self._track_openai_cost(input_tokens, output_tokens, model)

        # Extract text: find output item with type "message", then content with type "output_text"
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        return content.get("text", "{}")

        logger.warning(f"OpenAI: no output_text found in response")
        return "{}"

    def _track_openai_cost(self, input_tokens: int, output_tokens: int, model: str = ""):
        """Update running cost totals for OpenAI usage."""
        pricing = _OPENAI_PRICING.get(model or self.config.openai_model, (0.05, 0.20))
        cost = (input_tokens / 1_000_000 * pricing[0]) + (output_tokens / 1_000_000 * pricing[1])

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
            if _is_passthrough(transcript, result.translated):
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
        if direction == "es_to_en":
            src, tgt = "Spanish", "English"
        else:
            src, tgt = "English", "Spanish"

        forceful_prompt = (
            f'The following {src} text MUST be translated to {tgt}. '
            f'Do NOT repeat the {src}. Output ONLY the {tgt} translation.\n\n'
            f'Text: "{transcript}"\n\n'
            f'Respond with JSON: {{"corrected": "<corrected {src}>", '
            f'"translated": "<{tgt} translation>", "confidence": 0.5}}'
        )

        try:
            raw_text = await self._call_llm(
                system_prompt, forceful_prompt,
                retries=0, max_tokens=512, json_mode=True,
            )
            result = self._parse_response(raw_text, transcript)
            if not _is_passthrough(transcript, result.translated):
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
        # Strip chain-of-thought blocks if present
        if "<think>" in cleaned and "</think>" in cleaned:
            cleaned = cleaned.split("</think>", 1)[-1].strip()
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
                retries=1, max_tokens=200, temperature=0.2, json_mode=False,
            )
            return text.strip() or previous_summary
        except Exception as e:
            logger.warning(f"Topic summary update failed: {e}")
            return previous_summary

    async def close(self):
        await self.client.aclose()
