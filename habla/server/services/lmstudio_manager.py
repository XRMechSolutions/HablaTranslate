"""LM Studio process manager - starts, monitors, and stops LM Studio."""

import asyncio
import contextlib
import json
import logging
import os
import platform
import re
import subprocess
import threading
from pathlib import Path

import httpx

logger = logging.getLogger("habla.lmstudio")


class LMStudioManager:
    """Manages the LM Studio process and model loading lifecycle."""

    def __init__(self, config):
        """
        Args:
            config: TranslatorConfig instance (lmstudio_executable, lmstudio_model_paths,
                    lmstudio_url, quick_model)
        """
        self._config = config
        self._process: subprocess.Popen | None = None
        self._loaded_models: set[str] = set()
        self._monitor_task: asyncio.Task | None = None
        self._data_dir: Path | None = None  # set by main.py for restart model reload

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def is_running(self) -> bool:
        """Liveness check: /v1/models API responds.
        Intentionally does not check self._process so that externally-started
        LM Studio instances are also detected correctly.
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._config.lmstudio_url}/v1/models")
                return resp.is_success
        except Exception:
            return False

    async def ensure_running(self) -> None:
        """Start LM Studio if it is not already running.

        If already running, discover which models are loaded so the internal
        set is accurate.  Does NOT force-load models — that is driven by
        persisted user settings applied in main.py after this call.
        """
        if await self.is_running():
            logger.info("LM Studio already running")
            await self._discover_loaded_models()
            return
        await self.start()

    async def start(self) -> None:
        """Spawn LM Studio and wait for initialisation.

        Does NOT load models — that is driven by persisted user settings
        applied in main.py after ``ensure_running()`` returns.
        """
        logger.info("Starting LM Studio: %s", self._config.lmstudio_executable)

        port = self._port()
        self._process = subprocess.Popen(
            [self._config.lmstudio_executable, "--host", "0.0.0.0", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Drain stdout/stderr on background threads (mirrors C# event handlers)
        threading.Thread(
            target=self._pipe_reader,
            args=(self._process.stdout, "stdout"),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._pipe_reader,
            args=(self._process.stderr, "stderr"),
            daemon=True,
        ).start()

        logger.info("Waiting 15 s for LM Studio to initialise...")
        await asyncio.sleep(15)

        self._loaded_models.clear()

    async def stop(self) -> None:
        """Terminate LM Studio (process tree on Windows)."""
        if self._process is None:
            return
        if self._process.poll() is not None:
            self._process = None
            return

        pid = self._process.pid
        logger.info("Stopping LM Studio (PID %d)", pid)

        if platform.system() == "Windows":
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                )
                logger.info("LM Studio process tree killed (taskkill)")
            except Exception as exc:
                logger.warning("taskkill failed (%s), falling back to kill()", exc)
                self._process.kill()
        else:
            self._process.terminate()

        await asyncio.sleep(5)

        if self._process.poll() is None:
            logger.warning("LM Studio still alive after 5 s, sending kill()")
            self._process.kill()

        self._process = None
        self._loaded_models.clear()

    async def restart(self) -> None:
        """Stop then start LM Studio, reloading persisted model selections."""
        await self.stop()
        await self.start()
        # Reload user's persisted model selections
        if self._data_dir:
            saved = self.load_settings(self._data_dir)
            for key in ("lmstudio_model", "quick_model"):
                model_id = saved.get(key, "")
                if model_id:
                    await self.switch_model(new_id=model_id)

    def get_loaded_models(self) -> list[str]:
        """Return a snapshot of currently loaded model names (stems, no .gguf)."""
        return list(self._loaded_models)

    async def get_available_models(self) -> list[dict]:
        """Return all models downloaded in LM Studio (not just loaded).

        Uses ``lms ls --json``. Each entry has at least ``id`` (the load key)
        and ``path`` (full GGUF path).  Falls back to loaded-only list from
        the ``/v1/models`` API if the CLI is unavailable.
        """
        try:
            lms = self._get_lms_exe()
            result = await asyncio.to_thread(
                subprocess.run,
                [lms, "ls", "--json"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                raw = json.loads(result.stdout)
                models = []
                for entry in raw:
                    key = entry.get("modelKey") or entry.get("key") or entry.get("id") or entry.get("path", "")
                    if not key:
                        continue
                    models.append({
                        "id": key,
                        "path": entry.get("path", ""),
                        "size_bytes": entry.get("sizeBytes") or entry.get("size_bytes", 0),
                    })
                return models
        except Exception as exc:
            logger.warning("[lmstudio] lms ls failed (%s), falling back to loaded models", exc)

        # Fallback: return only loaded models from the API
        loaded = await self._query_api_models()
        return [{"id": m, "path": "", "size_bytes": 0} for m in sorted(loaded or [])]

    async def switch_model(self, new_id: str, old_id: str = "", other_keep: str = "") -> bool:
        """Unload *old_id* (if loaded and not needed) then load *new_id*.

        Args:
            new_id: Model identifier to load (key from ``lms ls``).
            old_id: Model identifier to unload first. Skipped if empty,
                    same as *new_id*, or same as *other_keep*.
            other_keep: Another model that must stay loaded (e.g. the quick
                        model when switching the main model).

        Returns True if *new_id* is loaded after the operation.
        """
        new_stem = self._model_stem(new_id)
        old_stem = self._model_stem(old_id) if old_id else ""
        keep_stem = self._model_stem(other_keep) if other_keep else ""

        # Unload old if it's different and not needed by the other role
        if old_stem and old_stem != new_stem and old_stem != keep_stem:
            if old_stem in self._loaded_models:
                await self._unload_model(old_stem)

        # Load new if not already loaded
        if new_stem in self._loaded_models:
            logger.info("[lmstudio] Model already loaded: %s", new_stem)
            return True

        # Find the GGUF path for the new model via lms ls
        path = await self._resolve_model_path(new_id)
        if not path:
            logger.error("[lmstudio] Cannot resolve path for model: %s", new_id)
            return False

        return await self._load_model(path)

    async def _resolve_model_path(self, model_id: str) -> str:
        """Find the GGUF file path for a model identifier.

        Matching is case-insensitive and tolerates missing quantization
        suffixes (e.g. ``towerinstruct-mistral-7b-v0.2`` matches the GGUF
        file ``TowerInstruct-Mistral-7B-v0.2-Q3_K_M.gguf``).
        """
        needle = self._model_stem(model_id).lower()
        try:
            lms = self._get_lms_exe()
            result = await asyncio.to_thread(
                subprocess.run,
                [lms, "ls", "--json"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                best_match = ""
                best_path = ""
                for entry in json.loads(result.stdout):
                    key = entry.get("modelKey") or entry.get("key") or entry.get("id") or ""
                    if not key:
                        continue
                    # Exact match (fast path)
                    if key == model_id:
                        return entry.get("path", "")
                    candidate = self._model_stem(key).lower()
                    # Exact stem match (case-insensitive)
                    if candidate == needle:
                        return entry.get("path", "")
                    # Prefix match: .env name may lack quantization suffix
                    # e.g. "towerinstruct-mistral-7b-v0.2" matches
                    #      "towerinstruct-mistral-7b-v0.2-q3_k_m"
                    if candidate.startswith(needle) and len(candidate) > len(best_match):
                        best_match = candidate
                        best_path = entry.get("path", "")
                if best_path:
                    return best_path
        except Exception as exc:
            logger.warning("[lmstudio] Could not resolve model path: %s", exc)
        # If model_id looks like a path already, use it directly
        if model_id.endswith(".gguf"):
            return model_id
        return ""

    # --- Settings persistence -----------------------------------------------

    @staticmethod
    def _settings_path(data_dir: Path) -> Path:
        return data_dir / "llm_settings.json"

    @staticmethod
    def load_settings(data_dir: Path) -> dict:
        """Read persisted LLM settings from disk.  Returns {} if none."""
        path = LMStudioManager._settings_path(data_dir)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[lmstudio] Could not read %s: %s", path, exc)
        return {}

    @staticmethod
    def save_settings(data_dir: Path, settings: dict) -> None:
        """Write LLM settings to disk."""
        path = LMStudioManager._settings_path(data_dir)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
            logger.info("[lmstudio] Settings saved to %s", path)
        except Exception as exc:
            logger.error("[lmstudio] Could not save settings: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_stem(name: str) -> str:
        """Normalise a model identifier to a comparable lowercase stem.

        1. Strip ``:N`` duplicate-instance suffix (LM Studio appends these).
        2. Take only the filename component (last path segment).
        3. Strip ``.gguf`` extension if present (but *not* arbitrary dots,
           since model names like ``ganymede-llama-3.3-3b-preview`` contain
           dots that ``Path.stem`` would incorrectly treat as extensions).
        4. Strip GGUF quantization suffixes (``-Q3_K_M``, ``.Q4_K_S``,
           ``-IQ2_XXS``, etc.) so ``.env`` display names match GGUF filenames.
        5. Lowercase for case-insensitive comparison.
        """
        cleaned = re.sub(r":\d+$", "", name)
        basename = cleaned.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if basename.lower().endswith(".gguf"):
            basename = basename[:-5]
        # Strip GGUF quantization suffix (Q3_K_M, Q4_K_S, IQ2_XXS, etc.)
        basename = re.sub(r"[-._][IiQq][Qq]?\d+(_[A-Za-z0-9]+)*$", "", basename)
        return basename.lower()

    def _port(self) -> int:
        """Extract port from lmstudio_url."""
        try:
            return int(self._config.lmstudio_url.rstrip("/").rsplit(":", 1)[-1])
        except (ValueError, IndexError):
            return 1234

    @staticmethod
    def _pipe_reader(pipe, label: str) -> None:
        """Read lines from a subprocess pipe and forward to logger."""
        try:
            for line in pipe:
                line = line.rstrip()
                if line:
                    logger.debug("LMStudio[%s]: %s", label, line)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    async def _discover_loaded_models(self) -> None:
        """Populate _loaded_models from a running LM Studio.

        Uses ``lms ps --json`` (authoritative — only actually loaded models),
        falling back to ``/v1/models`` if the CLI is unavailable.  The API
        fallback may include unloaded models on some LM Studio versions, so
        the CLI is strongly preferred.
        """
        live = await self._query_lms_ps()
        if live is not None:
            self._loaded_models = live
            logger.info(
                "[lmstudio] Discovered %d loaded model(s) via lms ps: %s",
                len(self._loaded_models), sorted(self._loaded_models),
            )
            return

        # Fallback: /v1/models API (may include non-loaded models)
        api = await self._query_api_models()
        if api is not None:
            self._loaded_models = api
            logger.info(
                "[lmstudio] Discovered %d model(s) via /v1/models (may include unloaded): %s",
                len(self._loaded_models), sorted(self._loaded_models),
            )
        else:
            logger.warning("[lmstudio] Could not discover loaded models via any method")

    async def _load_model(self, path: str) -> bool:
        """Try each loading strategy in order. Returns True if any succeeded."""
        filename = self._model_stem(path)
        if filename in self._loaded_models:
            logger.info("[lmstudio] Model already loaded, skipping: %s", filename)
            return True
        if await self._load_via_cli(path):
            return True
        if await self._load_via_api(path):
            return True
        if await self._load_via_alt(path):
            return True
        logger.error("[lmstudio] All strategies exhausted for model: %s", filename)
        return False

    # --- Strategy 1: lms CLI ------------------------------------------------

    def _get_lms_exe(self) -> str:
        """Locate lms.exe: ~/.cache/lm-studio/bin -> ~/.lmstudio/bin -> PATH."""
        home = Path.home()
        candidates = [
            home / ".cache" / "lm-studio" / "bin" / "lms.exe",
            home / ".lmstudio" / "bin" / "lms.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return "lms"  # hope it's on PATH

    def _find_model_key(self, path: str) -> str:
        """
        Run `lms ls --json`, find the key for the given model file.
        Falls back to deriving the key from the path components.
        """
        lms = self._get_lms_exe()
        filename = Path(path).name
        try:
            result = subprocess.run(
                [lms, "ls", "--json"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                models = json.loads(result.stdout)
                for model in models:
                    model_json = json.dumps(model)
                    if filename.lower() in model_json.lower():
                        key = model.get("key") or model.get("id") or model.get("path", "")
                        if key:
                            logger.debug("[lmstudio/cli] Found model key: %s", key)
                            return key
        except Exception as exc:
            logger.debug("[lmstudio/cli] lms ls failed: %s", exc)

        # Derive key from path: author/model-dir/filename.gguf
        parts = Path(path).parts
        if len(parts) >= 3:
            key = "/".join(parts[-3:])
        else:
            key = filename
        logger.debug("[lmstudio/cli] Derived model key: %s", key)
        return key

    async def _load_via_cli(self, path: str) -> bool:
        """Strategy 1: load model via ``lms load``.

        Tries the display-name form first (``lms load <stem>``), which lets
        the CLI resolve the GGUF path itself.  Falls back to the full key
        with ``--host``/``--port`` if the simple form fails.
        """
        filename = self._model_stem(path)
        logger.info("[lmstudio/cli] Loading: %s", filename)
        try:
            lms = self._get_lms_exe()

            # Try 1: display name only — lets lms resolve the GGUF itself
            result = await asyncio.to_thread(
                subprocess.run,
                [lms, "load", filename],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                logger.info("[lmstudio/cli] OK (display name): %s", filename)
                self._loaded_models.add(filename)
                await asyncio.sleep(10)
                await self._verify_loaded(path)
                return True

            logger.debug(
                "[lmstudio/cli] display-name load failed (exit %d), trying full key",
                result.returncode,
            )

            # Try 2: full key with host/port
            key = await asyncio.to_thread(self._find_model_key, path)
            port = self._port()

            result = await asyncio.to_thread(
                subprocess.run,
                [lms, "load", key, "--host", "localhost", "--port", str(port)],
                capture_output=True, text=True, timeout=60,
            )

            if result.returncode == 0:
                logger.info("[lmstudio/cli] OK (full key): %s", filename)
                self._loaded_models.add(filename)
                await asyncio.sleep(10)
                await self._verify_loaded(path)
                return True

            logger.warning(
                "[lmstudio/cli] FAILED (exit %d): %s - stderr: %s",
                result.returncode, filename, result.stderr.strip()
            )
            return False

        except subprocess.TimeoutExpired:
            logger.warning("[lmstudio/cli] FAILED (timeout): %s", filename)
            return False
        except Exception as exc:
            logger.warning("[lmstudio/cli] FAILED (%s): %s", exc, filename)
            return False

    # --- Strategy 2: POST /v1/completions -----------------------------------

    async def _load_via_api(self, path: str) -> bool:
        """Strategy 2: trigger model load by sending a minimal completions request."""
        filename = self._model_stem(path)
        logger.info("[lmstudio/api] Loading: %s", filename)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self._config.lmstudio_url}/v1/completions",
                    json={
                        "model": filename.lower(),
                        "prompt": "\n\n",
                        "max_tokens": 1,
                        "stream": False,
                    },
                )
            if resp.is_success:
                logger.info("[lmstudio/api] OK: %s", filename)
                self._loaded_models.add(filename)
                await asyncio.sleep(10)
                await self._verify_loaded(path)
                return True

            logger.warning(
                "[lmstudio/api] FAILED (HTTP %d): %s", resp.status_code, filename
            )
            return False

        except Exception as exc:
            logger.warning("[lmstudio/api] FAILED (%s): %s", exc, filename)
            return False

    # --- Strategy 3: POST /v1/models/load -----------------------------------

    async def _load_via_alt(self, path: str) -> bool:
        """Strategy 3: undocumented /v1/models/load endpoint."""
        filename = self._model_stem(path)
        logger.info("[lmstudio/alt] Loading: %s", filename)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self._config.lmstudio_url}/v1/models/load",
                    json={"model_path": path},
                )
            if resp.is_success:
                logger.info("[lmstudio/alt] OK: %s", filename)
                self._loaded_models.add(filename)
                await asyncio.sleep(15)
                return True

            logger.warning(
                "[lmstudio/alt] FAILED (HTTP %d) - all strategies exhausted for: %s",
                resp.status_code, filename,
            )
            return False

        except Exception as exc:
            logger.warning(
                "[lmstudio/alt] FAILED (%s) - all strategies exhausted for: %s",
                exc, filename,
            )
            return False

    # --- Unloading -----------------------------------------------------------

    async def _unload_model(self, identifier: str) -> bool:
        """Unload a model from LM Studio via `lms unload`.

        Args:
            identifier: Model stem name (e.g., 'test-model') to unload.

        Returns:
            True if unload succeeded, False otherwise.
        """
        logger.info("[lmstudio/unload] Unloading: %s", identifier)
        try:
            lms = self._get_lms_exe()
            result = await asyncio.to_thread(
                subprocess.run,
                [lms, "unload", identifier],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                self._loaded_models.discard(identifier)
                logger.info("[lmstudio/unload] OK: %s", identifier)
                return True
            logger.warning(
                "[lmstudio/unload] FAILED (exit %d): %s - stderr: %s",
                result.returncode, identifier, result.stderr.strip(),
            )
            return False
        except subprocess.TimeoutExpired:
            logger.warning("[lmstudio/unload] FAILED (timeout): %s", identifier)
            return False
        except Exception as exc:
            logger.warning("[lmstudio/unload] FAILED (%s): %s", exc, identifier)
            return False

    async def unload_all(self) -> None:
        """Unload all currently loaded models. Used before shutdown or full reload."""
        models = list(self._loaded_models)
        if not models:
            logger.info("[lmstudio/unload] No models to unload")
            return
        logger.info("[lmstudio/unload] Unloading all %d models: %s", len(models), sorted(models))
        for model in models:
            await self._unload_model(model)

    # --- Verification -------------------------------------------------------

    async def _verify_loaded(self, path: str) -> bool:
        """Confirm model appears in /v1/models after a 5 s settling wait."""
        filename = self._model_stem(path)
        await asyncio.sleep(5)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._config.lmstudio_url}/v1/models")
            if resp.is_success and filename.lower() in resp.text.lower():
                logger.info("[lmstudio/verify] %s confirmed in /v1/models", filename)
                return True
            logger.warning("[lmstudio/verify] %s NOT FOUND in /v1/models", filename)
            return False
        except Exception as exc:
            logger.warning("[lmstudio/verify] check failed (%s)", exc)
            return False

    # --- Live model verification ---------------------------------------------

    async def _refresh_loaded_models(self) -> None:
        """Re-query LM Studio to update _loaded_models from live state.
        Tries `lms ps --json` first (authoritative), falls back to /v1/models API.
        Detects silent model eviction that the in-memory set would miss.
        """
        live_models = await self._query_lms_ps()
        if live_models is None:
            live_models = await self._query_api_models()
        if live_models is None:
            return  # both methods failed, keep stale set

        evicted = self._loaded_models - live_models
        new = live_models - self._loaded_models
        if evicted:
            logger.warning(
                "[lmstudio/refresh] Models no longer loaded (evicted?): %s",
                sorted(evicted),
            )
        if new:
            logger.info(
                "[lmstudio/refresh] New models detected: %s", sorted(new),
            )
        self._loaded_models = live_models

    async def _query_lms_ps(self) -> set[str] | None:
        """Run `lms ps --json` and return set of loaded model stems, or None on failure."""
        try:
            lms = self._get_lms_exe()
            result = await asyncio.to_thread(
                subprocess.run,
                [lms, "ps", "--json"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                models = set()
                for entry in data:
                    # lms ps returns objects with "identifier", "path", or "id" fields
                    name = entry.get("identifier") or entry.get("id") or entry.get("path", "")
                    if name:
                        models.add(LMStudioManager._model_stem(name))
                logger.debug("[lmstudio/ps] Live models: %s", sorted(models))
                return models
        except FileNotFoundError:
            logger.debug("[lmstudio/ps] lms.exe not found, falling back to API")
        except Exception as exc:
            logger.debug("[lmstudio/ps] Failed: %s", exc)
        return None

    async def _query_api_models(self) -> set[str] | None:
        """Query /v1/models API and return set of loaded model stems, or None on failure."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._config.lmstudio_url}/v1/models")
            if resp.is_success:
                models = set()
                for m in resp.json().get("data", []):
                    model_id = m.get("id", "")
                    if model_id:
                        models.add(LMStudioManager._model_stem(model_id))
                return models
        except Exception as exc:
            logger.debug("[lmstudio/api-models] Failed: %s", exc)
        return None

    # --- Config match check -------------------------------------------------

    def _log_model_status(self) -> None:
        """Log currently loaded models (informational only, no enforcement)."""
        logger.info(
            "[lmstudio] Currently loaded models (%d): %s",
            len(self._loaded_models), sorted(self._loaded_models),
        )

    # ------------------------------------------------------------------
    # Background monitor
    # ------------------------------------------------------------------

    def start_monitor(self) -> None:
        """Start the background health-check loop as an asyncio task."""
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="lmstudio-monitor"
        )

    async def stop_monitor(self) -> None:
        """Cancel the background monitor and wait for it to finish."""
        if self._monitor_task is None:
            return
        self._monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._monitor_task
        self._monitor_task = None

    async def _monitor_loop(self) -> None:
        """
        20 s startup delay, then poll every 60 s.
        Restarts LM Studio if it stops responding.
        Logs a warning if loaded models diverge from config.
        """
        logger.info("[lmstudio/monitor] Starting (20 s initial delay)")
        await asyncio.sleep(20)
        await self.ensure_running()

        while True:
            await asyncio.sleep(60)
            try:
                if not await self.is_running():
                    logger.warning("[lmstudio/monitor] Not responding - restarting")
                    await self.restart()
                else:
                    logger.debug("[lmstudio/monitor] Health check OK")
                    await self._refresh_loaded_models()
                    self._log_model_status()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[lmstudio/monitor] Unexpected error: %s", exc)
