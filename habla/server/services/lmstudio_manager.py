"""LM Studio process manager - starts, monitors, and stops LM Studio."""

import asyncio
import contextlib
import logging
import os
import platform
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
        set is accurate even when we did not start the process ourselves.
        """
        if await self.is_running():
            logger.info("LM Studio already running")
            if not self._loaded_models:
                await self._discover_loaded_models()
            return
        await self.start()

    async def start(self) -> None:
        """Spawn LM Studio, wait for initialisation, then load configured models."""
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
        await self._load_models()

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
        """Stop then start LM Studio."""
        await self.stop()
        await self.start()

    def get_loaded_models(self) -> list[str]:
        """Return a snapshot of currently loaded model names (stems, no .gguf)."""
        return list(self._loaded_models)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        """Query /v1/models and populate _loaded_models from a running LM Studio.
        Used when ensure_running() short-circuits because the process was
        already up before Habla started.
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._config.lmstudio_url}/v1/models")
            if resp.is_success:
                for m in resp.json().get("data", []):
                    model_id = m.get("id", "")
                    if model_id:
                        self._loaded_models.add(Path(model_id).stem)
                logger.info(
                    "[lmstudio] Discovered %d already-loaded model(s): %s",
                    len(self._loaded_models), sorted(self._loaded_models),
                )
                self._check_models_match_config()
        except Exception as exc:
            logger.warning("[lmstudio] Could not discover loaded models: %s", exc)

    async def _load_models(self) -> None:
        """Load all configured model paths in sequence, then check config match."""
        paths = self._config.lmstudio_model_paths
        if not paths:
            logger.warning("No LMSTUDIO_MODEL_PATHS configured - skipping model load")
            return
        for path in paths:
            await self._load_model(path)
        self._check_models_match_config()

    async def _load_model(self, path: str) -> bool:
        """Try each loading strategy in order. Returns True if any succeeded."""
        filename = Path(path).stem
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
                import json
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
        """Strategy 1: load model via `lms load`."""
        filename = Path(path).stem
        logger.info("[lmstudio/cli] Loading: %s", filename)
        try:
            lms = self._get_lms_exe()
            key = await asyncio.to_thread(self._find_model_key, path)
            port = self._port()

            result = await asyncio.to_thread(
                subprocess.run,
                [lms, "load", key, "--host", "localhost", "--port", str(port)],
                capture_output=True, text=True, timeout=60,
            )

            if result.returncode == 0:
                logger.info("[lmstudio/cli] OK: %s", filename)
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
        filename = Path(path).stem
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
        filename = Path(path).stem
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

    # --- Verification -------------------------------------------------------

    async def _verify_loaded(self, path: str) -> bool:
        """Confirm model appears in /v1/models after a 5 s settling wait."""
        filename = Path(path).stem
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

    # --- Config match check -------------------------------------------------

    def _check_models_match_config(self) -> None:
        """
        Compare loaded models against config. Logs a warning for any model
        that was configured but did not load successfully.
        """
        configured = {Path(p).stem for p in self._config.lmstudio_model_paths}
        loaded = self._loaded_models
        missing = configured - loaded
        extra = loaded - configured

        if missing:
            for m in sorted(missing):
                logger.warning("[lmstudio] Configured model NOT loaded: %s", m)
        if extra:
            for m in sorted(extra):
                logger.debug("[lmstudio] Loaded model not in config (manual load?): %s", m)

        total = len(configured)
        ok = len(configured & loaded)
        logger.info("[lmstudio] %d/%d configured models loaded", ok, total)

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
                    self._check_models_match_config()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[lmstudio/monitor] Unexpected error: %s", exc)
