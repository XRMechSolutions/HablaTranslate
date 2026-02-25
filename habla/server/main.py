"""Habla — Real-time bidirectional speech translation server."""

import logging
import signal
import mimetypes
import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from server.config import load_config
from server.db.database import init_db, close_db
from server.pipeline.orchestrator import PipelineOrchestrator
from server.routes.websocket import websocket_endpoint, set_ws_pipeline, set_recording_config, set_ws_playback_service
from server.routes._state import set_pipeline, set_lmstudio_manager, set_playback_service
from server.routes.api_vocab import vocab_router
from server.routes.api_system import system_router
from server.routes.api_sessions import session_router
from server.routes.api_idioms import idiom_router
from server.routes.api_llm import llm_router, lmstudio_router
from server.routes.api_playback import playback_router
from server.services.playback import PlaybackService
from server.services.health import run_startup_checks, run_runtime_checks, ComponentStatus
from server.services.lmstudio_manager import LMStudioManager

# Logging — structured with file rotation
import os
import logging.handlers

_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
_log_dir = Path(os.getenv("LOG_DIR", "data"))
_log_dir.mkdir(parents=True, exist_ok=True)

# Console handler (human-readable)
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
))
_console.setLevel(_log_level)

# Rotating file handler (full detail, 10MB x 5 files)
_file_handler = logging.handlers.RotatingFileHandler(
    _log_dir / "habla.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s  [%(filename)s:%(lineno)d]",
))
_file_handler.setLevel(logging.DEBUG)

# Error-only file (quick scan for problems)
_error_handler = logging.handlers.RotatingFileHandler(
    _log_dir / "habla_errors.log",
    maxBytes=5 * 1024 * 1024,  # 5MB
    backupCount=3,
    encoding="utf-8",
)
_error_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s  [%(filename)s:%(lineno)d]",
))
_error_handler.setLevel(logging.ERROR)

# Configure root logger
logging.basicConfig(level=logging.DEBUG, handlers=[_console, _file_handler, _error_handler])
logger = logging.getLogger("habla")

# Ensure correct MIME types for module scripts on Windows
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

# Global pipeline, config, and LM Studio manager
pipeline: PipelineOrchestrator | None = None
app_config = None
lmstudio_manager: LMStudioManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global pipeline, app_config, lmstudio_manager
    config = load_config()
    app_config = config

    logger.info("=" * 50)
    logger.info("  Habla — Starting up")
    logger.info(f"  ASR: WhisperX {config.asr.model_size} on {config.asr.device}")
    logger.info(f"  LLM: {config.translator.provider}/{config.translator.model}")
    logger.info(f"  Diarization: Pyannote 3.1 on CPU")
    logger.info(f"  Mode: {config.session.mode}")
    logger.info(f"  Direction: {config.session.direction}")
    logger.info("=" * 50)

    # Initialize database
    await init_db(config.db_path)
    logger.info(f"Database ready: {config.db_path}")

    # Pre-pipeline health checks (external dependencies)
    startup_health = await run_startup_checks(config)
    if startup_health.overall == ComponentStatus.DOWN:
        # Log which components failed but don't abort — allow degraded mode
        logger.warning("Some components are DOWN. Server will start in degraded mode.")

    # Start LM Studio (only when lmstudio is the active provider)
    if config.translator.provider == "lmstudio":
        lmstudio_manager = LMStudioManager(config.translator)
        set_lmstudio_manager(lmstudio_manager)
        await lmstudio_manager.ensure_running()
        lmstudio_manager.start_monitor()

    # Initialize pipeline
    pipeline = PipelineOrchestrator(config)
    set_pipeline(pipeline)
    set_ws_pipeline(pipeline)
    set_recording_config(config.recording)

    if config.recording.enabled:
        logger.info(f"Audio recording enabled: {config.recording.output_dir}")

    # Initialize playback service (works even if recording is disabled — reads existing recordings)
    playback_service = PlaybackService(config.recording.output_dir)
    set_playback_service(playback_service)
    set_ws_playback_service(playback_service)
    logger.info(f"Playback service ready: {len(playback_service.list_recordings())} recordings available")

    await pipeline.startup()

    # Load cumulative OpenAI costs from DB
    await pipeline.translator.load_all_time_costs()

    # Post-pipeline health checks (verify models loaded)
    from server.services.health import check_whisperx, check_diarization
    asr_check = check_whisperx(pipeline._whisperx_model)
    diar_check = check_diarization(pipeline._diarize_pipeline)
    if asr_check.status != ComponentStatus.OK:
        logger.warning(f"ASR: {asr_check.message}")
    if diar_check.status != ComponentStatus.OK:
        logger.warning(f"Diarization: {diar_check.message}")

    # Register signal handlers for graceful shutdown logging
    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            pass  # Not all signals available on all platforms

    async def _console_loop():
        global pipeline
        logger.info("Console commands: restart | reload | quit")
        while True:
            try:
                line = await asyncio.to_thread(sys.stdin.readline)
            except Exception:
                await asyncio.sleep(0.5)
                continue
            if not line:
                await asyncio.sleep(0.2)
                continue
            cmd = line.strip().lower()
            if not cmd:
                continue
            if cmd == "restart":
                logger.info("Console: restarting pipeline...")
                try:
                    await pipeline.shutdown()
                except Exception as e:
                    logger.warning(f"Pipeline shutdown error during restart: {e}")
                pipeline = PipelineOrchestrator(config)
                set_pipeline(pipeline)
                set_ws_pipeline(pipeline)
                await pipeline.startup()
                logger.info("Console: restart complete")
            elif cmd == "reload":
                if pipeline:
                    logger.info("Console: reloading idiom patterns...")
                    try:
                        await pipeline.reload_idiom_patterns()
                    except Exception as e:
                        logger.warning(f"Idiom reload failed: {e}")
                else:
                    logger.warning("Console: pipeline not ready")
            elif cmd == "quit":
                logger.info("Console: initiating graceful shutdown...")
                # Send SIGINT to trigger uvicorn's graceful shutdown,
                # which runs the full lifespan teardown (DB close, LM Studio stop, etc.)
                os.kill(os.getpid(), signal.SIGINT)
                return
            else:
                logger.info(f"Console: unknown command '{cmd}'")

    app.state.console_task = asyncio.create_task(_console_loop())

    yield

    # Shutdown — pipeline drains queue and saves state before closing
    logger.info("Shutting down...")
    if getattr(app.state, "console_task", None):
        app.state.console_task.cancel()
    await pipeline.shutdown()
    if lmstudio_manager is not None:
        await lmstudio_manager.stop_monitor()
        await lmstudio_manager.stop()
    await close_db()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Habla",
    description="Real-time bidirectional speech translation",
    version="0.1.0",
    lifespan=lifespan,
)

# REST routes
app.include_router(vocab_router)
app.include_router(system_router)
app.include_router(session_router)
app.include_router(idiom_router)
app.include_router(llm_router)
app.include_router(lmstudio_router)
app.include_router(playback_router)


# WebSocket endpoint
@app.websocket("/ws/translate")
async def ws_translate(websocket: WebSocket):
    await websocket_endpoint(websocket)


# Serve client files
client_dir = Path(__file__).parent.parent / "client"
if client_dir.exists():
    app.mount("/static", StaticFiles(directory=str(client_dir)), name="static")


@app.get("/")
async def index():
    index_path = client_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Habla server running. Client not found at /client/"}


@app.get("/sw.js")
async def service_worker():
    """Serve service worker from root scope (required for PWA)."""
    sw_path = client_dir / "sw.js"
    if sw_path.exists():
        return FileResponse(str(sw_path), media_type="application/javascript")
    from fastapi.responses import Response
    return Response(status_code=404)


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring."""
    if pipeline is None:
        return {"status": "starting", "components": {}}
    health = await run_runtime_checks(pipeline)
    return health.to_dict()


@app.get("/vocab")
async def vocab_page():
    vocab_path = client_dir / "vocab.html"
    if vocab_path.exists():
        return FileResponse(str(vocab_path))
    return {"message": "Vocab page not found"}


@app.get("/history")
async def history_page():
    history_path = client_dir / "history.html"
    if history_path.exists():
        return FileResponse(str(history_path))
    return {"message": "History page not found"}
