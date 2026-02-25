"""Tests for LMStudioManager - LM Studio process lifecycle management."""

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import httpx
import pytest

from server.config import TranslatorConfig
from server.services.lmstudio_manager import LMStudioManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lms_config():
    return TranslatorConfig(
        provider="lmstudio",
        lmstudio_url="http://localhost:1234",
        lmstudio_executable="C:/Program Files/LM Studio/LM Studio.exe",
        lmstudio_model_paths=["path/to/author/model-dir/test-model.gguf"],
    )


@pytest.fixture
def manager(lms_config):
    return LMStudioManager(lms_config)


def _make_httpx_response(*, status_code=200, json_data=None, text=""):
    """Build a mock httpx.Response with the fields LMStudioManager reads."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.json.return_value = json_data or {}
    resp.text = text or (str(json_data) if json_data else "")
    return resp


def _mock_async_client(get_response=None, post_response=None, get_side_effect=None, post_side_effect=None):
    """Return a mock that works as `async with httpx.AsyncClient() as client:`."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    if get_side_effect:
        mock_client.get.side_effect = get_side_effect
    elif get_response is not None:
        mock_client.get.return_value = get_response

    if post_side_effect:
        mock_client.post.side_effect = post_side_effect
    elif post_response is not None:
        mock_client.post.return_value = post_response

    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=mock_client)
    context_manager.__aexit__ = AsyncMock(return_value=False)
    return context_manager, mock_client


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestInit:
    def test_init_stores_config(self, lms_config):
        mgr = LMStudioManager(lms_config)
        assert mgr._config is lms_config

    def test_init_starts_with_no_process(self, manager):
        assert manager._process is None

    def test_init_starts_with_empty_loaded_models(self, manager):
        assert manager._loaded_models == set()
        assert len(manager._loaded_models) == 0


# ---------------------------------------------------------------------------
# TestIsRunning
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIsRunning:
    async def test_is_running_returns_true_when_api_responds(self, manager):
        resp = _make_httpx_response(status_code=200, json_data={"data": []})
        ctx, _ = _mock_async_client(get_response=resp)

        with patch("server.services.lmstudio_manager.httpx.AsyncClient", return_value=ctx):
            result = await manager.is_running()

        assert result is True

    async def test_is_running_returns_false_on_connection_error(self, manager):
        ctx, mock_client = _mock_async_client()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        with patch("server.services.lmstudio_manager.httpx.AsyncClient", return_value=ctx):
            result = await manager.is_running()

        assert result is False

    async def test_is_running_returns_false_on_timeout(self, manager):
        ctx, mock_client = _mock_async_client()
        mock_client.get.side_effect = httpx.TimeoutException("Timed out")

        with patch("server.services.lmstudio_manager.httpx.AsyncClient", return_value=ctx):
            result = await manager.is_running()

        assert result is False


# ---------------------------------------------------------------------------
# TestEnsureRunning
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnsureRunning:
    async def test_ensure_running_starts_if_not_running(self, manager):
        manager.is_running = AsyncMock(return_value=False)
        manager.start = AsyncMock()

        await manager.ensure_running()

        manager.start.assert_awaited_once()

    async def test_ensure_running_skips_start_if_already_running(self, manager):
        manager.is_running = AsyncMock(return_value=True)
        manager._loaded_models = {"some-model"}
        manager.start = AsyncMock()

        await manager.ensure_running()

        manager.start.assert_not_awaited()

    async def test_ensure_running_discovers_models_when_already_running(self, manager):
        manager.is_running = AsyncMock(return_value=True)
        manager._loaded_models = set()  # empty triggers discovery
        manager._discover_loaded_models = AsyncMock()
        manager.start = AsyncMock()

        await manager.ensure_running()

        manager._discover_loaded_models.assert_awaited_once()
        manager.start.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestPort
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPort:
    def test_port_extracts_from_url(self, manager):
        assert manager._port() == 1234

    def test_port_defaults_to_1234_on_invalid_url(self, lms_config):
        lms_config.lmstudio_url = "not-a-url"
        mgr = LMStudioManager(lms_config)
        assert mgr._port() == 1234


# ---------------------------------------------------------------------------
# TestGetLoadedModels
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetLoadedModels:
    def test_get_loaded_models_returns_list_copy(self, manager):
        manager._loaded_models = {"model-a", "model-b"}

        result = manager.get_loaded_models()

        assert isinstance(result, list)
        assert set(result) == {"model-a", "model-b"}
        # Mutating the returned list must not affect internal state
        result.append("model-c")
        assert "model-c" not in manager._loaded_models


# ---------------------------------------------------------------------------
# TestDiscoverLoadedModels
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDiscoverLoadedModels:
    async def test_discover_populates_loaded_models_from_api(self, manager):
        resp = _make_httpx_response(
            status_code=200,
            json_data={"data": [{"id": "author/test-model"}, {"id": "other/second-model"}]},
        )
        ctx, _ = _mock_async_client(get_response=resp)

        with patch("server.services.lmstudio_manager.httpx.AsyncClient", return_value=ctx):
            await manager._discover_loaded_models()

        # Path("author/test-model").stem == "test-model"
        assert "test-model" in manager._loaded_models
        assert "second-model" in manager._loaded_models

    async def test_discover_handles_api_failure_gracefully(self, manager):
        ctx, mock_client = _mock_async_client()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        with patch("server.services.lmstudio_manager.httpx.AsyncClient", return_value=ctx):
            await manager._discover_loaded_models()

        assert len(manager._loaded_models) == 0


# ---------------------------------------------------------------------------
# TestLoadViaApi
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadViaApi:
    async def test_load_via_api_success_adds_to_loaded(self, manager):
        post_resp = _make_httpx_response(status_code=200)
        ctx, _ = _mock_async_client(post_response=post_resp)

        # Also mock _verify_loaded and asyncio.sleep
        manager._verify_loaded = AsyncMock(return_value=True)

        with patch("server.services.lmstudio_manager.httpx.AsyncClient", return_value=ctx), \
             patch("server.services.lmstudio_manager.asyncio.sleep", new_callable=AsyncMock):
            result = await manager._load_via_api("path/to/author/model-dir/test-model.gguf")

        assert result is True
        assert "test-model" in manager._loaded_models

    async def test_load_via_api_failure_returns_false(self, manager):
        post_resp = _make_httpx_response(status_code=500)
        ctx, _ = _mock_async_client(post_response=post_resp)

        with patch("server.services.lmstudio_manager.httpx.AsyncClient", return_value=ctx):
            result = await manager._load_via_api("path/to/author/model-dir/test-model.gguf")

        assert result is False
        assert "test-model" not in manager._loaded_models


# ---------------------------------------------------------------------------
# TestLoadViaAlt
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadViaAlt:
    async def test_load_via_alt_success_adds_to_loaded(self, manager):
        post_resp = _make_httpx_response(status_code=200)
        ctx, _ = _mock_async_client(post_response=post_resp)

        with patch("server.services.lmstudio_manager.httpx.AsyncClient", return_value=ctx), \
             patch("server.services.lmstudio_manager.asyncio.sleep", new_callable=AsyncMock):
            result = await manager._load_via_alt("path/to/author/model-dir/test-model.gguf")

        assert result is True
        assert "test-model" in manager._loaded_models

    async def test_load_via_alt_failure_returns_false(self, manager):
        post_resp = _make_httpx_response(status_code=500)
        ctx, _ = _mock_async_client(post_response=post_resp)

        with patch("server.services.lmstudio_manager.httpx.AsyncClient", return_value=ctx):
            result = await manager._load_via_alt("path/to/author/model-dir/test-model.gguf")

        assert result is False
        assert "test-model" not in manager._loaded_models


# ---------------------------------------------------------------------------
# TestVerifyLoaded
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVerifyLoaded:
    async def test_verify_loaded_finds_model_in_response(self, manager):
        resp = _make_httpx_response(
            status_code=200,
            text='{"data": [{"id": "author/test-model"}]}',
        )
        ctx, _ = _mock_async_client(get_response=resp)

        with patch("server.services.lmstudio_manager.httpx.AsyncClient", return_value=ctx), \
             patch("server.services.lmstudio_manager.asyncio.sleep", new_callable=AsyncMock):
            result = await manager._verify_loaded("path/to/author/model-dir/test-model.gguf")

        assert result is True

    async def test_verify_loaded_returns_false_when_missing(self, manager):
        resp = _make_httpx_response(
            status_code=200,
            text='{"data": [{"id": "author/other-model"}]}',
        )
        ctx, _ = _mock_async_client(get_response=resp)

        with patch("server.services.lmstudio_manager.httpx.AsyncClient", return_value=ctx), \
             patch("server.services.lmstudio_manager.asyncio.sleep", new_callable=AsyncMock):
            result = await manager._verify_loaded("path/to/author/model-dir/test-model.gguf")

        assert result is False


# ---------------------------------------------------------------------------
# TestCheckModelsMatchConfig
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckModelsMatchConfig:
    def test_all_configured_models_loaded_logs_ok(self, manager, caplog):
        manager._loaded_models = {"test-model"}

        with caplog.at_level("INFO", logger="habla.lmstudio"):
            manager._check_models_match_config()

        assert "1/1 configured models loaded" in caplog.text
        assert "NOT loaded" not in caplog.text

    def test_missing_model_logs_warning(self, manager, caplog):
        manager._loaded_models = set()  # nothing loaded

        with caplog.at_level("WARNING", logger="habla.lmstudio"):
            manager._check_models_match_config()

        assert "NOT loaded" in caplog.text
        assert "test-model" in caplog.text


# ---------------------------------------------------------------------------
# TestStop
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStop:
    async def test_stop_no_process_does_nothing(self, manager):
        manager._process = None

        await manager.stop()

        assert manager._process is None

    async def test_stop_already_exited_clears_process(self, manager):
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 0  # already exited
        manager._process = mock_proc

        await manager.stop()

        assert manager._process is None


# ---------------------------------------------------------------------------
# TestMonitor
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMonitor:
    async def test_start_monitor_creates_task(self, manager):
        # create_task needs a running event loop (provided by async test)
        async def fake_loop():
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                raise

        with patch.object(manager, "_monitor_loop", side_effect=fake_loop):
            manager.start_monitor()

        assert manager._monitor_task is not None
        assert isinstance(manager._monitor_task, asyncio.Task)

        # Cleanup: cancel so the event loop doesn't complain
        manager._monitor_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await manager._monitor_task

    async def test_stop_monitor_cancels_task(self, manager):
        async def fake_loop():
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                raise

        with patch.object(manager, "_monitor_loop", side_effect=fake_loop):
            manager.start_monitor()

        assert manager._monitor_task is not None

        await manager.stop_monitor()

        assert manager._monitor_task is None


# ---------------------------------------------------------------------------
# TestRefreshLoadedModels
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestRefreshLoadedModels:
    """Tests for live model verification via lms ps / API fallback."""

    async def test_refresh_detects_evicted_model(self, manager):
        """If a model disappears from LM Studio, _loaded_models is updated."""
        manager._loaded_models = {"model-a", "model-b"}

        # lms ps returns only model-a
        with patch.object(manager, "_query_lms_ps", new_callable=AsyncMock,
                          return_value={"model-a"}):
            await manager._refresh_loaded_models()

        assert manager._loaded_models == {"model-a"}

    async def test_refresh_detects_new_model(self, manager):
        """If a model is manually loaded, it appears in _loaded_models."""
        manager._loaded_models = {"model-a"}

        with patch.object(manager, "_query_lms_ps", new_callable=AsyncMock,
                          return_value={"model-a", "model-new"}):
            await manager._refresh_loaded_models()

        assert "model-new" in manager._loaded_models

    async def test_refresh_falls_back_to_api(self, manager):
        """If lms ps fails, falls back to /v1/models API."""
        manager._loaded_models = {"old-model"}

        with (
            patch.object(manager, "_query_lms_ps", new_callable=AsyncMock,
                         return_value=None),
            patch.object(manager, "_query_api_models", new_callable=AsyncMock,
                         return_value={"api-model"}),
        ):
            await manager._refresh_loaded_models()

        assert manager._loaded_models == {"api-model"}

    async def test_refresh_keeps_stale_set_on_total_failure(self, manager):
        """If both lms ps and API fail, keep existing set."""
        manager._loaded_models = {"stale-model"}

        with (
            patch.object(manager, "_query_lms_ps", new_callable=AsyncMock,
                         return_value=None),
            patch.object(manager, "_query_api_models", new_callable=AsyncMock,
                         return_value=None),
        ):
            await manager._refresh_loaded_models()

        assert manager._loaded_models == {"stale-model"}

    async def test_query_lms_ps_parses_json(self, manager):
        """_query_lms_ps correctly parses lms ps --json output."""
        ps_output = '[{"identifier": "author/model-dir/my-model.gguf"}]'
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ps_output

        with patch("server.services.lmstudio_manager.subprocess.run",
                   return_value=mock_result):
            result = await manager._query_lms_ps()

        assert result == {"my-model"}

    async def test_query_lms_ps_returns_none_on_missing_exe(self, manager):
        """If lms.exe is not found, returns None gracefully."""
        with patch("server.services.lmstudio_manager.subprocess.run",
                   side_effect=FileNotFoundError("lms not found")):
            result = await manager._query_lms_ps()

        assert result is None
