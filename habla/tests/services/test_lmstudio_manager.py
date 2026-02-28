"""Tests for LMStudioManager - LM Studio process lifecycle management."""

import asyncio
import json
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
        manager._discover_loaded_models = AsyncMock()

        await manager.ensure_running()

        manager.start.assert_not_awaited()
        manager._discover_loaded_models.assert_awaited_once()

    async def test_ensure_running_discovers_models_when_already_running(self, manager):
        manager.is_running = AsyncMock(return_value=True)
        manager._loaded_models = set()
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
    async def test_discover_populates_from_lms_ps(self, manager):
        """lms ps --json is the primary source for loaded models."""
        ps_result = {"test-model", "second-model"}
        manager._query_lms_ps = AsyncMock(return_value=ps_result)

        await manager._discover_loaded_models()

        assert manager._loaded_models == ps_result

    async def test_discover_falls_back_to_api(self, manager):
        """When lms ps fails, fall back to /v1/models API."""
        manager._query_lms_ps = AsyncMock(return_value=None)
        api_result = {"test-model", "second-model"}
        manager._query_api_models = AsyncMock(return_value=api_result)

        await manager._discover_loaded_models()

        assert manager._loaded_models == api_result

    async def test_discover_handles_all_failures_gracefully(self, manager):
        """When both lms ps and /v1/models fail, loaded_models stays empty."""
        manager._query_lms_ps = AsyncMock(return_value=None)
        manager._query_api_models = AsyncMock(return_value=None)

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
# TestModelStem
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestModelStem:
    def test_strips_duplicate_suffix(self):
        assert LMStudioManager._model_stem("my-model:3") == "my-model"

    def test_strips_high_number_suffix(self):
        assert LMStudioManager._model_stem("ganymede-llama-3.3-3b-preview:10") == "ganymede-llama-3.3-3b-preview"

    def test_preserves_base_name(self):
        assert LMStudioManager._model_stem("my-model") == "my-model"

    def test_extracts_stem_from_path(self):
        assert LMStudioManager._model_stem("author/model-dir/test-model.gguf") == "test-model"

    def test_handles_path_with_suffix(self):
        assert LMStudioManager._model_stem("author/test-model:2") == "test-model"

    def test_handles_empty_string(self):
        assert LMStudioManager._model_stem("") == ""

    def test_strips_quantization_suffix(self):
        assert LMStudioManager._model_stem("TowerInstruct-Mistral-7B-v0.2-Q3_K_M.gguf") == "towerinstruct-mistral-7b-v0.2"

    def test_strips_dot_quantization(self):
        assert LMStudioManager._model_stem("Ganymede-Llama-3.3-3B-Preview.Q4_K_S.gguf") == "ganymede-llama-3.3-3b-preview"

    def test_case_insensitive(self):
        assert LMStudioManager._model_stem("TowerInstruct-Mistral-7B") == LMStudioManager._model_stem("towerinstruct-mistral-7b")

    def test_gguf_path_matches_display_name(self):
        """GGUF path stem should match .env display name after normalization."""
        gguf_path = "tensorblock/TowerInstruct-Mistral-7B-v0.2-GGUF/TowerInstruct-Mistral-7B-v0.2-Q3_K_M.gguf"
        env_name = "towerinstruct-mistral-7b-v0.2"
        assert LMStudioManager._model_stem(gguf_path) == LMStudioManager._model_stem(env_name)

    def test_iq_quantization(self):
        assert LMStudioManager._model_stem("model-name-IQ2_XXS.gguf") == "model-name"

    def test_simple_quant(self):
        assert LMStudioManager._model_stem("model-Q8_0.gguf") == "model"


# ---------------------------------------------------------------------------
# TestLogModelStatus
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLogModelStatus:
    def test_logs_loaded_models(self, manager, caplog):
        manager._loaded_models = {"model-a", "model-b"}

        with caplog.at_level("INFO", logger="habla.lmstudio"):
            manager._log_model_status()

        assert "Currently loaded models (2)" in caplog.text
        assert "model-a" in caplog.text
        assert "model-b" in caplog.text

    def test_logs_empty_set(self, manager, caplog):
        manager._loaded_models = set()

        with caplog.at_level("INFO", logger="habla.lmstudio"):
            manager._log_model_status()

        assert "Currently loaded models (0)" in caplog.text


# ---------------------------------------------------------------------------
# TestGetAvailableModels
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetAvailableModels:
    async def test_returns_models_from_lms_ls(self, manager):
        ls_output = json.dumps([
            {"key": "author/model-a", "path": "/path/to/model-a.gguf", "sizeBytes": 1000},
            {"key": "author/model-b", "path": "/path/to/model-b.gguf", "sizeBytes": 2000},
        ])
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ls_output

        with patch("server.services.lmstudio_manager.subprocess.run",
                   return_value=mock_result):
            result = await manager.get_available_models()

        assert len(result) == 2
        assert result[0]["id"] == "author/model-a"
        assert result[0]["size_bytes"] == 1000
        assert result[1]["id"] == "author/model-b"

    async def test_falls_back_to_api_on_cli_failure(self, manager):
        """If lms ls fails, falls back to /v1/models API."""
        resp = _make_httpx_response(
            status_code=200,
            json_data={"data": [{"id": "loaded/model-x"}]},
        )
        ctx, _ = _mock_async_client(get_response=resp)

        with patch("server.services.lmstudio_manager.subprocess.run",
                   side_effect=FileNotFoundError("lms not found")), \
             patch("server.services.lmstudio_manager.httpx.AsyncClient", return_value=ctx):
            result = await manager.get_available_models()

        assert len(result) == 1
        assert result[0]["id"] == "model-x"

    async def test_skips_entries_without_key(self, manager):
        ls_output = json.dumps([
            {"key": "author/valid-model", "path": "/path/to/model.gguf"},
            {"nokey": True},  # missing key/id/path
        ])
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ls_output

        with patch("server.services.lmstudio_manager.subprocess.run",
                   return_value=mock_result):
            result = await manager.get_available_models()

        assert len(result) == 1
        assert result[0]["id"] == "author/valid-model"


# ---------------------------------------------------------------------------
# TestSwitchModel
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSwitchModel:
    async def test_switch_unloads_old_and_loads_new(self, manager):
        manager._loaded_models = {"old-model"}
        manager._unload_model = AsyncMock(return_value=True)
        manager._resolve_model_path = AsyncMock(return_value="/path/to/new-model.gguf")
        manager._load_model = AsyncMock(return_value=True)

        result = await manager.switch_model(
            new_id="author/new-model", old_id="author/old-model",
        )

        assert result is True
        manager._unload_model.assert_awaited_once_with("old-model")
        manager._load_model.assert_awaited_once_with("/path/to/new-model.gguf")

    async def test_switch_skips_unload_when_same_model(self, manager):
        manager._loaded_models = {"same-model"}
        manager._unload_model = AsyncMock()

        result = await manager.switch_model(
            new_id="author/same-model", old_id="author/same-model",
        )

        assert result is True
        manager._unload_model.assert_not_awaited()

    async def test_switch_keeps_other_model(self, manager):
        """Don't unload old_id if it's the same as other_keep."""
        manager._loaded_models = {"shared-model"}
        manager._unload_model = AsyncMock()
        manager._resolve_model_path = AsyncMock(return_value="/path/to/new.gguf")
        manager._load_model = AsyncMock(return_value=True)

        result = await manager.switch_model(
            new_id="author/new-model",
            old_id="author/shared-model",
            other_keep="author/shared-model",
        )

        assert result is True
        manager._unload_model.assert_not_awaited()

    async def test_switch_returns_true_if_already_loaded(self, manager):
        manager._loaded_models = {"target-model"}
        manager._unload_model = AsyncMock()
        manager._load_model = AsyncMock()

        result = await manager.switch_model(new_id="author/target-model")

        assert result is True
        manager._load_model.assert_not_awaited()

    async def test_switch_returns_false_when_path_not_resolved(self, manager):
        manager._loaded_models = set()
        manager._resolve_model_path = AsyncMock(return_value="")

        result = await manager.switch_model(new_id="author/missing-model")

        assert result is False

    async def test_switch_unloads_old_not_in_loaded_set(self, manager):
        """If old model is not in _loaded_models, skip unload gracefully."""
        manager._loaded_models = set()
        manager._unload_model = AsyncMock()
        manager._resolve_model_path = AsyncMock(return_value="/path/to/new.gguf")
        manager._load_model = AsyncMock(return_value=True)

        result = await manager.switch_model(
            new_id="author/new-model", old_id="author/gone-model",
        )

        assert result is True
        manager._unload_model.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestResolveModelPath
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolveModelPath:
    async def test_resolves_path_from_lms_ls(self, manager):
        ls_output = json.dumps([
            {"key": "author/target-model", "path": "/models/target-model.gguf"},
        ])
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ls_output

        with patch("server.services.lmstudio_manager.subprocess.run",
                   return_value=mock_result):
            result = await manager._resolve_model_path("author/target-model")

        assert result == "/models/target-model.gguf"

    async def test_resolves_by_stem_match(self, manager):
        ls_output = json.dumps([
            {"key": "author/my-model", "path": "/models/my-model.gguf"},
        ])
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ls_output

        with patch("server.services.lmstudio_manager.subprocess.run",
                   return_value=mock_result):
            result = await manager._resolve_model_path("my-model")

        assert result == "/models/my-model.gguf"

    async def test_falls_back_to_gguf_path(self, manager):
        with patch("server.services.lmstudio_manager.subprocess.run",
                   side_effect=FileNotFoundError):
            result = await manager._resolve_model_path("path/to/model.gguf")

        assert result == "path/to/model.gguf"

    async def test_returns_empty_when_not_found(self, manager):
        ls_output = json.dumps([
            {"key": "author/other-model", "path": "/models/other.gguf"},
        ])
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ls_output

        with patch("server.services.lmstudio_manager.subprocess.run",
                   return_value=mock_result):
            result = await manager._resolve_model_path("nonexistent-model")

        assert result == ""


# ---------------------------------------------------------------------------
# TestSettingsPersistence
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSettingsPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        settings = {
            "provider": "lmstudio",
            "lmstudio_model": "author/test-model",
            "quick_model": "author/quick-model",
        }
        LMStudioManager.save_settings(tmp_path, settings)

        loaded = LMStudioManager.load_settings(tmp_path)

        assert loaded == settings

    def test_load_returns_empty_when_no_file(self, tmp_path):
        result = LMStudioManager.load_settings(tmp_path)
        assert result == {}

    def test_load_returns_empty_on_corrupt_json(self, tmp_path):
        path = tmp_path / "llm_settings.json"
        path.write_text("not valid json{{{", encoding="utf-8")

        result = LMStudioManager.load_settings(tmp_path)
        assert result == {}

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        LMStudioManager.save_settings(nested, {"provider": "ollama"})

        loaded = LMStudioManager.load_settings(nested)
        assert loaded["provider"] == "ollama"


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


# ---------------------------------------------------------------------------
# TestLoadModelDuplicateGuard
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadModelDuplicateGuard:
    async def test_load_model_skips_already_loaded(self, manager):
        """If model is already in _loaded_models, _load_model returns True without loading."""
        manager._loaded_models = {"test-model"}
        manager._load_via_cli = AsyncMock()
        manager._load_via_api = AsyncMock()
        manager._load_via_alt = AsyncMock()

        result = await manager._load_model("path/to/test-model.gguf")

        assert result is True
        manager._load_via_cli.assert_not_awaited()
        manager._load_via_api.assert_not_awaited()
        manager._load_via_alt.assert_not_awaited()

    async def test_load_model_proceeds_when_not_loaded(self, manager):
        """If model is NOT in _loaded_models, loading strategies are attempted."""
        manager._loaded_models = set()
        manager._load_via_cli = AsyncMock(return_value=True)

        result = await manager._load_model("path/to/test-model.gguf")

        assert result is True
        manager._load_via_cli.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestUnloadModel
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUnloadModel:
    async def test_unload_success_removes_from_set(self, manager):
        manager._loaded_models = {"test-model"}
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("server.services.lmstudio_manager.subprocess.run",
                   return_value=mock_result):
            result = await manager._unload_model("test-model")

        assert result is True
        assert "test-model" not in manager._loaded_models

    async def test_unload_failure_keeps_set_unchanged(self, manager):
        manager._loaded_models = {"test-model"}
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"

        with patch("server.services.lmstudio_manager.subprocess.run",
                   return_value=mock_result):
            result = await manager._unload_model("test-model")

        assert result is False
        assert "test-model" in manager._loaded_models

    async def test_unload_timeout_returns_false(self, manager):
        manager._loaded_models = {"test-model"}

        with patch("server.services.lmstudio_manager.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="lms", timeout=30)):
            result = await manager._unload_model("test-model")

        assert result is False
        assert "test-model" in manager._loaded_models


# ---------------------------------------------------------------------------
# TestUnloadAll
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUnloadAll:
    async def test_unload_all_unloads_each_model(self, manager):
        manager._loaded_models = {"model-a", "model-b"}
        manager._unload_model = AsyncMock(return_value=True)

        await manager.unload_all()

        assert manager._unload_model.await_count == 2
        calls = {c.args[0] for c in manager._unload_model.await_args_list}
        assert calls == {"model-a", "model-b"}

    async def test_unload_all_with_no_models(self, manager):
        manager._loaded_models = set()
        manager._unload_model = AsyncMock()

        await manager.unload_all()

        manager._unload_model.assert_not_awaited()
