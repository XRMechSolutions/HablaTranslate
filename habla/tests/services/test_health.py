"""Tests for server.services.health — startup and runtime health checks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from server.services.health import (
    ComponentStatus,
    HealthCheck,
    SystemHealth,
    _check_active_llm,
    check_database,
    check_diarization,
    check_ffmpeg,
    check_hf_token,
    check_lmstudio,
    check_ollama,
    check_openai_key,
    check_whisperx,
    run_llm_health_monitor,
    run_runtime_checks,
    run_startup_checks,
    _runtime_cache,
)


# ---------------------------------------------------------------------------
# TestComponentStatus
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestComponentStatus:
    def test_ok_value(self):
        assert ComponentStatus.OK.value == "ok"

    def test_degraded_value(self):
        assert ComponentStatus.DEGRADED.value == "degraded"

    def test_down_value(self):
        assert ComponentStatus.DOWN.value == "down"

    def test_is_str_subclass(self):
        assert isinstance(ComponentStatus.OK, str)


# ---------------------------------------------------------------------------
# TestHealthCheck
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestHealthCheck:
    def test_default_values(self):
        hc = HealthCheck(component="test", status=ComponentStatus.OK)
        assert hc.component == "test"
        assert hc.status == ComponentStatus.OK
        assert hc.message == ""
        assert hc.latency_ms == 0.0

    def test_custom_values(self):
        hc = HealthCheck(
            component="ollama",
            status=ComponentStatus.DEGRADED,
            message="Model missing",
            latency_ms=42.5,
        )
        assert hc.component == "ollama"
        assert hc.status == ComponentStatus.DEGRADED
        assert hc.message == "Model missing"
        assert hc.latency_ms == 42.5


# ---------------------------------------------------------------------------
# TestSystemHealth
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestSystemHealth:
    def test_add_ok_check_keeps_overall_ok(self):
        health = SystemHealth()
        health.add(HealthCheck("a", ComponentStatus.OK, "fine"))
        assert health.overall == ComponentStatus.OK
        assert len(health.checks) == 1

    def test_add_degraded_check_sets_overall_degraded(self):
        health = SystemHealth()
        health.add(HealthCheck("a", ComponentStatus.OK))
        health.add(HealthCheck("b", ComponentStatus.DEGRADED, "warn"))
        assert health.overall == ComponentStatus.DEGRADED

    def test_add_down_check_sets_overall_down(self):
        health = SystemHealth()
        health.add(HealthCheck("a", ComponentStatus.DOWN, "dead"))
        assert health.overall == ComponentStatus.DOWN

    def test_down_overrides_degraded(self):
        health = SystemHealth()
        health.add(HealthCheck("a", ComponentStatus.DEGRADED))
        assert health.overall == ComponentStatus.DEGRADED
        health.add(HealthCheck("b", ComponentStatus.DOWN))
        assert health.overall == ComponentStatus.DOWN

    def test_degraded_does_not_override_down(self):
        health = SystemHealth()
        health.add(HealthCheck("a", ComponentStatus.DOWN))
        health.add(HealthCheck("b", ComponentStatus.DEGRADED))
        assert health.overall == ComponentStatus.DOWN

    def test_to_dict_returns_correct_structure(self):
        health = SystemHealth()
        health.add(HealthCheck("ollama", ComponentStatus.OK, "ready", 12.34))
        health.add(HealthCheck("ffmpeg", ComponentStatus.DEGRADED, "slow", 99.999))

        result = health.to_dict()
        assert result["status"] == "degraded"
        assert "ollama" in result["components"]
        assert result["components"]["ollama"]["status"] == "ok"
        assert result["components"]["ollama"]["message"] == "ready"
        assert result["components"]["ollama"]["latency_ms"] == 12.3
        assert result["components"]["ffmpeg"]["latency_ms"] == 100.0


# ---------------------------------------------------------------------------
# TestCheckOllama
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCheckOllama:
    async def test_check_ollama_model_found_returns_ok(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "qwen3:4b"}, {"name": "llama3:8b"}]
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("server.services.health.httpx.AsyncClient", return_value=mock_client):
            result = await check_ollama("http://localhost:11434", "qwen3:4b")

        assert result.status == ComponentStatus.OK
        assert result.component == "ollama"
        assert "qwen3:4b" in result.message
        assert result.latency_ms >= 0

    async def test_check_ollama_model_missing_returns_degraded(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "llama3:8b"}]
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("server.services.health.httpx.AsyncClient", return_value=mock_client):
            result = await check_ollama("http://localhost:11434", "qwen3:4b")

        assert result.status == ComponentStatus.DEGRADED
        assert "not found" in result.message
        assert "ollama pull" in result.message

    async def test_check_ollama_connection_refused_returns_down(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("server.services.health.httpx.AsyncClient", return_value=mock_client):
            result = await check_ollama("http://localhost:11434", "qwen3:4b")

        assert result.status == ComponentStatus.DOWN
        assert "Cannot connect" in result.message

    async def test_check_ollama_unexpected_error_returns_down(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=RuntimeError("something broke"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("server.services.health.httpx.AsyncClient", return_value=mock_client):
            result = await check_ollama("http://localhost:11434", "qwen3:4b")

        assert result.status == ComponentStatus.DOWN
        assert "something broke" in result.message


# ---------------------------------------------------------------------------
# TestCheckFfmpeg
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCheckFfmpeg:
    async def test_check_ffmpeg_installed_returns_ok(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(b"ffmpeg version 6.0 Copyright (c) 2000-2023\n", b"")
        )

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            result = await check_ffmpeg()

        assert result.status == ComponentStatus.OK
        assert result.component == "ffmpeg"
        assert "ffmpeg version" in result.message
        assert result.latency_ms >= 0

    async def test_check_ffmpeg_not_found_returns_down(self):
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, side_effect=FileNotFoundError):
            result = await check_ffmpeg()

        assert result.status == ComponentStatus.DOWN
        assert "not found" in result.message

    async def test_check_ffmpeg_nonzero_exit_returns_down(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            result = await check_ffmpeg()

        assert result.status == ComponentStatus.DOWN
        assert "non-zero" in result.message


# ---------------------------------------------------------------------------
# TestCheckDatabase
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCheckDatabase:
    async def test_check_database_healthy_returns_ok(self):
        mock_cursor = MagicMock()
        mock_cursor.fetchone = AsyncMock(return_value=(42,))
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_cursor)

        # The function imports get_db inline, so we patch at the source module
        with patch("server.db.database.get_db", new_callable=AsyncMock, return_value=mock_db) as mock_get:
            result = await check_database("data/habla.db")

        assert result.status == ComponentStatus.OK
        assert result.component == "database"
        assert "42 vocab items" in result.message
        assert result.latency_ms >= 0

    async def test_check_database_not_initialized_returns_down(self):
        with patch("server.db.database.get_db", new_callable=AsyncMock, side_effect=RuntimeError("DB not init")):
            result = await check_database("data/habla.db")

        assert result.status == ComponentStatus.DOWN
        assert "Not initialized" in result.message

    async def test_check_database_query_error_returns_down(self):
        with patch("server.db.database.get_db", new_callable=AsyncMock, side_effect=Exception("disk full")):
            result = await check_database("data/habla.db")

        assert result.status == ComponentStatus.DOWN
        assert "disk full" in result.message


# ---------------------------------------------------------------------------
# TestCheckWhisperx
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCheckWhisperx:
    def test_check_whisperx_loaded_returns_ok(self):
        model = MagicMock()
        result = check_whisperx(model)
        assert result.status == ComponentStatus.OK
        assert result.component == "whisperx"
        assert "loaded" in result.message.lower()

    def test_check_whisperx_none_returns_degraded(self):
        result = check_whisperx(None)
        assert result.status == ComponentStatus.DEGRADED
        assert "text-only" in result.message.lower()


# ---------------------------------------------------------------------------
# TestCheckDiarization
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCheckDiarization:
    def test_check_diarization_loaded_returns_ok(self):
        pipeline = MagicMock()
        result = check_diarization(pipeline)
        assert result.status == ComponentStatus.OK
        assert result.component == "diarization"
        assert "loaded" in result.message.lower()

    def test_check_diarization_none_returns_degraded(self):
        result = check_diarization(None)
        assert result.status == ComponentStatus.DEGRADED
        assert "disabled" in result.message.lower()


# ---------------------------------------------------------------------------
# TestCheckLmstudio
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCheckLmstudio:
    async def test_check_lmstudio_models_loaded_returns_ok(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"id": "model-a"}, {"id": "model-b"}]
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("server.services.health.httpx.AsyncClient", return_value=mock_client):
            result = await check_lmstudio("http://localhost:1234")

        assert result.status == ComponentStatus.OK
        assert result.component == "lmstudio"
        assert "2 model(s)" in result.message

    async def test_check_lmstudio_no_models_returns_degraded(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("server.services.health.httpx.AsyncClient", return_value=mock_client):
            result = await check_lmstudio("http://localhost:1234")

        assert result.status == ComponentStatus.DEGRADED
        assert "no models" in result.message.lower()

    async def test_check_lmstudio_connection_refused_returns_down(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("server.services.health.httpx.AsyncClient", return_value=mock_client):
            result = await check_lmstudio("http://localhost:1234")

        assert result.status == ComponentStatus.DOWN
        assert "Cannot connect" in result.message


# ---------------------------------------------------------------------------
# TestCheckOpenaiKey
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCheckOpenaiKey:
    def test_check_openai_key_set_returns_ok(self):
        result = check_openai_key("sk-proj-abcdef1234567890wxyz")
        assert result.status == ComponentStatus.OK
        assert result.component == "openai"
        assert "Configured" in result.message

    def test_check_openai_key_empty_returns_degraded(self):
        result = check_openai_key("")
        assert result.status == ComponentStatus.DEGRADED
        assert "not set" in result.message.lower()

    def test_key_not_leaked_in_message(self):
        key = "sk-proj-abcdef1234567890wxyz"
        result = check_openai_key(key)
        # Key material must never appear in health output
        assert key not in result.message
        assert key[:8] not in result.message


# ---------------------------------------------------------------------------
# TestCheckHfToken
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCheckHfToken:
    async def test_check_hf_token_set_returns_ok(self):
        result = await check_hf_token("hf_abcdef123456")
        assert result.status == ComponentStatus.OK
        assert result.component == "hf_token"

    async def test_check_hf_token_empty_returns_degraded(self):
        result = await check_hf_token("")
        assert result.status == ComponentStatus.DEGRADED
        assert "not set" in result.message.lower()


# ---------------------------------------------------------------------------
# TestCheckActiveLlm
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCheckActiveLlm:
    def _make_config(self, provider="ollama"):
        cfg = MagicMock()
        cfg.provider = provider
        cfg.ollama_url = "http://localhost:11434"
        cfg.ollama_model = "qwen3:4b"
        cfg.lmstudio_url = "http://localhost:1234"
        cfg.openai_api_key = "sk-test-key-1234567890ab"
        return cfg

    async def test_ollama_provider_calls_check_ollama(self):
        cfg = self._make_config("ollama")
        with patch("server.services.health.check_ollama", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = HealthCheck("ollama", ComponentStatus.OK, "fine")
            result = await _check_active_llm(cfg)

        mock_check.assert_awaited_once_with("http://localhost:11434", "qwen3:4b")
        assert result.component == "ollama"

    async def test_lmstudio_provider_calls_check_lmstudio(self):
        cfg = self._make_config("lmstudio")
        with patch("server.services.health.check_lmstudio", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = HealthCheck("lmstudio", ComponentStatus.OK, "fine")
            result = await _check_active_llm(cfg)

        mock_check.assert_awaited_once_with("http://localhost:1234")
        assert result.component == "lmstudio"

    async def test_openai_provider_calls_check_openai_key(self):
        cfg = self._make_config("openai")
        with patch("server.services.health.check_openai_key") as mock_check:
            mock_check.return_value = HealthCheck("openai", ComponentStatus.OK, "fine")
            result = await _check_active_llm(cfg)

        mock_check.assert_called_once_with("sk-test-key-1234567890ab")
        assert result.component == "openai"

    async def test_unknown_provider_returns_down(self):
        cfg = self._make_config("anthropic")
        result = await _check_active_llm(cfg)
        assert result.status == ComponentStatus.DOWN
        assert "Unknown provider" in result.message


# ---------------------------------------------------------------------------
# TestRunStartupChecks
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestRunStartupChecks:
    def _make_app_config(self):
        config = MagicMock()
        config.translator.provider = "ollama"
        config.translator.ollama_url = "http://localhost:11434"
        config.translator.ollama_model = "qwen3:4b"
        config.translator.model = "qwen3:4b"
        config.db_path = "data/habla.db"
        config.diarization.hf_token = "hf_test"
        return config

    async def test_startup_checks_returns_system_health(self):
        config = self._make_app_config()

        with (
            patch("server.services.health._check_active_llm", new_callable=AsyncMock,
                  return_value=HealthCheck("ollama", ComponentStatus.OK, "ok")),
            patch("server.services.health.check_ffmpeg", new_callable=AsyncMock,
                  return_value=HealthCheck("ffmpeg", ComponentStatus.OK, "ok")),
            patch("server.services.health.check_database", new_callable=AsyncMock,
                  return_value=HealthCheck("database", ComponentStatus.OK, "ok")),
            patch("server.services.health.check_hf_token", new_callable=AsyncMock,
                  return_value=HealthCheck("hf_token", ComponentStatus.OK, "ok")),
        ):
            result = await run_startup_checks(config)

        assert isinstance(result, SystemHealth)
        assert len(result.checks) == 4

    async def test_startup_checks_all_ok(self):
        config = self._make_app_config()

        with (
            patch("server.services.health._check_active_llm", new_callable=AsyncMock,
                  return_value=HealthCheck("ollama", ComponentStatus.OK, "ok")),
            patch("server.services.health.check_ffmpeg", new_callable=AsyncMock,
                  return_value=HealthCheck("ffmpeg", ComponentStatus.OK, "ok")),
            patch("server.services.health.check_database", new_callable=AsyncMock,
                  return_value=HealthCheck("database", ComponentStatus.OK, "ok")),
            patch("server.services.health.check_hf_token", new_callable=AsyncMock,
                  return_value=HealthCheck("hf_token", ComponentStatus.OK, "ok")),
        ):
            result = await run_startup_checks(config)

        assert result.overall == ComponentStatus.OK

    async def test_startup_checks_with_failures(self):
        config = self._make_app_config()

        with (
            patch("server.services.health._check_active_llm", new_callable=AsyncMock,
                  return_value=HealthCheck("ollama", ComponentStatus.DOWN, "dead")),
            patch("server.services.health.check_ffmpeg", new_callable=AsyncMock,
                  return_value=HealthCheck("ffmpeg", ComponentStatus.OK, "ok")),
            patch("server.services.health.check_database", new_callable=AsyncMock,
                  return_value=HealthCheck("database", ComponentStatus.DEGRADED, "slow")),
            patch("server.services.health.check_hf_token", new_callable=AsyncMock,
                  return_value=HealthCheck("hf_token", ComponentStatus.OK, "ok")),
        ):
            result = await run_startup_checks(config)

        assert result.overall == ComponentStatus.DOWN
        components = {c.component: c for c in result.checks}
        assert components["ollama"].status == ComponentStatus.DOWN
        assert components["database"].status == ComponentStatus.DEGRADED
        assert components["ffmpeg"].status == ComponentStatus.OK


# ---------------------------------------------------------------------------
# TestRunRuntimeChecks
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestRunRuntimeChecks:
    async def test_runtime_checks_includes_whisperx_and_diarization(self):
        _runtime_cache["result"] = None
        _runtime_cache["timestamp"] = 0.0
        pipeline = MagicMock()
        pipeline.config.translator.provider = "ollama"
        pipeline.config.translator.ollama_url = "http://localhost:11434"
        pipeline.config.translator.ollama_model = "qwen3:4b"
        pipeline.config.db_path = "data/habla.db"
        pipeline._whisperx_model = MagicMock()
        pipeline._diarize_pipeline = MagicMock()

        with (
            patch("server.services.health._check_active_llm", new_callable=AsyncMock,
                  return_value=HealthCheck("ollama", ComponentStatus.OK, "ok")),
            patch("server.services.health.check_ffmpeg", new_callable=AsyncMock,
                  return_value=HealthCheck("ffmpeg", ComponentStatus.OK, "ok")),
            patch("server.services.health.check_database", new_callable=AsyncMock,
                  return_value=HealthCheck("database", ComponentStatus.OK, "ok")),
        ):
            result = await run_runtime_checks(pipeline)

        assert isinstance(result, SystemHealth)
        component_names = [c.component for c in result.checks]
        assert "whisperx" in component_names
        assert "diarization" in component_names
        assert len(result.checks) == 5

        whisperx_check = next(c for c in result.checks if c.component == "whisperx")
        assert whisperx_check.status == ComponentStatus.OK

        diarization_check = next(c for c in result.checks if c.component == "diarization")
        assert diarization_check.status == ComponentStatus.OK

    async def test_runtime_checks_degraded_when_models_missing(self):
        _runtime_cache["result"] = None
        _runtime_cache["timestamp"] = 0.0
        pipeline = MagicMock()
        pipeline.config.translator.provider = "ollama"
        pipeline.config.translator.ollama_url = "http://localhost:11434"
        pipeline.config.translator.ollama_model = "qwen3:4b"
        pipeline.config.db_path = "data/habla.db"
        pipeline._whisperx_model = None
        pipeline._diarize_pipeline = None

        with (
            patch("server.services.health._check_active_llm", new_callable=AsyncMock,
                  return_value=HealthCheck("ollama", ComponentStatus.OK, "ok")),
            patch("server.services.health.check_ffmpeg", new_callable=AsyncMock,
                  return_value=HealthCheck("ffmpeg", ComponentStatus.OK, "ok")),
            patch("server.services.health.check_database", new_callable=AsyncMock,
                  return_value=HealthCheck("database", ComponentStatus.OK, "ok")),
        ):
            result = await run_runtime_checks(pipeline)

        assert result.overall == ComponentStatus.DEGRADED
        whisperx_check = next(c for c in result.checks if c.component == "whisperx")
        assert whisperx_check.status == ComponentStatus.DEGRADED
        diarization_check = next(c for c in result.checks if c.component == "diarization")
        assert diarization_check.status == ComponentStatus.DEGRADED


# ---------------------------------------------------------------------------
# TestLLMHealthMonitor
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestLLMHealthMonitor:
    """Tests for the background LLM health monitoring task."""

    @pytest.mark.asyncio
    async def test_monitor_detects_provider_down(self):
        """When provider goes down, sends error to active session."""
        config = MagicMock()
        config.provider = "ollama"
        session = MagicMock()
        session._send = AsyncMock()

        check_down = HealthCheck("ollama", ComponentStatus.DOWN, "Connection refused")

        with patch("server.services.health._check_active_llm",
                   new_callable=AsyncMock, return_value=check_down):
            task = asyncio.create_task(
                run_llm_health_monitor(
                    get_config=lambda: config,
                    get_session_fn=lambda: session,
                    interval=0.05,
                )
            )
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        session._send.assert_called()
        sent = session._send.call_args[0][0]
        assert sent["type"] == "error"
        assert "unavailable" in sent["message"]

    @pytest.mark.asyncio
    async def test_monitor_detects_recovery(self):
        """When provider recovers, sends status message to session."""
        config = MagicMock()
        config.provider = "ollama"
        session = MagicMock()
        session._send = AsyncMock()

        call_count = 0
        async def alternating_check(_config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return HealthCheck("ollama", ComponentStatus.DOWN, "down")
            return HealthCheck("ollama", ComponentStatus.OK, "ok")

        with patch("server.services.health._check_active_llm", side_effect=alternating_check):
            task = asyncio.create_task(
                run_llm_health_monitor(
                    get_config=lambda: config,
                    get_session_fn=lambda: session,
                    interval=0.05,
                )
            )
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        calls = [c[0][0] for c in session._send.call_args_list]
        assert any(c["type"] == "status" and "back online" in c.get("message", "") for c in calls)

    @pytest.mark.asyncio
    async def test_monitor_no_notification_when_stable(self):
        """No messages sent when provider stays OK."""
        config = MagicMock()
        config.provider = "ollama"
        session = MagicMock()
        session._send = AsyncMock()

        check_ok = HealthCheck("ollama", ComponentStatus.OK, "ok")

        with patch("server.services.health._check_active_llm",
                   new_callable=AsyncMock, return_value=check_ok):
            task = asyncio.create_task(
                run_llm_health_monitor(
                    get_config=lambda: config,
                    get_session_fn=lambda: session,
                    interval=0.05,
                )
            )
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        session._send.assert_not_called()

    @pytest.mark.asyncio
    async def test_monitor_no_session_no_crash(self):
        """Provider goes down but no active session — should not raise."""
        config = MagicMock()
        config.provider = "ollama"

        check_down = HealthCheck("ollama", ComponentStatus.DOWN, "down")

        with patch("server.services.health._check_active_llm",
                   new_callable=AsyncMock, return_value=check_down):
            task = asyncio.create_task(
                run_llm_health_monitor(
                    get_config=lambda: config,
                    get_session_fn=lambda: None,
                    interval=0.05,
                )
            )
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # No exception means success
