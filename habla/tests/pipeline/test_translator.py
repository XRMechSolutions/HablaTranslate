"""Unit tests for the Translator service."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from server.pipeline.translator import Translator, _is_retryable, _extract_field
from server.models.schemas import TranslationResult, FlaggedPhrase


class TestRetryLogic:
    """Test retry and error classification logic."""

    def test_is_retryable_timeout(self):
        """Timeout errors should be retryable."""
        error = httpx.TimeoutException("Timeout")
        assert _is_retryable(error) is True

    def test_is_retryable_connection_error(self):
        """Connection errors should be retryable."""
        error = httpx.ConnectError("Connection failed")
        assert _is_retryable(error) is True

    def test_is_retryable_server_error(self):
        """5xx errors should be retryable."""
        response = MagicMock()
        response.status_code = 500
        error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
        assert _is_retryable(error) is True

    def test_not_retryable_permanent_error(self):
        """Model not found errors should not be retryable."""
        response = MagicMock()
        response.status_code = 404
        error = httpx.HTTPStatusError("model not found", request=MagicMock(), response=response)
        assert _is_retryable(error) is False

    def test_not_retryable_client_error(self):
        """4xx errors should not be retryable."""
        response = MagicMock()
        response.status_code = 400
        error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
        assert _is_retryable(error) is False


class TestFieldExtraction:
    """Test JSON field extraction from malformed responses."""

    def test_extract_field_simple(self):
        """Extract field from well-formed JSON string."""
        text = '{"corrected": "Hello world", "translated": "Hola mundo"}'
        assert _extract_field(text, "corrected") == "Hello world"
        assert _extract_field(text, "translated") == "Hola mundo"

    def test_extract_field_with_newlines(self):
        """Extract field with embedded newlines."""
        text = '{"corrected": "Line 1\\nLine 2\\nLine 3"}'
        result = _extract_field(text, "corrected")
        assert "Line 1" in result and "Line 2" in result

    def test_extract_field_missing(self):
        """Return empty string for missing field."""
        text = '{"other": "value"}'
        assert _extract_field(text, "corrected") == ""

    def test_extract_field_malformed(self):
        """Handle malformed JSON gracefully."""
        text = 'Some text "corrected": "extracted value" more text'
        result = _extract_field(text, "corrected")
        assert result == "extracted value"


class TestTranslatorConfig:
    """Test translator configuration and initialization."""

    def test_init_default_config(self, translator_config):
        """Initialize translator with default config."""
        translator = Translator(translator_config)
        assert translator.config.provider == "ollama"
        assert translator.config.ollama_model == "qwen3:4b"
        assert translator._metrics["requests"] == 0

    def test_metrics_initial_state(self, translator_config):
        """Metrics should start at zero."""
        translator = Translator(translator_config)
        metrics = translator.metrics
        assert metrics["requests"] == 0
        assert metrics["successes"] == 0
        assert metrics["failures"] == 0
        assert metrics["retries"] == 0
        assert metrics["provider"] == "ollama"

    def test_costs_initial_state(self, translator_config):
        """OpenAI costs should start at zero."""
        translator = Translator(translator_config)
        costs = translator.costs
        assert costs["session_cost_usd"] == 0.0
        assert costs["all_time_cost_usd"] == 0.0
        assert costs["session_input_tokens"] == 0


class TestProviderSwitching:
    """Test runtime provider switching."""

    def test_switch_provider_ollama(self, translator_config):
        """Switch to Ollama provider."""
        translator = Translator(translator_config)
        translator.switch_provider("ollama", model="llama3:8b")
        assert translator.config.provider == "ollama"
        assert translator.config.ollama_model == "llama3:8b"

    def test_switch_provider_lmstudio(self, translator_config):
        """Switch to LM Studio provider."""
        translator = Translator(translator_config)
        translator.switch_provider("lmstudio", model="phi-4", url="http://localhost:1234")
        assert translator.config.provider == "lmstudio"
        assert translator.config.lmstudio_model == "phi-4"
        assert translator.config.lmstudio_url == "http://localhost:1234"

    def test_switch_provider_openai(self, translator_config):
        """Switch to OpenAI provider."""
        translator = Translator(translator_config)
        translator.switch_provider("openai", model="gpt-5-mini")
        assert translator.config.provider == "openai"
        assert translator.config.openai_model == "gpt-5-mini"

    def test_switch_provider_resets_session_costs(self, translator_config):
        """Switching provider should reset session costs."""
        translator = Translator(translator_config)
        translator._costs["session_cost_usd"] = 1.50
        translator._costs["session_input_tokens"] = 1000
        translator.switch_provider("lmstudio", model="test")
        assert translator._costs["session_cost_usd"] == 0.0
        assert translator._costs["session_input_tokens"] == 0

    def test_switch_provider_invalid(self, translator_config):
        """Invalid provider should raise ValueError."""
        translator = Translator(translator_config)
        with pytest.raises(ValueError, match="Unknown provider"):
            translator.switch_provider("invalid_provider")


class TestAutoDetectModel:
    """Test auto-detection of LM Studio models."""

    @pytest.mark.asyncio
    async def test_auto_detect_lmstudio_model(self, translator_config, mock_httpx_client):
        """Auto-detect LM Studio model when none configured."""
        translator_config.provider = "lmstudio"
        translator_config.lmstudio_model = ""
        translator = Translator(translator_config)
        translator.client = mock_httpx_client

        await translator.auto_detect_model()
        assert translator.config.lmstudio_model == "test-model-1"

    @pytest.mark.asyncio
    async def test_auto_detect_skips_if_already_configured(self, translator_config, mock_httpx_client):
        """Auto-detect should skip if model already configured."""
        translator_config.provider = "lmstudio"
        translator_config.lmstudio_model = "existing-model"
        translator = Translator(translator_config)
        translator.client = mock_httpx_client

        await translator.auto_detect_model()
        assert translator.config.lmstudio_model == "existing-model"

    @pytest.mark.asyncio
    async def test_auto_detect_skips_non_lmstudio(self, translator_config, mock_httpx_client):
        """Auto-detect should only run for LM Studio."""
        translator_config.provider = "ollama"
        translator = Translator(translator_config)
        translator.client = mock_httpx_client

        await translator.auto_detect_model()
        # Should not change anything


class TestOllamaProvider:
    """Test Ollama-specific API calls."""

    @pytest.mark.asyncio
    async def test_call_ollama_success(self, translator_config, mock_httpx_client):
        """Successful Ollama API call."""
        translator = Translator(translator_config)
        translator.client = mock_httpx_client

        result = await translator._call_ollama(
            "System prompt", "User prompt", temperature=0.3, max_tokens=512, json_mode=True
        )

        assert "corrected" in result
        assert "translated" in result

    @pytest.mark.asyncio
    async def test_call_ollama_json_mode(self, translator_config):
        """Ollama should include format: json when json_mode is True."""
        translator = Translator(translator_config)

        # Track the POST call
        post_body = {}

        async def mock_post(url, **kwargs):
            nonlocal post_body
            post_body.update(kwargs.get("json", {}))
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {"response": '{"corrected": "test", "translated": "test"}'}
            response.raise_for_status = MagicMock()
            return response

        translator.client.post = mock_post

        await translator._call_ollama(
            "System prompt", "User prompt", temperature=0.3, max_tokens=512, json_mode=True
        )

        # Verify the POST call included "format": "json"
        assert post_body.get("format") == "json"


class TestLMStudioProvider:
    """Test LM Studio-specific API calls."""

    @pytest.mark.asyncio
    async def test_call_lmstudio_success(self, translator_config, mock_httpx_client):
        """Successful LM Studio API call."""
        translator_config.provider = "lmstudio"
        translator = Translator(translator_config)
        translator.client = mock_httpx_client

        result = await translator._call_lmstudio(
            "System prompt", "User prompt", temperature=0.3, max_tokens=512, json_mode=True
        )

        assert "corrected" in result
        assert "translated" in result

    @pytest.mark.asyncio
    async def test_call_lmstudio_uses_model_override(self, translator_config):
        """LM Studio should use model override if provided."""
        translator_config.provider = "lmstudio"
        translator = Translator(translator_config)

        post_body = {}

        async def mock_post(url, **kwargs):
            nonlocal post_body
            post_body.update(kwargs.get("json", {}))
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "choices": [{"message": {"content": '{"corrected": "test", "translated": "test"}'}}]
            }
            response.raise_for_status = MagicMock()
            return response

        translator.client.post = mock_post

        await translator._call_lmstudio(
            "System", "User", temperature=0.3, max_tokens=512, json_mode=True,
            model_override="override-model"
        )

        assert post_body.get("model") == "override-model"


class TestOpenAIProvider:
    """Test OpenAI-specific API calls."""

    @pytest.mark.asyncio
    async def test_call_openai_success(self, translator_config, mock_httpx_client):
        """Successful OpenAI API call."""
        translator_config.provider = "openai"
        translator = Translator(translator_config)
        translator.client = mock_httpx_client

        result = await translator._call_openai(
            "System prompt", "User prompt", temperature=0.3, max_tokens=512, json_mode=True
        )

        assert "corrected" in result
        assert "translated" in result

    @pytest.mark.asyncio
    async def test_call_openai_tracks_costs(self, translator_config, mock_httpx_client):
        """OpenAI calls should track token usage and costs."""
        translator_config.provider = "openai"
        translator = Translator(translator_config)
        translator.client = mock_httpx_client

        await translator._call_openai(
            "System", "User", temperature=0.3, max_tokens=512, json_mode=True
        )

        costs = translator.costs
        assert costs["session_input_tokens"] == 100
        assert costs["session_output_tokens"] == 50
        assert costs["session_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_call_openai_uses_responses_api(self, translator_config):
        """OpenAI should use /v1/responses endpoint, not /v1/chat/completions."""
        translator_config.provider = "openai"
        translator = Translator(translator_config)

        called_url = None

        async def mock_post(url, **kwargs):
            nonlocal called_url
            called_url = url
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"corrected": "test", "translated": "test"}'}]
                }],
                "usage": {"input_tokens": 100, "output_tokens": 50}
            }
            response.raise_for_status = MagicMock()
            return response

        translator.client.post = mock_post

        await translator._call_openai(
            "System", "User", temperature=0.3, max_tokens=512, json_mode=True
        )

        assert "/v1/responses" in called_url


class TestTranslationParsing:
    """Test parsing of LLM responses into TranslationResult."""

    def test_parse_valid_json(self, translator_config):
        """Parse well-formed JSON response."""
        translator = Translator(translator_config)
        raw = json.dumps({
            "corrected": "Corrected text",
            "translated": "Translated text",
            "confidence": 0.9,
            "flagged_phrases": [],
        })

        result = translator._parse_response(raw, "Original")
        assert result.corrected == "Corrected text"
        assert result.translated == "Translated text"
        assert result.confidence == 0.9

    def test_parse_with_flagged_phrases(self, translator_config):
        """Parse response with idioms."""
        translator = Translator(translator_config)
        raw = json.dumps({
            "corrected": "Text",
            "translated": "Translation",
            "confidence": 0.8,
            "flagged_phrases": [
                {
                    "phrase": "tomar el pelo",
                    "literal": "to take the hair",
                    "meaning": "to pull someone's leg",
                    "type": "idiom",
                    "save_worthy": True,
                }
            ],
        })

        result = translator._parse_response(raw, "Original")
        assert len(result.flagged_phrases) == 1
        assert result.flagged_phrases[0].phrase == "tomar el pelo"
        assert result.flagged_phrases[0].source == "llm"

    def test_parse_with_correction_detail(self, translator_config):
        """Parse response with correction detail."""
        translator = Translator(translator_config)
        raw = json.dumps({
            "corrected": "Text",
            "translated": "Translation",
            "confidence": 0.8,
            "is_correction": True,
            "correction_detail": {
                "wrong": "yo es",
                "right": "yo soy",
                "explanation": "Verb conjugation error",
            },
        })

        result = translator._parse_response(raw, "Original")
        assert result.is_correction is True
        assert result.correction_detail is not None
        assert result.correction_detail.wrong == "yo es"
        assert result.correction_detail.right == "yo soy"

    def test_parse_strips_think_tags(self, translator_config):
        """Parse response with chain-of-thought tags."""
        translator = Translator(translator_config)
        raw = '<think>Reasoning here</think>{"corrected": "Text", "translated": "Translation", "confidence": 0.7}'

        result = translator._parse_response(raw, "Original")
        assert result.corrected == "Text"
        assert result.translated == "Translation"

    def test_parse_strips_markdown_code_blocks(self, translator_config):
        """Parse response wrapped in markdown code blocks."""
        translator = Translator(translator_config)
        raw = '```json\n{"corrected": "Text", "translated": "Translation", "confidence": 0.7}\n```'

        result = translator._parse_response(raw, "Original")
        assert result.corrected == "Text"
        assert result.translated == "Translation"

    def test_parse_fallback_on_invalid_json(self, translator_config):
        """Fallback extraction when JSON is malformed."""
        translator = Translator(translator_config)
        raw = 'Some text "corrected": "Fixed text" and "translated": "Traducción" here'

        result = translator._parse_response(raw, "Original")
        assert result.corrected == "Fixed text"
        assert result.translated == "Traducción"
        assert result.confidence == 0.1  # Low confidence on fallback

    def test_parse_complete_failure(self, translator_config):
        """Complete parse failure returns original as translation."""
        translator = Translator(translator_config)
        raw = "Completely unparseable garbage"

        result = translator._parse_response(raw, "Original text")
        assert result.corrected == "Original text"
        assert "garbage" in result.translated.lower()


class TestRetryBehavior:
    """Test retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, translator_config):
        """Timeout should trigger retry with backoff."""
        translator = Translator(translator_config)

        # Make first call timeout, second succeed
        call_count = 0

        async def mock_post_with_timeout(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("Timeout")
            # Second call succeeds
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {"response": '{"corrected": "test", "translated": "test"}'}
            response.raise_for_status = MagicMock()
            return response

        translator.client.post = mock_post_with_timeout

        result = await translator._call_provider(
            "ollama", "System", "User", retries=1, max_tokens=512, temperature=0.3
        )

        assert call_count == 2  # First failed, second succeeded
        assert translator._metrics["retries"] == 1
        assert translator._metrics["timeouts"] == 1
        assert translator._metrics["successes"] == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_permanent_error(self, translator_config, mock_httpx_client):
        """Permanent errors should fail immediately."""
        translator = Translator(translator_config)
        translator.client = mock_httpx_client

        async def mock_post_permanent_error(url, **kwargs):
            response = MagicMock()
            response.status_code = 404
            response.text = "model not found"
            raise httpx.HTTPStatusError("model not found", request=MagicMock(), response=response)

        translator.client.post = mock_post_permanent_error

        with pytest.raises(httpx.HTTPStatusError):
            await translator._call_provider(
                "ollama", "System", "User", retries=3, max_tokens=512, temperature=0.3
            )

        assert translator._metrics["retries"] == 0  # No retries on permanent error
        assert translator._metrics["failures"] == 1


class TestFallbackLogic:
    """Test provider fallback behavior."""

    @pytest.mark.asyncio
    async def test_no_fallback_from_local_provider(self, translator_config):
        """Local providers should never fall back to cloud."""
        translator_config.provider = "ollama"
        translator_config.fallback_to_local = True
        translator = Translator(translator_config)

        async def mock_post_always_fail(url, **kwargs):
            raise httpx.TimeoutException("Always fails")

        translator.client.post = mock_post_always_fail

        with pytest.raises(httpx.TimeoutException):
            await translator._call_llm("System", "User", retries=0)

    @pytest.mark.asyncio
    async def test_fallback_disabled(self, translator_config):
        """Fallback should not occur if disabled."""
        translator_config.provider = "openai"
        translator_config.fallback_to_local = False
        translator = Translator(translator_config)

        async def mock_post_always_fail(url, **kwargs):
            raise httpx.TimeoutException("Always fails")

        translator.client.post = mock_post_always_fail

        with pytest.raises(httpx.TimeoutException):
            await translator._call_llm("System", "User", retries=0)


class TestTranslateMethod:
    """Test the main translate() method."""

    @pytest.mark.asyncio
    async def test_translate_success(self, translator_config, mock_httpx_client):
        """Full translation flow should succeed."""
        translator = Translator(translator_config)
        translator.client = mock_httpx_client

        result = await translator.translate(
            transcript="Hola, ¿cómo estás?",
            speaker_label="Speaker A",
            direction="es_to_en",
            mode="conversation",
            context_exchanges=[],
            topic_summary="",
        )

        assert isinstance(result, TranslationResult)
        assert result.corrected
        assert result.translated
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_translate_timeout_returns_fallback(self, translator_config):
        """Timeout should return transcript with error marker."""
        translator = Translator(translator_config)

        async def mock_post_timeout(url, **kwargs):
            raise httpx.TimeoutException("Timeout")

        translator.client.post = mock_post_timeout

        result = await translator.translate(
            transcript="Original text",
            speaker_label="Speaker A",
            direction="es_to_en",
            mode="conversation",
            context_exchanges=[],
        )

        assert result.corrected == "Original text"
        assert "[translation timeout]" in result.translated
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_translate_error_returns_fallback(self, translator_config):
        """Other errors should return transcript with error marker."""
        translator = Translator(translator_config)

        async def mock_post_error(url, **kwargs):
            raise ValueError("Something broke")

        translator.client.post = mock_post_error

        result = await translator.translate(
            transcript="Original text",
            speaker_label="Speaker A",
            direction="es_to_en",
            mode="conversation",
            context_exchanges=[],
        )

        assert result.corrected == "Original text"
        assert "[translation error]" in result.translated
        assert result.confidence == 0.0


class TestTopicSummary:
    """Test topic summary updates."""

    @pytest.mark.asyncio
    async def test_update_topic_summary_success(self, translator_config, mock_httpx_client):
        """Topic summary should update successfully."""
        translator = Translator(translator_config)
        translator.client = mock_httpx_client

        # Mock response for topic summary
        summary_response = MagicMock()
        summary_response.status_code = 200
        summary_response.json.return_value = {"response": "Updated topic summary"}
        summary_response.raise_for_status = MagicMock()

        async def mock_post_summary(url, **kwargs):
            return summary_response

        translator.client.post = mock_post_summary

        result = await translator.update_topic_summary(
            previous_summary="Previous summary",
            latest_source="Hola",
            latest_translation="Hello",
            speaker_label="Speaker A",
        )

        assert result == "Updated topic summary"

    @pytest.mark.asyncio
    async def test_update_topic_summary_failure_returns_previous(self, translator_config):
        """Failed topic summary should return previous summary."""
        translator = Translator(translator_config)

        async def mock_post_error(url, **kwargs):
            raise httpx.TimeoutException("Timeout")

        translator.client.post = mock_post_error

        result = await translator.update_topic_summary(
            previous_summary="Previous summary",
            latest_source="Hola",
            latest_translation="Hello",
            speaker_label="Speaker A",
        )

        assert result == "Previous summary"
