"""Shared pytest fixtures for Habla tests."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import httpx

from server.config import TranslatorConfig
from server.models.schemas import SpeakerProfile


@pytest.fixture
def translator_config():
    """Default translator config for testing."""
    return TranslatorConfig(
        provider="ollama",
        ollama_url="http://localhost:11434",
        ollama_model="qwen3:4b",
        lmstudio_url="http://localhost:1234",
        lmstudio_model="test-model",
        openai_api_key="test-key",
        openai_model="gpt-5-nano",
        timeout_seconds=30.0,
        temperature=0.3,
        fallback_to_local=True,
    )


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient for LLM API calls."""
    client = AsyncMock(spec=httpx.AsyncClient)

    # Default Ollama response
    ollama_response = MagicMock()
    ollama_response.status_code = 200
    ollama_response.json.return_value = {
        "response": json.dumps({
            "corrected": "Corrected text",
            "translated": "Translated text",
            "confidence": 0.9,
            "flagged_phrases": [],
        })
    }
    ollama_response.raise_for_status = MagicMock()

    # Default LM Studio response
    lmstudio_response = MagicMock()
    lmstudio_response.status_code = 200
    lmstudio_response.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "corrected": "Corrected text",
                    "translated": "Translated text",
                    "confidence": 0.9,
                    "flagged_phrases": [],
                })
            }
        }]
    }
    lmstudio_response.raise_for_status = MagicMock()

    # Default OpenAI response
    openai_response = MagicMock()
    openai_response.status_code = 200
    openai_response.json.return_value = {
        "output": [{
            "type": "message",
            "content": [{
                "type": "output_text",
                "text": json.dumps({
                    "corrected": "Corrected text",
                    "translated": "Translated text",
                    "confidence": 0.9,
                    "flagged_phrases": [],
                })
            }]
        }],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
        }
    }
    openai_response.raise_for_status = MagicMock()

    async def mock_post(url, **kwargs):
        if "/api/generate" in url:
            return ollama_response
        elif "/v1/chat/completions" in url:
            return lmstudio_response
        elif "/v1/responses" in url:
            return openai_response
        raise ValueError(f"Unknown URL: {url}")

    async def mock_get(url):
        if "/v1/models" in url:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "data": [
                    {"id": "test-model-1"},
                    {"id": "test-model-2"},
                ]
            }
            resp.raise_for_status = MagicMock()
            return resp
        elif "/api/tags" in url:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "models": [
                    {"name": "qwen3:4b"},
                    {"name": "llama3:8b"},
                ]
            }
            resp.raise_for_status = MagicMock()
            return resp
        raise ValueError(f"Unknown URL: {url}")

    client.post = mock_post
    client.get = mock_get

    return client


@pytest.fixture
def sample_idiom_json_file(tmp_path):
    """Create a temporary JSON file with idiom patterns."""
    data = [
        {
            "pattern": r"importar\s+un\s+pepino",
            "canonical": "importar un pepino",
            "literal": "to matter a cucumber",
            "meaning": "to not care at all",
            "region": "universal",
            "frequency": "very common"
        },
        {
            "pattern": r"estar\s+en\s+las\s+nubes",
            "canonical": "estar en las nubes",
            "literal": "to be in the clouds",
            "meaning": "to be daydreaming",
            "region": "universal",
            "frequency": "common"
        }
    ]

    file_path = tmp_path / "test_idioms.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return file_path


@pytest.fixture
def sample_speaker_profiles():
    """Sample speaker profiles for testing."""
    return [
        SpeakerProfile(
            id="SPEAKER_00",
            label="Speaker A",
            custom_name="",
            color="#3B82F6",
            utterance_count=5,
        ),
        SpeakerProfile(
            id="SPEAKER_01",
            label="Speaker B",
            custom_name="Maria",
            color="#10B981",
            utterance_count=3,
            role_hint="teacher",
        ),
    ]


@pytest.fixture
def sample_translation_context():
    """Sample conversation context for translation tests."""
    return [
        {
            "speaker": "Speaker A",
            "source": "Hola, ¿cómo estás?",
            "translation": "Hello, how are you?",
        },
        {
            "speaker": "Speaker B",
            "source": "Muy bien, gracias.",
            "translation": "Very well, thanks.",
        },
    ]


# --- Playback test fixtures ---

def _write_minimal_wav(path: Path, duration_seconds: float = 0.5):
    """Write a minimal valid WAV file (16-bit mono 16kHz silence)."""
    import struct
    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


@pytest.fixture
def recordings_dir(tmp_path):
    """Create a fake recordings directory with various recording states."""
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir()

    # Recording with raw stream, segments, metadata, and ground truth
    full = rec_dir / "rec_full"
    full.mkdir()
    (full / "raw_stream.webm").write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 100)
    _write_minimal_wav(full / "segment_001.wav", 1.0)
    _write_minimal_wav(full / "segment_002.wav", 0.8)
    (full / "metadata.json").write_text(json.dumps({
        "started_at": "2026-02-22T10:00:00",
        "ended_at": "2026-02-22T10:05:00",
        "total_segments": 2,
        "segments": [
            {"segment_id": 1, "duration_seconds": 1.0},
            {"segment_id": 2, "duration_seconds": 0.8},
        ],
    }))
    (full / "ground_truth.json").write_text(json.dumps({
        "generated_at": "2026-02-22T12:00:00",
        "whisper_model": "large-v3",
        "segments": [
            {"segment_id": 1, "transcript": "Hola", "translation": "Hello"},
            {"segment_id": 2, "transcript": "Adios", "translation": "Goodbye"},
        ],
    }))

    # Recording with segments only (no raw stream)
    segs = rec_dir / "rec_segments_only"
    segs.mkdir()
    _write_minimal_wav(segs / "segment_001.wav", 0.5)
    (segs / "metadata.json").write_text(json.dumps({
        "started_at": "2026-02-21T14:00:00",
        "segments": [{"segment_id": 1, "duration_seconds": 0.5}],
    }))

    # Empty recording dir
    (rec_dir / "rec_empty").mkdir()

    return rec_dir


@pytest.fixture
def mock_session():
    """Mock ClientSession with real asyncio primitives for playback testing."""
    session = MagicMock()

    session.listening = False
    session.playback_mode = False

    # VAD mock
    session.vad = MagicMock()
    session.vad.reset = MagicMock()
    session.vad.feed_pcm = AsyncMock()
    session.vad.flush = AsyncMock()
    session.vad.on_partial_audio = None

    # Decoder mock
    session.decoder = MagicMock()
    session.decoder.reset = MagicMock()
    session.decoder.start_streaming = AsyncMock()
    session.decoder.stop_streaming = AsyncMock(return_value=b"")

    # Real asyncio primitives (can't mock these)
    session._chunk_inbox = []
    session._chunk_event = asyncio.Event()
    session._decode_task = None

    # Async method mocks
    session._continuous_decode_loop = AsyncMock()
    session._flush_pending_now = AsyncMock()
    session._on_partial_audio = AsyncMock()
    session._send = AsyncMock()

    # Pipeline mock with real queue
    session.pipeline = MagicMock()
    session.pipeline._queue = asyncio.Queue()
    session.pipeline.process_wav = AsyncMock(return_value=None)

    return session
