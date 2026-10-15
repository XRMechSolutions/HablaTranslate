"""Unit tests for VAD buffer and audio decoder."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from server.pipeline.vad_buffer import VADConfig, StreamingVADBuffer, AudioDecoder, _webm_init_segment


class TestVADConfig:
    """Test VAD configuration."""

    def test_default_config(self):
        """VADConfig should have sensible defaults."""
        config = VADConfig()
        assert config.sample_rate == 16000
        assert config.silence_duration_ms == 600
        assert config.min_speech_ms == 200  # live value (AudioConfig), from shared/asr_profile.json
        assert config.frame_ms == 32  # one 512-sample frame at 16 kHz
        assert config.speech_threshold == 0.35

    def test_custom_config(self):
        """VADConfig should accept custom values."""
        config = VADConfig(
            silence_duration_ms=800,
            min_speech_ms=500,
            speech_threshold=0.5,
        )
        assert config.silence_duration_ms == 800
        assert config.min_speech_ms == 500
        assert config.speech_threshold == 0.5


class TestStreamingVADBufferInit:
    """Test StreamingVADBuffer initialization."""

    def test_init_default(self):
        """VAD buffer should initialize with defaults."""
        vad = StreamingVADBuffer()
        assert vad.config.sample_rate == 16000
        assert vad.segments_emitted == 0
        assert vad._is_speaking is False

    def test_init_with_config(self):
        """VAD buffer should accept custom config."""
        config = VADConfig(silence_duration_ms=1000)
        vad = StreamingVADBuffer(config=config)
        assert vad.config.silence_duration_ms == 1000

    def test_init_with_callback(self):
        """VAD buffer should accept callback."""
        async def callback(pcm, duration):
            pass

        vad = StreamingVADBuffer(on_segment=callback)
        assert vad.on_segment == callback

    def test_init_computes_frame_thresholds(self):
        """VAD should compute silence/speech frame thresholds."""
        config = VADConfig(silence_duration_ms=600, frame_ms=30)
        vad = StreamingVADBuffer(config=config)
        # 600ms / 30ms per frame = 20 frames
        assert vad._silence_threshold_frames == 20


class TestVADInitialize:
    """Test VAD model loading."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Successful initialization should load Silero VAD."""
        vad = StreamingVADBuffer()

        # Patch torch.hub.load inside the initialize method context
        import sys
        mock_torch = MagicMock()
        mock_model = MagicMock()
        mock_utils = MagicMock()
        mock_torch.hub.load.return_value = (mock_model, mock_utils)

        with patch.dict(sys.modules, {'torch': mock_torch}):
            await vad.initialize()

            assert vad._vad_ready is True
            assert vad._vad_model is not None

    @pytest.mark.asyncio
    async def test_initialize_failure_fallback(self):
        """Failed initialization should fall back to energy-based VAD."""
        vad = StreamingVADBuffer()

        import sys
        mock_torch = MagicMock()
        mock_torch.hub.load.side_effect = Exception("Torch not available")

        with patch.dict(sys.modules, {'torch': mock_torch}):
            await vad.initialize()

            assert vad._vad_ready is False


class TestVADSpeechDetection:
    """Test speech detection logic."""

    def test_is_speech_frame_energy_based(self):
        """Energy-based VAD should detect loud audio as speech. Returns (is_speech, prob)."""
        vad = StreamingVADBuffer()
        vad._vad_ready = False  # Force energy-based detection

        # Create loud frame
        loud_frame = np.random.randint(-5000, 5000, size=512, dtype=np.int16)
        is_speech, _ = vad._is_speech_frame(loud_frame)
        assert bool(is_speech) is True

        # Create quiet frame
        quiet_frame = np.zeros(512, dtype=np.int16)
        is_speech, _ = vad._is_speech_frame(quiet_frame)
        assert bool(is_speech) is False

    def test_is_speech_frame_wrong_size(self):
        """Wrong frame size should return (False, 0.0)."""
        vad = StreamingVADBuffer()
        wrong_size_frame = np.zeros(256, dtype=np.int16)
        is_speech, prob = vad._is_speech_frame(wrong_size_frame)
        assert is_speech is False
        assert prob == 0.0

    def test_is_speech_frame_with_model(self):
        """Model-based VAD should use Silero threshold. Returns (is_speech, prob)."""
        import sys
        vad = StreamingVADBuffer()
        vad._vad_ready = True

        # Mock torch module
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_torch.from_numpy.return_value = mock_tensor

        # Mock Silero VAD model
        mock_model = MagicMock()
        mock_model.return_value.item.return_value = 0.8  # High speech probability
        vad._vad_model = mock_model

        frame = np.zeros(512, dtype=np.int16)

        with patch.dict(sys.modules, {'torch': mock_torch}):
            is_speech, prob = vad._is_speech_frame(frame)

        assert is_speech is True
        assert prob == pytest.approx(0.8)

        # Low probability should return False
        mock_model.return_value.item.return_value = 0.2
        with patch.dict(sys.modules, {'torch': mock_torch}):
            is_speech, prob = vad._is_speech_frame(frame)
        assert is_speech is False
        assert prob == pytest.approx(0.2)

    def test_model_input_is_normalized(self):
        """Silero gets float32 samples in [-1, 1], not raw int16 values."""
        import torch
        vad = StreamingVADBuffer()
        vad._vad_ready = True
        seen = {}

        def model(tensor, sr):
            seen["t"] = tensor
            return torch.tensor(0.1)

        vad._vad_model = model
        frame = np.full(512, -32768, dtype=np.int16)
        frame[0] = 16384
        vad._is_speech_frame(frame)
        t = seen["t"]
        assert t.dtype == torch.float32
        assert float(t.min()) == -1.0
        assert float(t[0]) == 0.5

    def test_reset_clears_model_state(self):
        vad = StreamingVADBuffer()
        vad._vad_model = MagicMock()
        vad.reset()
        vad._vad_model.reset_states.assert_called_once()


class TestNoiseFloorCalibration:
    """Test noise floor calibration on session start."""

    def test_vad_config_has_calibration_duration(self):
        """VADConfig should have calibration_duration_ms default."""
        config = VADConfig()
        assert config.calibration_duration_ms == 2000

    def test_new_vad_starts_calibrating(self):
        """New VAD buffer should start in calibrating state."""
        vad = StreamingVADBuffer()
        assert vad._calibrating is True
        assert vad._calibration_frame_count == 0

    @pytest.mark.asyncio
    async def test_calibration_completes_after_duration(self):
        """Calibration should complete after configured duration worth of frames."""
        config = VADConfig(calibration_duration_ms=90, frame_ms=30)  # 3 frames
        vad = StreamingVADBuffer(config=config)
        vad._vad_ready = False

        # Feed 3 frames of quiet audio (enough to complete calibration)
        quiet_frame = np.full(512, 100, dtype=np.int16).tobytes()
        for _ in range(3):
            await vad.feed_pcm(quiet_frame)

        assert vad._calibrating is False
        assert vad._noise_floor_rms > 0

    @pytest.mark.asyncio
    async def test_calibration_sets_threshold_from_noise(self):
        """Energy threshold should be set based on calibrated noise floor."""
        config = VADConfig(calibration_duration_ms=30, frame_ms=30)  # 1 frame
        vad = StreamingVADBuffer(config=config)
        vad._vad_ready = False

        # Feed a frame with known energy
        frame = np.full(512, 200, dtype=np.int16).tobytes()
        await vad.feed_pcm(frame)

        # Threshold = max(300, noise_floor * 3.0)
        expected_threshold = max(300.0, vad._noise_floor_rms * 3.0)
        assert vad._energy_threshold == pytest.approx(expected_threshold)

    @pytest.mark.asyncio
    async def test_calibration_frames_not_detected_as_speech(self):
        """During calibration, frames should not trigger speech detection."""
        config = VADConfig(calibration_duration_ms=120, frame_ms=30)  # 4 frames
        vad = StreamingVADBuffer(config=config)
        vad._vad_ready = False

        # Feed loud frames during calibration — should NOT trigger speech
        loud_frame = np.random.randint(-5000, 5000, size=512, dtype=np.int16).tobytes()
        for _ in range(3):  # still calibrating (need 4 frames)
            await vad.feed_pcm(loud_frame)

        assert vad._calibrating is True
        assert vad._is_speaking is False
        assert len(vad._speech_buffer) == 0

    @pytest.mark.asyncio
    async def test_loud_room_raises_threshold(self):
        """A loud calibration environment should raise the energy threshold above default."""
        config = VADConfig(calibration_duration_ms=30, frame_ms=30)
        vad = StreamingVADBuffer(config=config)
        vad._vad_ready = False

        # Feed a very loud frame
        loud_frame = np.full(512, 2000, dtype=np.int16).tobytes()
        await vad.feed_pcm(loud_frame)

        # Threshold should be well above the default 500
        assert vad._energy_threshold > 500.0

    def test_reset_re_enables_calibration(self):
        """Reset should re-enable calibration for new session."""
        vad = StreamingVADBuffer()
        # Simulate completed calibration
        vad._calibrating = False
        vad._calibration_frame_count = 100
        vad._calibration_energy_sum = 50000.0

        vad.reset()

        assert vad._calibrating is True
        assert vad._calibration_frame_count == 0
        assert vad._calibration_energy_sum == 0.0


class TestVADFeedPCM:
    """Test PCM audio feeding and segmentation."""

    @pytest.mark.asyncio
    async def test_feed_pcm_accumulates_to_frame_boundary(self):
        """PCM should accumulate until full frame available."""
        vad = StreamingVADBuffer()
        vad._vad_ready = False

        # Feed partial frame
        partial = np.zeros(256, dtype=np.int16).tobytes()
        await vad.feed_pcm(partial)

        # Buffer should hold the partial
        assert len(vad._pcm_buffer) == len(partial)

    @pytest.mark.asyncio
    async def test_feed_pcm_odd_bytes_truncated(self):
        """Odd-length PCM should be truncated to align."""
        vad = StreamingVADBuffer()

        # Feed odd number of bytes
        odd_bytes = b"123"
        await vad.feed_pcm(odd_bytes)

        # Should truncate 1 byte
        assert len(vad._pcm_buffer) == 2

    @pytest.mark.asyncio
    async def test_feed_pcm_speech_onset(self):
        """Speech onset should trigger state change."""
        config = VADConfig(calibration_duration_ms=0)
        vad = StreamingVADBuffer(config=config)
        vad._vad_ready = False

        # Create loud frame (512 samples * 2 bytes = 1024 bytes)
        loud_frame = np.random.randint(-5000, 5000, size=512, dtype=np.int16).tobytes()

        await vad.feed_pcm(loud_frame)

        assert vad._is_speaking is True
        assert len(vad._speech_buffer) > 0

    @pytest.mark.asyncio
    async def test_feed_pcm_silence_during_speech(self):
        """Silence during speech should increment counter."""
        config = VADConfig(calibration_duration_ms=0)
        vad = StreamingVADBuffer(config=config)
        vad._vad_ready = False
        vad._is_speaking = True

        # Feed quiet frame
        quiet_frame = np.zeros(512, dtype=np.int16).tobytes()
        await vad.feed_pcm(quiet_frame)

        assert vad._silence_frames > 0

    @pytest.mark.asyncio
    async def test_feed_pcm_segment_emission(self):
        """Sufficient silence should emit segment."""
        segment_received = []

        async def callback(pcm, duration, audio_rms, vad_avg_prob):
            segment_received.append((pcm, duration, audio_rms, vad_avg_prob))

        config = VADConfig(silence_duration_ms=60, frame_ms=30, calibration_duration_ms=0)  # 2 frames of silence
        vad = StreamingVADBuffer(config=config, on_segment=callback)
        vad._vad_ready = False

        # Speech onset
        loud_frame = np.random.randint(-5000, 5000, size=512, dtype=np.int16).tobytes()
        await vad.feed_pcm(loud_frame)

        # Add more speech to meet minimum
        for _ in range(15):  # Ensure min_speech_frames is met
            await vad.feed_pcm(loud_frame)

        # Now silence
        quiet_frame = np.zeros(512, dtype=np.int16).tobytes()
        await vad.feed_pcm(quiet_frame)
        await vad.feed_pcm(quiet_frame)

        assert len(segment_received) == 1
        assert vad.segments_emitted == 1

    @pytest.mark.asyncio
    async def test_feed_pcm_discards_short_segments(self):
        """Segments below min_speech_ms should be discarded."""
        segment_received = []

        async def callback(pcm, duration):
            segment_received.append((pcm, duration))

        config = VADConfig(min_speech_ms=1000, silence_duration_ms=60, frame_ms=30, calibration_duration_ms=0)
        vad = StreamingVADBuffer(config=config, on_segment=callback)
        vad._vad_ready = False

        # Very short speech
        loud_frame = np.random.randint(-5000, 5000, size=512, dtype=np.int16).tobytes()
        await vad.feed_pcm(loud_frame)

        # Silence
        quiet_frame = np.zeros(512, dtype=np.int16).tobytes()
        for _ in range(5):
            await vad.feed_pcm(quiet_frame)

        # No segment should be emitted (too short)
        assert len(segment_received) == 0

    @pytest.mark.asyncio
    async def test_feed_pcm_max_segment_split(self):
        """Long monologues should force split at max_segment_seconds."""
        segment_received = []

        async def callback(pcm, duration, audio_rms, vad_avg_prob):
            segment_received.append((pcm, duration, audio_rms, vad_avg_prob))

        config = VADConfig(max_segment_seconds=1.0, frame_ms=30, min_speech_ms=100, calibration_duration_ms=0)
        vad = StreamingVADBuffer(config=config, on_segment=callback)
        vad._vad_ready = False

        # Feed continuous speech far beyond max
        loud_frame = np.random.randint(-5000, 5000, size=512, dtype=np.int16).tobytes()
        for _ in range(50):  # ~1.5 seconds of speech
            await vad.feed_pcm(loud_frame)

        # Should have forced at least one split
        assert len(segment_received) >= 1


class TestVADPartialAudio:
    """Test streaming partial audio snapshots."""

    @pytest.mark.asyncio
    async def test_partial_audio_emission(self):
        """Partial audio should emit during long speech."""
        partials_received = []

        async def partial_callback(pcm, duration):
            partials_received.append((pcm, duration))

        config = VADConfig(frame_ms=30, min_speech_ms=100, calibration_duration_ms=0)
        vad = StreamingVADBuffer(config=config)
        vad.on_partial_audio = partial_callback
        vad._vad_ready = False

        # Feed speech continuously
        loud_frame = np.random.randint(-5000, 5000, size=512, dtype=np.int16).tobytes()
        for _ in range(40):  # ~1.2 seconds
            await vad.feed_pcm(loud_frame)

        # Should have emitted at least one partial
        assert len(partials_received) >= 1

    @pytest.mark.asyncio
    async def test_partial_audio_not_before_min_speech(self):
        """Partial audio should not emit before min_speech_ms."""
        partials_received = []

        async def partial_callback(pcm, duration):
            partials_received.append((pcm, duration))

        config = VADConfig(min_speech_ms=1000, calibration_duration_ms=0)
        vad = StreamingVADBuffer(config=config)
        vad.on_partial_audio = partial_callback
        vad._vad_ready = False

        # Feed just a little speech
        loud_frame = np.random.randint(-5000, 5000, size=512, dtype=np.int16).tobytes()
        for _ in range(5):
            await vad.feed_pcm(loud_frame)

        # No partials yet (below min_speech_ms)
        assert len(partials_received) == 0


class TestVADFlushAndReset:
    """Test flushing and resetting VAD state."""

    @pytest.mark.asyncio
    async def test_flush_emits_ongoing_speech(self):
        """Flush should emit any ongoing speech."""
        segment_received = []

        async def callback(pcm, duration, audio_rms, vad_avg_prob):
            segment_received.append((pcm, duration, audio_rms, vad_avg_prob))

        config = VADConfig(min_speech_ms=100, calibration_duration_ms=0)
        vad = StreamingVADBuffer(config=config, on_segment=callback)
        vad._vad_ready = False

        # Start speech
        loud_frame = np.random.randint(-5000, 5000, size=512, dtype=np.int16).tobytes()
        for _ in range(20):
            await vad.feed_pcm(loud_frame)

        # Flush without silence
        await vad.flush()

        assert len(segment_received) == 1

    @pytest.mark.asyncio
    async def test_flush_does_not_emit_short_speech(self):
        """Flush should not emit speech below min_speech_ms."""
        segment_received = []

        async def callback(pcm, duration, audio_rms, vad_avg_prob):
            segment_received.append((pcm, duration, audio_rms, vad_avg_prob))

        config = VADConfig(min_speech_ms=1000, calibration_duration_ms=0)
        vad = StreamingVADBuffer(config=config, on_segment=callback)
        vad._vad_ready = False

        # Very short speech
        loud_frame = np.random.randint(-5000, 5000, size=512, dtype=np.int16).tobytes()
        await vad.feed_pcm(loud_frame)

        await vad.flush()

        assert len(segment_received) == 0

    def test_reset_clears_state(self):
        """Reset should clear all buffers and state."""
        vad = StreamingVADBuffer()
        vad._is_speaking = True
        vad._speech_frames = 10
        vad._pcm_buffer.extend(b"test")
        vad._speech_buffer.extend(b"speech")

        vad.reset()

        assert vad._is_speaking is False
        assert vad._speech_frames == 0
        assert len(vad._pcm_buffer) == 0
        assert len(vad._speech_buffer) == 0


class TestAudioDecoderInit:
    """Test AudioDecoder initialization."""

    def test_init_default(self):
        """AudioDecoder should initialize with defaults."""
        decoder = AudioDecoder()
        assert decoder.sample_rate == 16000
        assert decoder._process is None

    def test_init_custom_sample_rate(self):
        """AudioDecoder should accept custom sample rate."""
        decoder = AudioDecoder(sample_rate=48000)
        assert decoder.sample_rate == 48000


class TestAudioDecoderStreaming:
    """Test streaming audio decode mode."""

    @pytest.mark.asyncio
    async def test_start_streaming(self):
        """start_streaming should spawn ffmpeg process."""
        decoder = AudioDecoder()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            # Use MagicMock for process; only async methods get AsyncMock.
            # stdin.write is sync, stdin.drain/wait_closed are async.
            mock_stdin = MagicMock()
            mock_stdin.close = MagicMock()
            mock_stdin.wait_closed = AsyncMock()
            mock_stdin.drain = AsyncMock()
            mock_stdout = MagicMock()
            mock_stdout.read = AsyncMock(side_effect=[b"", None])
            mock_process = MagicMock()
            mock_process.stdin = mock_stdin
            mock_process.stdout = mock_stdout
            mock_process.wait = AsyncMock()
            mock_process.returncode = None
            mock_exec.return_value = mock_process

            await decoder.start_streaming()

            assert decoder._process is not None
            assert decoder._read_task is not None

            # Cleanup
            await decoder.stop_streaming()

    @pytest.mark.asyncio
    async def test_stop_streaming(self):
        """stop_streaming should close ffmpeg and return remaining PCM."""
        decoder = AudioDecoder()

        # Create mock process
        mock_process = MagicMock()
        mock_stdin = MagicMock()
        mock_stdin.close = MagicMock()
        mock_stdin.wait_closed = AsyncMock()
        mock_process.stdin = mock_stdin
        mock_process.wait = AsyncMock()

        decoder._process = mock_process

        # Create mock task
        async def mock_task():
            pass

        decoder._read_task = asyncio.create_task(mock_task())
        decoder._pcm_chunks = [b"remaining"]

        pcm = await decoder.stop_streaming()

        assert pcm == b"remaining"
        assert decoder._process is None

    @pytest.mark.asyncio
    async def test_feed_chunk(self):
        """feed_chunk should write to ffmpeg stdin."""
        decoder = AudioDecoder()
        # stdin.write is sync; only stdin.drain is async.
        mock_stdin = MagicMock()
        mock_stdin.drain = AsyncMock()
        decoder._process = MagicMock()
        decoder._process.returncode = None  # process is running
        decoder._process.stdin = mock_stdin

        await decoder.feed_chunk(b"test_webm_data")

        mock_stdin.write.assert_called_once_with(b"test_webm_data")

    @pytest.mark.asyncio
    async def test_feed_chunk_no_process(self):
        """feed_chunk without active process should return empty."""
        decoder = AudioDecoder()
        pcm = await decoder.feed_chunk(b"data")
        assert pcm == b""


WEBM_HEADER = b"\x1a\x45\xdf\xa3" + b"segment-info-and-tracks"
WEBM_FIRST_CHUNK = WEBM_HEADER + b"\x1f\x43\xb6\x75" + b"first-cluster-audio"


def _mock_ffmpeg(returncode=None):
    """A mock ffmpeg process; stdin.write is sync, drain/wait_closed are async."""
    stdin = MagicMock()
    stdin.drain = AsyncMock()
    stdin.wait_closed = AsyncMock()
    stdout = MagicMock()
    stdout.read = AsyncMock(return_value=b"")
    process = MagicMock()
    process.stdin = stdin
    process.stdout = stdout
    process.wait = AsyncMock()
    process.returncode = returncode
    return process


class TestWebmInitSegment:
    """The header replayed into a restarted decoder."""

    def test_header_is_bytes_before_first_cluster(self):
        assert _webm_init_segment(WEBM_FIRST_CHUNK) == WEBM_HEADER

    def test_chunk_without_cluster_is_all_header(self):
        assert _webm_init_segment(WEBM_HEADER) == WEBM_HEADER

    def test_non_webm_has_no_header(self):
        assert _webm_init_segment(b"OggS" + b"opus-pages") is None


class TestAudioDecoderRestart:
    """ffmpeg crash recovery: restart with the WebM header replayed, capped per stream."""

    @pytest.mark.asyncio
    async def test_crash_restarts_and_replays_header_before_next_chunk(self):
        decoder = AudioDecoder()
        first = _mock_ffmpeg()
        decoder._process = first
        await decoder.feed_chunk(WEBM_FIRST_CHUNK)

        first.returncode = 1  # ffmpeg died
        second = _mock_ffmpeg()
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=second)):
            await decoder.feed_chunk(b"later-cluster")

        writes = [c.args[0] for c in second.stdin.write.call_args_list]
        assert writes == [WEBM_HEADER, b"later-cluster"]
        assert decoder.is_dead is False
        assert decoder._restarts == 1
        await decoder.stop_streaming()

    @pytest.mark.asyncio
    async def test_marks_dead_after_max_restarts(self):
        decoder = AudioDecoder()
        decoder._process = _mock_ffmpeg()
        await decoder.feed_chunk(WEBM_FIRST_CHUNK)
        decoder._restarts = AudioDecoder.MAX_RESTARTS
        decoder._process.returncode = 1

        with patch("asyncio.create_subprocess_exec", new=AsyncMock()) as spawn:
            pcm = await decoder.feed_chunk(b"later-cluster")

        assert pcm == b""
        assert decoder.is_dead is True
        spawn.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_webm_stream_is_not_restarted(self):
        decoder = AudioDecoder()
        decoder._process = _mock_ffmpeg()
        await decoder.feed_chunk(b"OggS" + b"opus-pages")
        decoder._process.returncode = 1

        with patch("asyncio.create_subprocess_exec", new=AsyncMock()) as spawn:
            await decoder.feed_chunk(b"more-pages")

        assert decoder.is_dead is True
        spawn.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_stream_forgets_header_and_restart_count(self):
        decoder = AudioDecoder()
        decoder._process = _mock_ffmpeg()
        await decoder.feed_chunk(WEBM_FIRST_CHUNK)
        decoder._restarts = 2

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=_mock_ffmpeg())):
            await decoder.start_streaming()

        assert decoder._init_segment is None
        assert decoder._restarts == 0
        await decoder.stop_streaming()


class TestAudioDecoderReset:
    """Test AudioDecoder reset."""

    def test_reset_clears_pcm_chunks(self):
        """reset should clear buffered PCM chunks."""
        decoder = AudioDecoder()
        decoder._pcm_chunks.append(b"data")

        decoder.reset()

        assert len(decoder._pcm_chunks) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_vad_pre_speech_padding(self):
        """Speech onset should include pre-speech padding."""
        config = VADConfig(calibration_duration_ms=0)
        vad = StreamingVADBuffer(config=config)
        vad._vad_ready = False

        # Feed quiet frames to fill pre-speech ring
        quiet = np.zeros(512, dtype=np.int16).tobytes()
        for _ in range(5):
            await vad.feed_pcm(quiet)

        # Now loud frame
        loud = np.random.randint(-5000, 5000, size=512, dtype=np.int16).tobytes()
        await vad.feed_pcm(loud)

        # Speech buffer should include padding
        assert len(vad._speech_buffer) > len(loud)

    @pytest.mark.asyncio
    async def test_vad_stats_tracking(self):
        """VAD should track emitted segments and total duration."""
        segment_received = []

        async def callback(pcm, duration, audio_rms, vad_avg_prob):
            segment_received.append((pcm, duration, audio_rms, vad_avg_prob))

        config = VADConfig(silence_duration_ms=60, frame_ms=30, min_speech_ms=100, calibration_duration_ms=0)
        vad = StreamingVADBuffer(config=config, on_segment=callback)
        vad._vad_ready = False

        # Emit a segment
        loud = np.random.randint(-5000, 5000, size=512, dtype=np.int16).tobytes()
        for _ in range(20):
            await vad.feed_pcm(loud)

        quiet = np.zeros(512, dtype=np.int16).tobytes()
        for _ in range(5):
            await vad.feed_pcm(quiet)

        assert vad.segments_emitted == 1
        assert vad.total_speech_seconds > 0
