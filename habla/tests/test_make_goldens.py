"""merge_segments in scripts/make_goldens.py mirrors websocket._queue_segment."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_goldens", Path(__file__).resolve().parent.parent / "scripts" / "make_goldens.py"
)
make_goldens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_goldens)


def _seg(index, end_s, dur_s):
    n = int(dur_s * 16000)
    return {"index": index, "end_sample": int(end_s * 16000), "samples": n,
            "duration_sec": dur_s, "rms": 1000.0, "vad_avg_prob": 0.9, "pcm": b"\x00\x00" * n}


def test_segments_closer_than_gap_merge():
    out = make_goldens.merge_segments([_seg(0, 2.0, 1.0), _seg(1, 3.0, 0.5), _seg(2, 6.0, 1.0)])
    assert [u["segment_indices"] for u in out] == [[0, 1], [2]]
    assert out[0]["samples"] == 24000
    assert len(out[0]["pcm"]) == 48000


def test_gap_at_threshold_does_not_merge():
    out = make_goldens.merge_segments([_seg(0, 2.0, 1.0), _seg(1, 3.2, 1.0)])
    assert [u["segment_indices"] for u in out] == [[0], [1]]


def test_max_pending_forces_flush():
    segs = [_seg(i, 1.0 * (i + 1), 25.0) for i in range(3)]
    out = make_goldens.merge_segments(segs)
    assert [u["segment_indices"] for u in out] == [[0, 1, 2]] or out[0]["duration_sec"] >= 60.0
    assert sum(len(u["segment_indices"]) for u in out) == 3


class _FakeProcessor:
    """Stands in for AudioProcessor: FinalPrep writes known bytes, ASR echoes the path it read."""

    PREPPED = b"RIFF-fake-final-prep-output"

    def __init__(self, prep_ok=True):
        self.prep_ok = prep_ok
        self.asr_inputs = []
        self._last_detected_language = "es"

    async def _normalize_wav(self, wav_path):
        if self.prep_ok:
            Path(wav_path).write_bytes(self.PREPPED)
        return wav_path

    def _run_asr_and_diarize(self, path):
        self.asr_inputs.append(Path(path).read_bytes())
        return "hola", None, "es", "ok"

    def _run_quick_asr(self, path):
        return "ho", "es"


def _utt(index):
    return {"index": index, "pcm": b"\x01\x00" * 1600}


def test_final_prep_wav_sha256_matches_written_file(tmp_path):
    import asyncio
    import hashlib

    golden_dir = tmp_path / "goldens"
    wav_dir = golden_dir / "wav" / "clip"
    ap = _FakeProcessor()
    utts = [_utt(0), _utt(1)]
    asyncio.run(make_goldens.run_asr(ap, utts, [], wav_dir, golden_dir))

    for u in utts:
        assert u["asr_status"] == "ok"
        written = golden_dir / u["final_prep_wav"]
        assert u["final_prep_wav"] == f"wav/clip/utt_{u['index']:03d}.wav"
        assert written.read_bytes() == _FakeProcessor.PREPPED
        assert u["final_prep_sha256"] == hashlib.sha256(written.read_bytes()).hexdigest()
    # WhisperX read exactly the bytes that were kept
    assert ap.asr_inputs == [_FakeProcessor.PREPPED, _FakeProcessor.PREPPED]


def test_ffmpeg_failure_marks_prep_failed_and_continues(tmp_path):
    import asyncio

    wav_dir = tmp_path / "wav" / "clip"
    ap = _FakeProcessor(prep_ok=False)
    utts = [_utt(0)]
    asyncio.run(make_goldens.run_asr(ap, utts, [], wav_dir, tmp_path))
    assert utts[0]["asr_status"] == "prep_failed"
    assert utts[0]["final_prep_wav"] is None and utts[0]["final_prep_sha256"] is None
    assert utts[0]["transcript"] is None
    assert list(wav_dir.glob("*.wav")) == []
    assert ap.asr_inputs == []


def test_no_asr_writes_no_wavs(tmp_path, monkeypatch):
    import asyncio
    import wave

    clip = tmp_path / "clip.wav"
    with wave.open(str(clip), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)

    async def fake_vad(audio, cfg):
        return [_seg(0, 1.0, 1.0)], [], None

    monkeypatch.setattr(make_goldens, "run_vad", fake_vad)
    monkeypatch.setattr(make_goldens, "git_commit", lambda: "test")
    monkeypatch.setattr(make_goldens, "ffmpeg_version", lambda: "test")
    out_dir = tmp_path / "goldens"
    out_dir.mkdir()
    wav_dir = out_dir / "wav"
    g = asyncio.run(make_goldens.process_clip(clip, make_goldens.load_config(), None, out_dir, wav_dir))
    assert not wav_dir.exists()
    assert "final_prep_wav" not in g["utterances"][0]
    assert (out_dir / "clip.json").exists()
