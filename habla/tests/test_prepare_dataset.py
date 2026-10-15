"""scripts/prepare_dataset.py reads the metadata.json that AudioRecorder writes."""

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "prepare_dataset", Path(__file__).resolve().parent.parent / "scripts" / "prepare_dataset.py"
)
prepare_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_dataset)


def test_segment_metadata_list_format(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({
        "segments": [
            {"segment_id": 1, "filename": "segment_001.wav", "audio_rms": 900.0},
            {"segment_id": 2, "filename": "segment_002.wav", "audio_rms": 450.0},
        ]
    }), encoding="utf-8")
    meta = prepare_dataset._load_segment_metadata(tmp_path)
    assert set(meta) == {1, 2}
    assert meta[2]["audio_rms"] == 450.0


def test_segment_metadata_legacy_dict_format(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"segments": {"3": {"audio_rms": 1.0}}}),
                                            encoding="utf-8")
    assert prepare_dataset._load_segment_metadata(tmp_path) == {3: {"audio_rms": 1.0}}


def test_default_min_confidence_keeps_negative_logprob(tmp_path, monkeypatch):
    """confidence is Whisper avg_logprob (<= 0); the default must not drop it."""
    session = tmp_path / "1_20260101_000000"
    session.mkdir()
    (session / "segment_001.wav").write_bytes(b"RIFF")
    (session / "ground_truth.json").write_text(json.dumps({"segments": [
        {"segment_id": 1, "filename": "segment_001.wav", "transcript": "hola qué tal amigo",
         "confidence": -0.8, "duration_seconds": 2.0},
    ]}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["prepare_dataset.py"])
    args = prepare_dataset.parse_args()
    samples, stats = prepare_dataset.collect_samples(
        tmp_path, args.min_duration, args.max_duration, args.min_confidence,
    )
    assert stats["skipped_confidence"] == 0
