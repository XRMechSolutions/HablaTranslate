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
