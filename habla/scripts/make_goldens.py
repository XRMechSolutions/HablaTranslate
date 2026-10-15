"""Run test clips through the server pipeline and write golden JSON files.

The Android port is checked against these: segment boundaries from the VAD,
merged utterances, partial snapshot sizes, final transcripts and detected
language per utterance, and (when an
OpenAI key is configured) the translation result. Each step calls the same
server code the live pipeline uses, so the goldens describe what the server
actually does, not a re-implementation of it.

Clips: shared/testdata/clips/*.wav (16 kHz mono PCM16; public-domain sources
listed in shared/testdata/clips/sources.json).
Output: shared/testdata/goldens/<clip>.json

Usage (from habla/):
    .venv/Scripts/python.exe scripts/make_goldens.py            # all clips
    .venv/Scripts/python.exe scripts/make_goldens.py --no-asr   # VAD only (no GPU)
    .venv/Scripts/python.exe scripts/make_goldens.py --clip mixed_es_en_es
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import shared_assets  # noqa: E402
from server.config import AppConfig, load_config  # noqa: E402
from server.pipeline.audio_processor import AudioProcessor, prepare_partial_pcm  # noqa: E402
from server.pipeline.vad_buffer import StreamingVADBuffer, VADConfig  # noqa: E402

SAMPLE_RATE = 16000
FEED_CHUNK_SAMPLES = 1600  # 100 ms, like a client sending small chunks

logger = logging.getLogger("habla.goldens")


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path.name}: need 16 kHz mono PCM16")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()


def write_wav(path: str, pcm: bytes) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def ffmpeg_version() -> str:
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout
        return out.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def run_vad(audio: np.ndarray, cfg: AppConfig) -> tuple[list[dict], list[dict], float | None]:
    """Feed the clip through StreamingVADBuffer exactly as routes/websocket.py configures it."""
    a = cfg.audio
    vad = StreamingVADBuffer(config=VADConfig(
        sample_rate=a.sample_rate,
        silence_duration_ms=a.vad_silence_ms,
        min_speech_ms=a.min_speech_ms,
        max_segment_seconds=a.max_segment_seconds,
        speech_threshold=cfg.asr.vad_threshold,
        frame_ms=a.vad_frame_ms,
        pre_speech_padding_ms=a.pre_speech_padding_ms,
        calibration_duration_ms=a.calibration_duration_ms,
    ))
    await vad.initialize()
    if not vad._vad_ready:
        raise RuntimeError("Silero VAD failed to load; goldens need the real model")

    fed = 0
    segments: list[dict] = []
    partials: list[dict] = []
    noise_floor: list[float] = []

    async def on_segment(pcm: bytes, duration: float, rms: float, prob: float):
        n = len(pcm) // 2
        # Segments are emitted after the frame that completed them; frames are
        # whole 512-sample blocks, so the end is the last processed frame boundary.
        end = (fed // 512) * 512
        segments.append({
            "index": len(segments), "start_sample": end - n, "end_sample": end,
            "samples": n, "duration_sec": round(float(duration), 6), "rms": round(float(rms), 3),
            "vad_avg_prob": round(float(prob), 6), "pcm": pcm,
        })

    async def on_partial(pcm: bytes, duration: float):
        partials.append({
            "segment_index": len(segments), "at_sample": (fed // 512) * 512,
            "samples": len(pcm) // 2, "duration_sec": round(float(duration), 6), "pcm": pcm,
        })

    async def on_calibrated(rms: float):
        noise_floor.append(rms)

    vad.on_segment = on_segment
    vad.on_partial_audio = on_partial
    vad.on_calibration_complete = on_calibrated

    data = audio.tobytes()
    step = FEED_CHUNK_SAMPLES * 2
    for i in range(0, len(data), step):
        chunk = data[i:i + step]
        fed += len(chunk) // 2
        await vad.feed_pcm(chunk)
    await vad.flush()
    return segments, partials, (noise_floor[0] if noise_floor else None)


def merge_segments(segments: list[dict]) -> list[dict]:
    """Group VAD segments the way routes/websocket.py _queue_segment does.

    The server waits merge.gap_seconds (wall clock) after each VAD segment
    before sending the pending audio on; a segment that arrives sooner joins
    it. Replayed in real time, a segment "arrives" at its end_sample, so two
    segments merge when their end samples are less than the gap apart. Pending
    audio is flushed at once when it reaches merge.max_pending_seconds.
    """
    merge = shared_assets.asr_profile()["merge"]
    gap = int(merge["gap_seconds"] * SAMPLE_RATE)
    utterances: list[dict] = []
    pending: list[dict] = []

    def flush():
        if not pending:
            return
        weight = sum(p["duration_sec"] for p in pending)
        utterances.append({
            "index": len(utterances),
            "segment_indices": [p["index"] for p in pending],
            "samples": sum(p["samples"] for p in pending),
            "duration_sec": round(weight, 6),
            "rms": round(sum(p["rms"] * p["duration_sec"] for p in pending) / weight, 3) if weight else 0.0,
            "vad_avg_prob": round(sum(p["vad_avg_prob"] * p["duration_sec"] for p in pending) / weight, 6)
            if weight else 0.0,
            "pcm": b"".join(p["pcm"] for p in pending),
        })
        pending.clear()

    for seg in segments:
        if pending and seg["end_sample"] - pending[-1]["end_sample"] >= gap:
            flush()
        pending.append(seg)
        if sum(p["duration_sec"] for p in pending) >= merge["max_pending_seconds"]:
            flush()
    flush()
    return utterances


def load_asr(cfg: AppConfig) -> AudioProcessor:
    import torch
    import whisperx

    ap = AudioProcessor(cfg, source_language_getter=lambda: cfg.asr.language)
    torch.backends.cudnn.enabled = False  # same workaround as orchestrator.startup
    ap._whisperx_model = whisperx.load_model(
        cfg.asr.model_size, device=cfg.asr.device, compute_type=cfg.asr.compute_type,
        language=None if cfg.asr.auto_language else cfg.asr.language,
        vad_options={"vad_onset": 0.01, "vad_offset": 0.01},
        asr_options={"beam_size": cfg.asr.beam_size, "best_of": cfg.asr.beam_size},
    )
    torch.backends.cudnn.enabled = True
    return ap


async def run_asr(ap: AudioProcessor, segments: list[dict], partials: list[dict]) -> None:
    """Final pass per merged utterance (normalize with ffmpeg, detect, transcribe) and quick pass per partial."""
    tmp = Path(tempfile.mkdtemp(prefix="habla_goldens_"))
    try:
        for seg in segments:
            raw = str(tmp / f"seg_{seg['index']:03d}.wav")
            write_wav(raw, seg["pcm"])
            norm = await ap._normalize_wav(raw)
            text, _, detected, reason = await asyncio.to_thread(ap._run_asr_and_diarize, norm)
            seg.update({"transcript": text, "detected_language": detected, "asr_status": reason,
                        "voted_language": ap._last_detected_language})
        for i, part in enumerate(partials):
            path = str(tmp / f"partial_{i:03d}.wav")
            write_wav(path, prepare_partial_pcm(part["pcm"]))
            text, lang = await asyncio.to_thread(ap._run_quick_asr, path)
            part.update({"transcript": text, "language": lang})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def run_translation(cfg: AppConfig, segments: list[dict]) -> str | None:
    """Translate each final with the server Translator; returns the model used, or None if skipped."""
    if not cfg.translator.openai_api_key:
        return None
    from server.config import DEFAULT_TARGET_LANG
    from server.pipeline.translator import Translator

    cfg.translator.provider = "openai"
    translator = Translator(cfg.translator)
    context: list[dict] = []
    try:
        for seg in segments:
            if seg.get("asr_status") != "ok":
                continue
            src = seg["detected_language"] or "es"
            tgt = DEFAULT_TARGET_LANG if src != DEFAULT_TARGET_LANG else "es"
            result = await translator.translate(
                seg["transcript"], "Speaker A", f"{src}_to_{tgt}", "conversation", context,
            )
            seg["translation"] = result.model_dump()
            context.append({"speaker_label": "Speaker A", "corrected": result.corrected,
                            "translated": result.translated})
    finally:
        await translator.close()
    return cfg.translator.openai_model


def strip_pcm(items: list[dict]) -> list[dict]:
    return [{k: v for k, v in it.items() if k != "pcm"} for it in items]


async def process_clip(path: Path, cfg: AppConfig, ap: AudioProcessor | None, out_dir: Path) -> dict:
    audio = read_wav(path)
    if ap is not None:
        ap._last_detected_language = None
        ap._language_votes.clear()
    segments, partials, noise_floor = await run_vad(audio, cfg)
    utterances = merge_segments(segments)
    if ap is not None:
        await run_asr(ap, utterances, partials)
    translate_model = await run_translation(cfg, utterances) if ap is not None else None

    profile = shared_assets.asr_profile()
    golden = {
        "clip": path.name,
        "clip_samples": int(audio.shape[0]),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "server_commit": git_commit(),
        "ffmpeg": ffmpeg_version(),
        "asr": None if ap is None else {
            "engine": "whisperx", "model": cfg.asr.model_size, "compute_type": cfg.asr.compute_type,
            "beam_size": cfg.asr.beam_size,
        },
        "translate_model": translate_model,
        "profile_version": profile["version"],
        "feed_chunk_samples": FEED_CHUNK_SAMPLES,
        "noise_floor_rms": None if noise_floor is None else round(float(noise_floor), 3),
        "segments": strip_pcm(segments),
        "utterances": strip_pcm(utterances),
        "partials": strip_pcm(partials),
    }
    out = out_dir / f"{path.stem}.json"
    out.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return golden


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clip", help="clip stem to process (default: all)")
    parser.add_argument("--no-asr", action="store_true", help="VAD only; skip WhisperX and translation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    testdata = shared_assets.shared_dir() / "testdata"
    clips = sorted((testdata / "clips").glob("*.wav"))
    if args.clip:
        clips = [c for c in clips if c.stem == args.clip]
    if not clips:
        print("no clips found", file=sys.stderr)
        return 1
    out_dir = testdata / "goldens"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    ap = None if args.no_asr else load_asr(cfg)
    for clip in clips:
        g = await process_clip(clip, cfg, ap, out_dir)
        langs = [u.get("detected_language") for u in g["utterances"]]
        print(f"{clip.stem}: {len(g['segments'])} VAD segments -> {len(g['utterances'])} utterances, "
              f"{len(g['partials'])} partials, languages={langs}")
    if not cfg.translator.openai_api_key and not args.no_asr:
        print("OPENAI_API_KEY not set: translations skipped (translate_model = null)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
