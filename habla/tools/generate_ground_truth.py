"""Generate ground truth transcripts and translations for recorded audio.

Runs a larger Whisper model (e.g. large-v3) on saved recording segments to
produce high-quality reference transcripts, then uses the configured LLM to
generate ideal translations with full conversational context.

Usage (run from the habla/ directory with the server stopped):
  python -m tools.generate_ground_truth --recording-id <id>
  python -m tools.generate_ground_truth --all
  python -m tools.generate_ground_truth --all --whisper-model large-v3
  python -m tools.generate_ground_truth --list

The server should be stopped during ground truth generation because the
larger Whisper model needs significant VRAM (~3GB for large-v3).
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("habla.ground_truth")


def list_recordings(recordings_dir: Path) -> list[dict]:
    """List available recordings with summary info."""
    if not recordings_dir.exists():
        return []
    results = []
    for d in sorted(recordings_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        gt_path = d / "ground_truth.json"
        segments = list(d.glob("segment_*.wav"))
        info = {
            "id": d.name,
            "segments": len(segments),
            "has_ground_truth": gt_path.exists(),
        }
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                info["started_at"] = meta.get("started_at", "")
                info["total_duration"] = sum(
                    s.get("duration_seconds", 0) for s in meta.get("segments", [])
                )
            except Exception:
                pass
        results.append(info)
    return results


def load_whisper_model(whisper_model: str = "large-v3", device: str = "cuda"):
    """Load WhisperX model once for reuse across recordings."""
    import whisperx

    t0 = time.monotonic()
    model = whisperx.load_model(
        whisper_model,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
    )
    logger.info(f"WhisperX {whisper_model} loaded in {time.monotonic() - t0:.1f}s")
    return model


async def generate_ground_truth(
    recording_dir: Path,
    whisper_model_obj,
    whisper_model_name: str = "large-v3",
    direction: str = "es_to_en",
):
    """Generate ground truth for a single recording directory."""
    segments = sorted(recording_dir.glob("segment_*.wav"))
    if not segments:
        logger.warning(f"No segment WAVs in {recording_dir.name}, skipping")
        return False

    logger.info(f"Processing {recording_dir.name}: {len(segments)} segments")

    model = whisper_model_obj

    # Determine source language from direction
    src_lang = "es" if direction.startswith("es") else "en"

    # Transcribe each segment
    gt_segments = []
    full_transcript_parts = []

    for wav_path in segments:
        seg_id = int(wav_path.stem.split("_")[1])
        logger.info(f"  Transcribing segment {seg_id}: {wav_path.name}")

        t1 = time.monotonic()
        result = model.transcribe(
            str(wav_path),
            batch_size=16,
            language=src_lang,
            task="transcribe",
        )
        elapsed = time.monotonic() - t1

        transcript = " ".join(
            s["text"].strip() for s in result.get("segments", [])
        ).strip()

        if not transcript:
            logger.warning(f"  Segment {seg_id}: empty transcript")
            transcript = "[inaudible]"

        # Get confidence from segments if available
        confidences = [
            s.get("avg_logprob", 0) for s in result.get("segments", [])
            if "avg_logprob" in s
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        gt_segments.append({
            "segment_id": seg_id,
            "filename": wav_path.name,
            "transcript": transcript,
            "confidence": round(avg_confidence, 3),
            "asr_time_seconds": round(elapsed, 2),
        })
        full_transcript_parts.append(f"[{seg_id}] {transcript}")
        logger.info(f"  Segment {seg_id} ({elapsed:.1f}s): {transcript[:80]}")

    full_transcript = "\n".join(full_transcript_parts)
    logger.info(f"Full transcript ({len(gt_segments)} segments):\n{full_transcript}")

    # Now generate ideal translations using the LLM
    logger.info("Generating translations via LLM...")
    translations = await _generate_translations(
        gt_segments, full_transcript, direction
    )

    # Merge translations into ground truth segments
    for gt_seg in gt_segments:
        seg_id = gt_seg["segment_id"]
        if seg_id in translations:
            t = translations[seg_id]
            gt_seg["translation"] = t.get("translation", "")
            gt_seg["asr_corrections"] = t.get("asr_corrections", "")
        else:
            gt_seg["translation"] = ""
            gt_seg["asr_corrections"] = ""

    # Build full translation
    full_translation = "\n".join(
        gt_seg.get("translation", "") for gt_seg in gt_segments
    )

    # Write ground truth file
    ground_truth = {
        "generated_at": datetime.now().isoformat(),
        "whisper_model": whisper_model_name,
        "direction": direction,
        "total_segments": len(gt_segments),
        "segments": gt_segments,
        "full_transcript": full_transcript,
        "full_translation": full_translation,
    }

    gt_path = recording_dir / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    logger.info(f"Ground truth written to {gt_path}")
    return True


async def _generate_translations(
    gt_segments: list[dict],
    full_transcript: str,
    direction: str,
) -> dict:
    """Use the configured LLM to generate ideal translations."""
    # Import server config to get LLM settings
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from server.config import load_config
    from server.pipeline.translator import Translator

    config = load_config()
    translator = Translator(config.translator)

    if config.translator.provider == "lmstudio":
        await translator.auto_detect_model()

    src_lang = "Spanish" if direction.startswith("es") else "English"
    tgt_lang = "English" if direction.startswith("es") else "Spanish"

    # Build prompt with full context
    segments_text = "\n".join(
        f"Segment {s['segment_id']}: {s['transcript']}"
        for s in gt_segments
    )

    system_prompt = (
        f"You are a professional {src_lang}-to-{tgt_lang} translator producing "
        f"reference translations for evaluating a real-time translation system. "
        f"You have the complete conversation transcript for full context."
    )

    user_prompt = (
        f"Here is a complete conversation transcript in {src_lang}, split into "
        f"segments as they were captured by a speech recognition system:\n\n"
        f"{segments_text}\n\n"
        f"For EACH segment, provide:\n"
        f"1. The ideal {tgt_lang} translation\n"
        f"2. Any likely ASR errors (where the speech recognition probably misheard "
        f"something based on conversational context)\n\n"
        f"Return a JSON array where each element has:\n"
        f'  "segment_id": number,\n'
        f'  "translation": string (the ideal {tgt_lang} translation),\n'
        f'  "asr_corrections": string (describe any likely misheard words, or empty string if transcript looks correct)\n\n'
        f"Return ONLY the JSON array, no other text."
    )

    try:
        raw = await translator._call_llm(
            system_prompt, user_prompt,
            max_tokens=4096, json_mode=True, retries=2,
        )

        # Parse the response
        # Try to find JSON array in the response
        raw = raw.strip()
        if raw.startswith("["):
            data = json.loads(raw)
        elif "[" in raw:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            data = json.loads(raw[start:end])
        else:
            logger.warning(f"LLM response is not a JSON array: {raw[:200]}")
            return {}

        result = {}
        for item in data:
            sid = item.get("segment_id")
            if sid is not None:
                result[sid] = {
                    "translation": item.get("translation", ""),
                    "asr_corrections": item.get("asr_corrections", ""),
                }
        logger.info(f"Got translations for {len(result)} segments")
        return result

    except Exception as e:
        logger.error(f"LLM translation failed: {e}")
        return {}
    finally:
        await translator.client.aclose()


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth transcripts and translations for Habla recordings"
    )
    parser.add_argument("--recording-id", help="Process a specific recording by directory name")
    parser.add_argument("--all", action="store_true", help="Process all recordings without ground truth")
    parser.add_argument("--list", action="store_true", help="List available recordings")
    parser.add_argument("--force", action="store_true", help="Regenerate even if ground_truth.json exists")
    parser.add_argument("--whisper-model", default="large-v3", help="WhisperX model size (default: large-v3)")
    parser.add_argument("--direction", default="es_to_en", choices=["es_to_en", "en_to_es"],
                        help="Translation direction (default: es_to_en)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for WhisperX (default: cuda)")
    args = parser.parse_args()

    recordings_dir = Path("data/audio/recordings")

    if args.list:
        recs = list_recordings(recordings_dir)
        if not recs:
            print("No recordings found.")
            return
        print(f"{'ID':<45} {'Segments':>8} {'Duration':>10} {'GT':>4}")
        print("-" * 72)
        for r in recs:
            dur = f"{r.get('total_duration', 0):.0f}s" if r.get("total_duration") else "-"
            gt = "Yes" if r["has_ground_truth"] else "No"
            print(f"{r['id']:<45} {r['segments']:>8} {dur:>10} {gt:>4}")
        return

    if args.recording_id:
        rec_dir = recordings_dir / args.recording_id
        if not rec_dir.is_dir():
            print(f"Recording not found: {args.recording_id}")
            sys.exit(1)
        if (rec_dir / "ground_truth.json").exists() and not args.force:
            print(f"Ground truth already exists for {args.recording_id}. Use --force to regenerate.")
            sys.exit(0)
        model = load_whisper_model(args.whisper_model, args.device)
        asyncio.run(generate_ground_truth(
            rec_dir, model, args.whisper_model, args.direction,
        ))

    elif args.all:
        recs = list_recordings(recordings_dir)
        to_process = [
            r for r in recs
            if (not r["has_ground_truth"] or args.force) and r["segments"] > 0
        ]
        if not to_process:
            print("No recordings need processing.")
            return
        print(f"Processing {len(to_process)} recordings...")
        model = load_whisper_model(args.whisper_model, args.device)
        for r in to_process:
            rec_dir = recordings_dir / r["id"]
            try:
                asyncio.run(generate_ground_truth(
                    rec_dir, model, args.whisper_model, args.direction,
                ))
            except Exception as e:
                logger.error(f"Failed to process {r['id']}: {e}")
                continue
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
