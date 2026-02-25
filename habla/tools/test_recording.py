"""Integration test harness for recording playback quality assessment.

Runs real recordings through the full pipeline (ASR + diarization + translation)
and produces detailed comparison reports. Requires GPU models loaded — run with
the server stopped so models don't compete for VRAM.

Usage:
    cd habla

    # List available recordings
    python -m tools.test_recording --list

    # Test a specific recording (all segments)
    python -m tools.test_recording --recording-id <id>

    # Test first N segments only (quick smoke test)
    python -m tools.test_recording --recording-id <id> --limit 3

    # Test with ground truth comparison
    python -m tools.test_recording --recording-id <id> --compare-gt

    # Test all recordings
    python -m tools.test_recording --all --limit 5

    # Verbose mode (show each segment's full output)
    python -m tools.test_recording --recording-id <id> -v

Output:
    - Console: readable summary with per-segment results and timing
    - JSON: detailed results written to data/audio/recordings/<id>/test_results.json
    - Logs: full pipeline logs to test_recording.log
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure habla/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.config import AppConfig
from server.db.database import init_db
from server.pipeline.orchestrator import PipelineOrchestrator
from server.models.schemas import WSTranslation, WSPartialTranscript, WSSpeakersUpdate

# --- Logging ---

LOG_FILE = Path("test_recording.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("test_recording")

# Quiet noisy libraries
for lib in ("httpx", "httpcore", "asyncio", "urllib3"):
    logging.getLogger(lib).setLevel(logging.WARNING)


RECORDINGS_DIR = Path("data/audio/recordings")


# --- Result Collector ---

class ResultCollector:
    """Captures all pipeline callback output for a test run."""

    def __init__(self):
        self.translations: list[dict] = []
        self.partials: list[dict] = []
        self.final_transcripts: list[dict] = []
        self.speakers: list[dict] = []
        self.errors: list[str] = []
        self._current_segment_start: float = 0

    def clear_segment(self):
        """Reset per-segment timing."""
        self._current_segment_start = time.monotonic()

    @property
    def last_translation(self) -> Optional[dict]:
        return self.translations[-1] if self.translations else None

    async def on_translation(self, msg: WSTranslation):
        data = msg.model_dump()
        data["_elapsed_ms"] = round((time.monotonic() - self._current_segment_start) * 1000)
        self.translations.append(data)

    async def on_partial(self, msg: WSPartialTranscript):
        data = msg.model_dump()
        data["_elapsed_ms"] = round((time.monotonic() - self._current_segment_start) * 1000)
        self.partials.append(data)

    async def on_speakers(self, msg: WSSpeakersUpdate):
        self.speakers.append(msg.model_dump())

    async def on_final_transcript(self, msg: dict):
        data = dict(msg)
        data["_elapsed_ms"] = round((time.monotonic() - self._current_segment_start) * 1000)
        self.final_transcripts.append(data)

    async def on_error(self, msg: str):
        self.errors.append(msg)
        logger.error(f"Pipeline error: {msg}")


# --- Word-level diff (reused from client for GT comparison) ---

def word_diff(reference: str, actual: str) -> list[dict]:
    """LCS-based word-level diff. Returns list of {type, word} dicts."""
    ref_words = reference.strip().split()
    act_words = actual.strip().split()

    if not ref_words and not act_words:
        return []

    m, n = len(ref_words), len(act_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1].lower() == act_words[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack
    parts = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1].lower() == act_words[j - 1].lower():
            parts.append({"type": "eq", "word": act_words[j - 1]})
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            parts.append({"type": "add", "word": act_words[j - 1]})
            j -= 1
        else:
            parts.append({"type": "del", "word": ref_words[i - 1]})
            i -= 1

    parts.reverse()
    return parts


def format_diff(parts: list[dict]) -> str:
    """Format word diff for terminal display."""
    result = []
    for p in parts:
        if p["type"] == "eq":
            result.append(p["word"])
        elif p["type"] == "del":
            result.append(f"[-{p['word']}-]")
        elif p["type"] == "add":
            result.append(f"[+{p['word']}+]")
    return " ".join(result)


def diff_accuracy(parts: list[dict]) -> float:
    """Calculate accuracy from diff parts (proportion of matching words)."""
    if not parts:
        return 1.0
    eq_count = sum(1 for p in parts if p["type"] == "eq")
    total = len(parts)
    return eq_count / total if total > 0 else 1.0


# --- Test Runner ---

async def test_recording(
    recording_id: str,
    pipeline: PipelineOrchestrator,
    collector: ResultCollector,
    limit: Optional[int] = None,
    verbose: bool = False,
    compare_gt: bool = False,
) -> dict:
    """Run a single recording through the pipeline and collect results.

    Returns a detailed results dict.
    """
    session_dir = RECORDINGS_DIR / recording_id
    metadata_path = session_dir / "metadata.json"

    if not session_dir.is_dir():
        logger.error(f"Recording not found: {recording_id}")
        return {"error": f"Recording not found: {recording_id}"}

    # Load metadata
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    # Load ground truth if requested
    ground_truth = None
    gt_path = session_dir / "ground_truth.json"
    if compare_gt and gt_path.exists():
        with open(gt_path) as f:
            ground_truth = json.load(f)
        logger.info(f"Loaded ground truth: {len(ground_truth.get('segments', []))} segments")
    elif compare_gt:
        logger.warning(f"No ground_truth.json found for {recording_id}")

    # Gather segment WAVs
    segments = sorted(session_dir.glob("segment_*.wav"))
    if limit:
        segments = segments[:limit]

    total = len(segments)
    if total == 0:
        logger.error(f"No segment WAVs in {recording_id}")
        return {"error": "No segments"}

    logger.info(f"Testing {recording_id}: {total} segments")
    print(f"\n{'='*70}")
    print(f"Recording: {recording_id}")
    print(f"Segments:  {total}")
    if metadata.get("started_at"):
        print(f"Date:      {metadata['started_at'][:19]}")
    print(f"{'='*70}")

    # Create a fresh session
    await pipeline.create_session()

    # Per-segment results
    segment_results = []
    run_start = time.monotonic()

    for i, wav_path in enumerate(segments):
        seg_num = i + 1
        seg_meta = None
        if metadata.get("segments") and i < len(metadata["segments"]):
            seg_meta = metadata["segments"][i]

        duration = 0.0
        if seg_meta:
            duration = seg_meta.get("duration_seconds", 0)
        else:
            # Estimate from file size: WAV header=44, 16-bit mono 16kHz = 32000 bytes/sec
            file_size = wav_path.stat().st_size
            duration = max(0, (file_size - 44)) / 32000.0

        # Track which translation index we're at before processing
        trans_before = len(collector.translations)
        final_before = len(collector.final_transcripts)
        partial_before = len(collector.partials)

        collector.clear_segment()
        seg_start = time.monotonic()

        try:
            exchange = await pipeline.process_wav(str(wav_path))
        except Exception as e:
            logger.error(f"Segment {seg_num} failed: {e}")
            segment_results.append({
                "segment_id": seg_num,
                "error": str(e),
                "duration_seconds": duration,
            })
            continue

        seg_elapsed = time.monotonic() - seg_start

        # Collect what came through callbacks for this segment
        new_translations = collector.translations[trans_before:]
        new_finals = collector.final_transcripts[final_before:]
        new_partials = collector.partials[partial_before:]

        # Translation fires asynchronously after ASR — wait for it
        wait_deadline = time.monotonic() + 15.0  # 15s max wait
        while len(collector.translations) <= trans_before:
            if time.monotonic() > wait_deadline:
                logger.warning(f"Segment {seg_num}: translation timed out after 15s")
                break
            await asyncio.sleep(0.2)
        new_translations = collector.translations[trans_before:]

        # Build segment result
        result = {
            "segment_id": seg_num,
            "wav_file": wav_path.name,
            "audio_duration_seconds": round(duration, 2),
            "processing_time_seconds": round(seg_elapsed, 2),
            "rtf": round(seg_elapsed / duration, 2) if duration > 0 else None,
            "partial_count": len(new_partials),
        }

        if new_finals:
            result["transcript"] = new_finals[-1].get("text", "")
            result["transcript_time_ms"] = new_finals[-1].get("_elapsed_ms", 0)

        if new_translations:
            t = new_translations[-1]
            result["source"] = t.get("source", "")
            result["corrected"] = t.get("corrected", "")
            result["translated"] = t.get("translated", "")
            result["confidence"] = t.get("confidence", 0)
            result["speaker"] = t.get("speaker", {}).get("label", "")
            result["idiom_count"] = len(t.get("idioms", []))
            result["idioms"] = [
                {"phrase": i["phrase"], "meaning": i["meaning"]}
                for i in t.get("idioms", [])
            ]
            result["translation_time_ms"] = t.get("_elapsed_ms", 0)

        # Ground truth comparison
        if ground_truth and ground_truth.get("segments"):
            gt_segs = ground_truth["segments"]
            if i < len(gt_segs):
                gt = gt_segs[i]
                result["gt_transcript"] = gt.get("transcript", "")
                result["gt_translation"] = gt.get("translation", "")

                if result.get("transcript") and gt.get("transcript"):
                    diff = word_diff(gt["transcript"], result["transcript"])
                    result["transcript_diff"] = format_diff(diff)
                    result["transcript_accuracy"] = round(diff_accuracy(diff), 3)

                if result.get("translated") and gt.get("translation"):
                    diff = word_diff(gt["translation"], result["translated"])
                    result["translation_diff"] = format_diff(diff)
                    result["translation_accuracy"] = round(diff_accuracy(diff), 3)

                if gt.get("asr_corrections"):
                    result["gt_asr_notes"] = gt["asr_corrections"]

        segment_results.append(result)

        # Console output
        rtf_str = f"RTF={result['rtf']:.1f}" if result.get("rtf") else "RTF=?"
        conf_str = f"{result.get('confidence', 0)*100:.0f}%" if result.get("confidence") else "?%"

        print(f"\n  Segment {seg_num}/{total} ({duration:.1f}s audio, {seg_elapsed:.1f}s proc, {rtf_str}, {conf_str})")

        if result.get("transcript"):
            print(f"    SRC: {result['transcript'][:100]}")
        if result.get("translated"):
            print(f"    TGT: {result['translated'][:100]}")
        if result.get("idiom_count", 0) > 0:
            for idiom in result["idioms"]:
                print(f"    IDIOM: {idiom['phrase']} = {idiom['meaning']}")

        if verbose:
            if result.get("corrected") and result["corrected"] != result.get("source"):
                print(f"    CORRECTED: {result['corrected'][:100]}")
            if result.get("speaker"):
                print(f"    SPEAKER: {result['speaker']}")
            if result.get("partial_count"):
                print(f"    PARTIALS: {result['partial_count']}")

        # Ground truth comparison output
        if result.get("transcript_diff"):
            acc = result.get("transcript_accuracy", 0) * 100
            print(f"    GT TRANSCRIPT ({acc:.0f}%): {result['transcript_diff'][:120]}")
        if result.get("translation_diff"):
            acc = result.get("translation_accuracy", 0) * 100
            print(f"    GT TRANSLATION ({acc:.0f}%): {result['translation_diff'][:120]}")
        if result.get("gt_asr_notes"):
            print(f"    GT NOTE: {result['gt_asr_notes'][:100]}")

    total_elapsed = time.monotonic() - run_start
    total_audio = sum(r.get("audio_duration_seconds", 0) for r in segment_results)
    avg_confidence = 0
    conf_values = [r.get("confidence", 0) for r in segment_results if r.get("confidence")]
    if conf_values:
        avg_confidence = sum(conf_values) / len(conf_values)

    # Summary
    summary = {
        "recording_id": recording_id,
        "segments_tested": len(segment_results),
        "segments_with_errors": sum(1 for r in segment_results if "error" in r),
        "total_audio_seconds": round(total_audio, 1),
        "total_processing_seconds": round(total_elapsed, 1),
        "overall_rtf": round(total_elapsed / total_audio, 2) if total_audio > 0 else None,
        "avg_confidence": round(avg_confidence, 3),
        "total_idioms_detected": sum(r.get("idiom_count", 0) for r in segment_results),
        "pipeline_errors": list(collector.errors),
    }

    # Ground truth summary
    if compare_gt and ground_truth:
        trans_accs = [r["transcript_accuracy"] for r in segment_results if "transcript_accuracy" in r]
        transl_accs = [r["translation_accuracy"] for r in segment_results if "translation_accuracy" in r]
        if trans_accs:
            summary["avg_transcript_accuracy"] = round(sum(trans_accs) / len(trans_accs), 3)
        if transl_accs:
            summary["avg_translation_accuracy"] = round(sum(transl_accs) / len(transl_accs), 3)

    print(f"\n{'='*70}")
    print(f"SUMMARY: {recording_id}")
    print(f"  Segments:    {summary['segments_tested']} tested, {summary['segments_with_errors']} errors")
    print(f"  Audio:       {summary['total_audio_seconds']}s total")
    print(f"  Processing:  {summary['total_processing_seconds']}s total (RTF={summary.get('overall_rtf', '?')})")
    print(f"  Confidence:  {summary['avg_confidence']*100:.1f}% avg")
    print(f"  Idioms:      {summary['total_idioms_detected']} detected")
    if summary.get("avg_transcript_accuracy") is not None:
        print(f"  GT Transcript Accuracy: {summary['avg_transcript_accuracy']*100:.1f}%")
    if summary.get("avg_translation_accuracy") is not None:
        print(f"  GT Translation Accuracy: {summary['avg_translation_accuracy']*100:.1f}%")
    if summary["pipeline_errors"]:
        print(f"  Errors: {summary['pipeline_errors']}")
    print(f"{'='*70}")

    # Full results
    full_results = {
        "tested_at": datetime.now().isoformat(),
        "config": {
            "asr_model": pipeline.config.asr.model_size,
            "asr_device": pipeline.config.asr.device,
            "llm_provider": pipeline.translator.config.provider,
            "llm_model": pipeline.translator.config.model,
            "direction": pipeline.direction,
            "mode": pipeline.mode,
        },
        "summary": summary,
        "segments": segment_results,
    }

    # Save results
    results_path = session_dir / "test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {results_path}")
    print(f"\nResults: {results_path}")

    await pipeline.close_session()

    return full_results


# --- Pipeline Bootstrap ---

async def create_pipeline() -> PipelineOrchestrator:
    """Create and start a real pipeline for testing."""
    config = AppConfig()

    # Initialize the database so sessions/exchanges can be saved
    await init_db(config.db_path)

    pipeline = PipelineOrchestrator(config)

    logger.info("Starting pipeline (loading models)...")
    print("Loading models (WhisperX + Pyannote + LLM)...")
    start = time.monotonic()
    await pipeline.startup()
    elapsed = time.monotonic() - start
    print(f"Pipeline ready in {elapsed:.1f}s")

    return pipeline


def list_recordings():
    """Print all available recordings."""
    if not RECORDINGS_DIR.exists():
        print("No recordings directory found.")
        return

    print(f"\nRecordings in {RECORDINGS_DIR}:\n")
    print(f"  {'ID':<50} {'Segs':>5} {'Duration':>10} {'GT':>4} {'Tested':>7}")
    print(f"  {'-'*50} {'-'*5} {'-'*10} {'-'*4} {'-'*7}")

    for d in sorted(RECORDINGS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue

        seg_count = len(list(d.glob("segment_*.wav")))
        has_gt = (d / "ground_truth.json").exists()
        has_results = (d / "test_results.json").exists()

        total_dur = 0
        meta_path = d / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                total_dur = sum(s.get("duration_seconds", 0) for s in meta.get("segments", []))
            except Exception:
                pass

        dur_str = f"{total_dur:.0f}s" if total_dur else "?"
        gt_str = "yes" if has_gt else ""
        tested_str = "yes" if has_results else ""

        print(f"  {d.name:<50} {seg_count:>5} {dur_str:>10} {gt_str:>4} {tested_str:>7}")

    print()


# --- Main ---

async def main():
    parser = argparse.ArgumentParser(
        description="Integration test harness for recording playback quality"
    )
    parser.add_argument("--list", action="store_true", help="List available recordings")
    parser.add_argument("--recording-id", type=str, help="Test a specific recording")
    parser.add_argument("--all", action="store_true", help="Test all recordings")
    parser.add_argument("--limit", type=int, default=None, help="Max segments to test per recording")
    parser.add_argument("--compare-gt", action="store_true", help="Compare against ground truth")
    parser.add_argument("--language", type=str, default=None,
                        help="Force ASR language (e.g. 'es', 'en'). Disables auto-detection.")
    parser.add_argument("--direction", type=str, default=None,
                        help="Override translation direction ('es_to_en' or 'en_to_es')")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.list:
        list_recordings()
        return

    if not args.recording_id and not args.all:
        parser.print_help()
        return

    # Boot the real pipeline
    pipeline = await create_pipeline()

    # Apply overrides
    if args.language:
        pipeline.config.asr.auto_language = False
        pipeline.config.asr.language = args.language
        pipeline._last_detected_language = args.language
        logger.info(f"Forced ASR language: {args.language} (auto-detect OFF)")
        print(f"Language: {args.language} (forced, auto-detect OFF)")
    if args.direction:
        pipeline.set_direction(args.direction)
        logger.info(f"Direction override: {args.direction}")
        print(f"Direction: {args.direction}")
    collector = ResultCollector()

    pipeline.set_callbacks(
        on_translation=collector.on_translation,
        on_partial=collector.on_partial,
        on_speakers=collector.on_speakers,
        on_final_transcript=collector.on_final_transcript,
        on_error=collector.on_error,
    )

    try:
        if args.recording_id:
            await test_recording(
                args.recording_id, pipeline, collector,
                limit=args.limit, verbose=args.verbose,
                compare_gt=args.compare_gt,
            )
        elif args.all:
            if not RECORDINGS_DIR.exists():
                print("No recordings directory found.")
                return

            for d in sorted(RECORDINGS_DIR.iterdir()):
                if not d.is_dir():
                    continue
                if not list(d.glob("segment_*.wav")):
                    continue

                # Fresh collector per recording
                collector = ResultCollector()
                pipeline.set_callbacks(
                    on_translation=collector.on_translation,
                    on_partial=collector.on_partial,
                    on_speakers=collector.on_speakers,
                    on_final_transcript=collector.on_final_transcript,
                    on_error=collector.on_error,
                )

                await test_recording(
                    d.name, pipeline, collector,
                    limit=args.limit, verbose=args.verbose,
                    compare_gt=args.compare_gt,
                )
    finally:
        # Wait for any trailing in-flight translations before shutting down
        # (translations fire async after ASR completes)
        logger.info("Waiting for in-flight translations to complete...")
        drain_deadline = time.monotonic() + 20.0
        last_count = len(collector.translations)
        stable_ticks = 0
        while time.monotonic() < drain_deadline:
            await asyncio.sleep(0.5)
            current_count = len(collector.translations)
            if current_count == last_count:
                stable_ticks += 1
                if stable_ticks >= 4:  # 2s of no new translations
                    break
            else:
                stable_ticks = 0
                last_count = current_count

        await pipeline.shutdown()

    print(f"\nDetailed logs: {LOG_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
