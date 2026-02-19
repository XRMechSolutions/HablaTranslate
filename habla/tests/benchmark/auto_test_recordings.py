"""Auto-test script for running benchmarks on newly recorded audio.

This script monitors the recordings directory and automatically runs
pipeline benchmarks on new recordings, generating performance reports.

Usage:
    # Watch for new recordings and auto-test
    python tests/benchmark/auto_test_recordings.py --watch

    # Process all existing recordings
    python tests/benchmark/auto_test_recordings.py --all

    # Process specific recording session
    python tests/benchmark/auto_test_recordings.py --session 12345_20260216_143022

    # Promote good samples to test suite
    python tests/benchmark/auto_test_recordings.py --promote --min-confidence 0.9
"""

import argparse
import asyncio
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from server.config import TranslatorConfig
from server.pipeline.translator import Translator
from server.services.idiom_scanner import IdiomScanner, create_starter_idioms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("auto_test")


RECORDINGS_DIR = Path("data/audio/recordings")
SAMPLES_DIR = Path("tests/benchmark/audio_samples")
RESULTS_DIR = Path("tests/benchmark/results/auto_test")
PROCESSED_MARKER = ".tested"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class RecordingBenchmark:
    """Benchmark a single recorded session."""

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.session_id = session_dir.name
        self.metadata_file = session_dir / "metadata.json"
        self.results = {
            "session_id": self.session_id,
            "tested_at": datetime.now().isoformat(),
            "segments": [],
            "summary": {}
        }

        # Load session metadata
        self.metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    async def run(self):
        """Run benchmarks on all segments in this recording."""
        logger.info(f"Testing recording: {self.session_id}")

        # Find all segment WAV files
        segments = sorted(self.session_dir.glob("segment_*.wav"))

        if not segments:
            logger.warning(f"No segments found in {self.session_id}")
            return

        # Initialize translator for re-translation tests
        translator_config = TranslatorConfig(
            provider="ollama",
            ollama_url="http://localhost:11434",
            ollama_model="qwen3:4b",
        )
        translator = Translator(translator_config)

        # Initialize idiom scanner
        idiom_scanner = IdiomScanner()
        idiom_scanner.load_from_db(create_starter_idioms())

        try:
            for segment_file in segments:
                segment_result = await self._test_segment(
                    segment_file,
                    translator,
                    idiom_scanner
                )
                self.results["segments"].append(segment_result)

            # Calculate summary statistics
            self._calculate_summary()

            # Save results
            self._save_results()

            # Mark as processed
            self._mark_processed()

            logger.info(f"Completed: {self.session_id} - {len(segments)} segments tested")

        finally:
            await translator.close()

    async def _test_segment(
        self,
        segment_file: Path,
        translator: Translator,
        idiom_scanner: IdiomScanner
    ) -> dict:
        """Test a single segment."""
        segment_name = segment_file.stem
        logger.info(f"  Testing: {segment_name}")

        # Get original metadata for this segment
        segment_id = int(segment_name.split("_")[1])
        original_meta = None
        for seg in self.metadata.get("segments", []):
            if seg.get("segment_id") == segment_id:
                original_meta = seg
                break

        result = {
            "segment_id": segment_id,
            "filename": segment_file.name,
            "original": original_meta or {},
            "retest": {}
        }

        if not original_meta:
            logger.warning(f"    No metadata found for {segment_name}")
            return result

        original_transcript = original_meta.get("raw_transcript", "")
        original_translation = original_meta.get("translation", "")

        # Idiom scan performance
        start = time.perf_counter()
        idiom_matches = idiom_scanner.scan(original_transcript)
        idiom_time_ms = (time.perf_counter() - start) * 1000

        # Re-translate to check consistency
        start = time.perf_counter()
        try:
            retest_result = await translator.translate(
                transcript=original_transcript,
                speaker_label=original_meta.get("speaker", "SPEAKER_00"),
                direction="es_to_en",
                mode="conversation",
                context_exchanges=[],
            )
            translation_time_ms = (time.perf_counter() - start) * 1000

            result["retest"] = {
                "translation": retest_result.translated,
                "confidence": retest_result.confidence,
                "idioms_detected": len(retest_result.flagged_phrases),
                "translation_time_ms": round(translation_time_ms, 2),
                "matches_original": retest_result.translated == original_translation,
            }

        except Exception as e:
            logger.error(f"    Translation failed: {e}")
            result["retest"]["error"] = str(e)

        result["idiom_scan_ms"] = round(idiom_time_ms, 2)
        result["pattern_matches"] = len(idiom_matches)

        # Quality checks
        result["quality"] = self._assess_quality(result)

        return result

    def _assess_quality(self, segment_result: dict) -> dict:
        """Assess segment quality for potential promotion to test samples."""
        quality = {
            "confidence_ok": False,
            "has_idioms": False,
            "translation_stable": False,
            "good_for_testing": False,
            "issues": []
        }

        original = segment_result.get("original", {})
        retest = segment_result.get("retest", {})

        # Check confidence
        orig_conf = original.get("confidence", 0)
        retest_conf = retest.get("confidence", 0)

        if orig_conf >= 0.85:
            quality["confidence_ok"] = True
        else:
            quality["issues"].append(f"Low confidence: {orig_conf:.2f}")

        # Check for idioms
        if original.get("idioms_detected", 0) > 0:
            quality["has_idioms"] = True

        # Check translation stability (should be consistent)
        if retest.get("matches_original", False):
            quality["translation_stable"] = True
        elif "error" not in retest:
            quality["issues"].append("Translation differs on retest")

        # Overall assessment
        quality["good_for_testing"] = (
            quality["confidence_ok"] and
            quality["translation_stable"] and
            len(quality["issues"]) == 0
        )

        return quality

    def _calculate_summary(self):
        """Calculate summary statistics across all segments."""
        segments = self.results["segments"]

        if not segments:
            return

        # Confidence stats
        confidences = [
            s.get("original", {}).get("confidence", 0)
            for s in segments
        ]

        # Translation time stats
        trans_times = [
            s.get("retest", {}).get("translation_time_ms", 0)
            for s in segments
            if "translation_time_ms" in s.get("retest", {})
        ]

        # Quality stats
        good_segments = [
            s for s in segments
            if s.get("quality", {}).get("good_for_testing", False)
        ]

        idiom_segments = [
            s for s in segments
            if s.get("original", {}).get("idioms_detected", 0) > 0
        ]

        self.results["summary"] = {
            "total_segments": len(segments),
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0,
            "min_confidence": round(min(confidences), 3) if confidences else 0,
            "max_confidence": round(max(confidences), 3) if confidences else 0,
            "avg_translation_time_ms": round(sum(trans_times) / len(trans_times), 2) if trans_times else 0,
            "segments_with_idioms": len(idiom_segments),
            "good_for_testing": len(good_segments),
            "promotion_candidates": [
                {
                    "segment_id": s["segment_id"],
                    "filename": s["filename"],
                    "confidence": s.get("original", {}).get("confidence", 0),
                    "has_idioms": s.get("original", {}).get("idioms_detected", 0) > 0
                }
                for s in good_segments
            ]
        }

    def _save_results(self):
        """Save benchmark results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = RESULTS_DIR / f"{self.session_id}_{timestamp}.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved: {result_file}")

        # Print summary
        summary = self.results["summary"]
        print(f"\n{'='*60}")
        print(f"  Recording: {self.session_id}")
        print(f"{'='*60}")
        print(f"  Segments: {summary.get('total_segments', 0)}")
        print(f"  Avg Confidence: {summary.get('avg_confidence', 0):.3f}")
        print(f"  Avg Translation Time: {summary.get('avg_translation_time_ms', 0):.1f}ms")
        print(f"  Segments with Idioms: {summary.get('segments_with_idioms', 0)}")
        print(f"  Good for Testing: {summary.get('good_for_testing', 0)}")
        print(f"{'='*60}\n")

        if summary.get("promotion_candidates"):
            print("Promotion Candidates:")
            for candidate in summary["promotion_candidates"]:
                print(f"  - {candidate['filename']} (confidence: {candidate['confidence']:.2f}, idioms: {candidate['has_idioms']})")
            print()

    def _mark_processed(self):
        """Mark this recording as processed."""
        marker = self.session_dir / PROCESSED_MARKER
        marker.touch()


async def test_recording(session_dir: Path):
    """Test a single recording session."""
    benchmark = RecordingBenchmark(session_dir)
    await benchmark.run()


async def test_all_recordings():
    """Test all recordings that haven't been processed yet."""
    if not RECORDINGS_DIR.exists():
        logger.warning(f"Recordings directory not found: {RECORDINGS_DIR}")
        return

    sessions = [
        d for d in RECORDINGS_DIR.iterdir()
        if d.is_dir() and not (d / PROCESSED_MARKER).exists()
    ]

    if not sessions:
        logger.info("No new recordings to test")
        return

    logger.info(f"Found {len(sessions)} new recordings to test")

    for session_dir in sessions:
        await test_recording(session_dir)


async def watch_recordings(interval_seconds: int = 60):
    """Watch for new recordings and auto-test them."""
    logger.info(f"Watching {RECORDINGS_DIR} for new recordings (check every {interval_seconds}s)")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            await test_all_recordings()
            await asyncio.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Stopping watch mode")


def promote_segments(min_confidence: float = 0.9, has_idioms: bool = False):
    """Promote good recordings to test samples directory."""
    logger.info(f"Promoting segments (min_confidence={min_confidence}, has_idioms={has_idioms})")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Find all test results
    result_files = sorted(RESULTS_DIR.glob("*.json"))

    promoted_count = 0

    for result_file in result_files:
        with open(result_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        session_id = results.get("session_id", "")
        candidates = results.get("summary", {}).get("promotion_candidates", [])

        for candidate in candidates:
            conf = candidate.get("confidence", 0)
            has_idiom = candidate.get("has_idioms", False)

            # Check criteria
            if conf < min_confidence:
                continue

            if has_idioms and not has_idiom:
                continue

            # Copy segment to samples directory
            segment_file = RECORDINGS_DIR / session_id / candidate["filename"]

            if not segment_file.exists():
                logger.warning(f"Segment not found: {segment_file}")
                continue

            # Generate appropriate name based on characteristics
            if has_idiom:
                prefix = "idioms"
            elif conf >= 0.95:
                prefix = "conversation"
            else:
                prefix = "natural"

            # Find next available number
            existing = list(SAMPLES_DIR.glob(f"{prefix}_es_*.wav"))
            next_num = len(existing) + 1
            dest_name = f"{prefix}_es_{next_num:02d}.wav"
            dest_file = SAMPLES_DIR / dest_name

            shutil.copy2(segment_file, dest_file)
            logger.info(f"Promoted: {candidate['filename']} -> {dest_name}")
            promoted_count += 1

    logger.info(f"Promoted {promoted_count} segments to test samples")


def main():
    parser = argparse.ArgumentParser(description="Auto-test recorded audio")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch for new recordings and auto-test"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all existing recordings"
    )
    parser.add_argument(
        "--session",
        type=str,
        help="Process specific recording session"
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote good segments to test samples"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.9,
        help="Minimum confidence for promotion (default: 0.9)"
    )
    parser.add_argument(
        "--require-idioms",
        action="store_true",
        help="Only promote segments with idioms"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Watch mode check interval in seconds (default: 60)"
    )

    args = parser.parse_args()

    if args.promote:
        promote_segments(
            min_confidence=args.min_confidence,
            has_idioms=args.require_idioms
        )
    elif args.watch:
        asyncio.run(watch_recordings(interval_seconds=args.interval))
    elif args.all:
        asyncio.run(test_all_recordings())
    elif args.session:
        session_dir = RECORDINGS_DIR / args.session
        if not session_dir.exists():
            logger.error(f"Session not found: {session_dir}")
            return
        asyncio.run(test_recording(session_dir))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
