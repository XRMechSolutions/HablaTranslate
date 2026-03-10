#!/usr/bin/env python3
"""
Auto-tune Habla parameters based on recorded audio sessions.

Analyzes saved recordings and metadata to recommend optimal:
- VAD threshold (speech_threshold)
- Pre-speech padding (pre_speech_padding_ms)
- Silence duration (silence_duration_ms)
- AGC enablement
- Per-speaker volume profiles

Usage:
    python scripts/auto_tune_parameters.py
    python scripts/auto_tune_parameters.py --recordings-dir data/audio/recordings
    python scripts/auto_tune_parameters.py --apply  # Auto-apply recommendations
"""

import json
import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def analyze_recordings(recordings_dir: Path) -> Tuple[Dict, Dict]:
    """Analyze saved recordings and compute statistics."""

    results = {
        "confidence_scores": [],
        "durations": [],
        "clipped_onsets": 0,
        "total_segments": 0,
        "speaker_confidences": {},
        "low_confidence_segments": [],
        "session_count": 0,
    }

    if not recordings_dir.exists():
        print(f"Error: Recordings directory not found: {recordings_dir}")
        return results, {}

    # Load all metadata files
    session_dirs = [d for d in recordings_dir.iterdir() if d.is_dir()]
    results["session_count"] = len(session_dirs)

    for session_dir in session_dirs:
        metadata_file = session_dir / "metadata.json"
        if not metadata_file.exists():
            continue

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {metadata_file}: {e}")
            continue

        for segment in metadata.get("segments", []):
            results["total_segments"] += 1

            # Collect confidence scores
            confidence = segment.get("confidence", 1.0)
            results["confidence_scores"].append(confidence)

            # Track duration
            duration = segment.get("duration_seconds", 0)
            results["durations"].append(duration)

            # Track per-speaker confidences
            speaker = segment.get("speaker", "UNKNOWN")
            if speaker not in results["speaker_confidences"]:
                results["speaker_confidences"][speaker] = []
            results["speaker_confidences"][speaker].append(confidence)

            # Detect clipped onsets (common Spanish words missing first letter)
            transcript = segment.get("raw_transcript", "").lower().strip()

            # Common patterns of clipped Spanish words
            clipped_patterns = [
                ("ola", "hola"),        # Missing H
                ("ueno", "bueno"),      # Missing B
                ("racias", "gracias"),  # Missing G
                ("asta", "hasta"),      # Missing H
                ("ace", "hace"),        # Missing H
            ]

            for clipped, correct in clipped_patterns:
                if transcript.startswith(clipped):
                    results["clipped_onsets"] += 1
                    break

            # Track low-confidence segments for review
            if confidence < 0.7:
                results["low_confidence_segments"].append({
                    "session": session_dir.name,
                    "transcript": transcript,
                    "confidence": confidence,
                    "speaker": speaker,
                })

    # Calculate recommendations
    recommendations = compute_recommendations(results)

    return results, recommendations


def analyze_quality_metrics_db(db_path: Path) -> Dict:
    """Query the quality_metrics table for rejection statistics.

    Returns a dict with:
      available           — False if DB missing or query failed
      total               — total segments tracked
      rejected            — asr_rejected + asr_empty combined
      asr_rejected_count  — segments where ASR ran but output was garbage
      asr_empty_count     — segments where WhisperX returned no segments at all
      rejection_rate      — rejected / total
      avg_rms_ok          — avg RMS for ok/low_confidence segments
      avg_rms_rejected    — avg RMS for asr_rejected + asr_empty segments
      short_rejection_rate — rejection rate for segments under 1 second
      clipped_onset_count — segments flagged as clipped-onset
    """
    empty: Dict = {
        "available": False,
        "total": 0,
        "rejected": 0,          # asr_rejected + asr_empty combined
        "asr_rejected_count": 0,  # garbage transcript (noise, digits-only)
        "asr_empty_count": 0,     # zero segments from WhisperX VAD (audio too quiet)
        "rejection_rate": 0.0,
        "avg_rms_ok": None,
        "avg_rms_rejected": None,
        "avg_vad_ok": None,
        "avg_vad_rejected": None,
        "short_rejection_rate": 0.0,
        "clipped_onset_count": 0,
    }

    if not db_path.exists():
        return empty

    try:
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        cur.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status IN ('asr_rejected','asr_empty') THEN 1 ELSE 0 END) as rejected,
                SUM(CASE WHEN status = 'asr_rejected' THEN 1 ELSE 0 END) as asr_rejected_count,
                SUM(CASE WHEN status = 'asr_empty'    THEN 1 ELSE 0 END) as asr_empty_count,
                AVG(CASE WHEN status IN ('ok','low_confidence') THEN audio_rms END) as avg_rms_ok,
                AVG(CASE WHEN status IN ('asr_rejected','asr_empty') THEN audio_rms END) as avg_rms_rejected,
                SUM(CASE WHEN clipped_onset = 1 THEN 1 ELSE 0 END) as clipped_onset_count
            FROM quality_metrics
        """)
        row = dict(cur.fetchone())

        total = row["total"] or 0
        rejected = row["rejected"] or 0
        asr_rejected_count = row["asr_rejected_count"] or 0
        asr_empty_count = row["asr_empty_count"] or 0

        # Short-segment rejection: segments under 1 second
        cur.execute("""
            SELECT
                COUNT(*) as short_total,
                SUM(CASE WHEN status IN ('asr_rejected','asr_empty') THEN 1 ELSE 0 END) as short_rejected
            FROM quality_metrics
            WHERE duration_seconds < 1.0 AND duration_seconds > 0
        """)
        short_row = dict(cur.fetchone())
        short_total = short_row["short_total"] or 0
        short_rejected = short_row["short_rejected"] or 0

        con.close()

        return {
            "available": True,
            "total": total,
            "rejected": rejected,
            "asr_rejected_count": asr_rejected_count,
            "asr_empty_count": asr_empty_count,
            "rejection_rate": rejected / total if total > 0 else 0.0,
            "avg_rms_ok": row["avg_rms_ok"],
            "avg_rms_rejected": row["avg_rms_rejected"],
            "avg_vad_ok": None,   # vad_avg_prob not stored in quality_metrics
            "avg_vad_rejected": None,
            "short_rejection_rate": short_rejected / short_total if short_total > 0 else 0.0,
            "clipped_onset_count": row["clipped_onset_count"] or 0,
        }

    except Exception as e:
        print(f"Warning: Failed to query quality_metrics DB: {e}")
        return empty


def compute_recommendations(results: Dict, db_stats: Dict | None = None) -> Dict:
    """Compute parameter recommendations based on analysis."""

    recommendations = {}
    db = db_stats or {}

    if results["total_segments"] == 0 and not db.get("total"):
        print("No segments found. Record some audio first.")
        return recommendations

    # 1. VAD Threshold Recommendation
    confidence_scores = results["confidence_scores"]
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.9
    std_confidence = np.std(confidence_scores) if len(confidence_scores) > 1 else 0

    if avg_confidence < 0.75:
        recommendations["speech_threshold"] = 0.25
        recommendations["speech_threshold_reason"] = f"Low avg confidence ({avg_confidence:.2f}) indicates quiet speech"
    elif avg_confidence > 0.95 and std_confidence < 0.05:
        recommendations["speech_threshold"] = 0.40
        recommendations["speech_threshold_reason"] = f"High consistent confidence ({avg_confidence:.2f}) allows higher threshold"
    else:
        recommendations["speech_threshold"] = 0.35
        recommendations["speech_threshold_reason"] = "Default threshold is appropriate"

    # 2. Pre-speech Padding Recommendation
    clipped_rate = results["clipped_onsets"] / max(results["total_segments"], 1)

    if clipped_rate > 0.15:
        recommendations["pre_speech_padding_ms"] = 500
        recommendations["pre_speech_padding_reason"] = f"{clipped_rate:.1%} of segments have clipped onsets"
    elif clipped_rate > 0.05:
        recommendations["pre_speech_padding_ms"] = 400
        recommendations["pre_speech_padding_reason"] = f"{clipped_rate:.1%} of segments have clipped onsets"
    else:
        recommendations["pre_speech_padding_ms"] = 300
        recommendations["pre_speech_padding_reason"] = "Default padding is sufficient"

    # 3. Silence Duration Recommendation
    durations = results["durations"]
    avg_duration = np.mean(durations) if durations else 3.0

    if avg_duration < 2.0:
        recommendations["silence_duration_ms"] = 400
        recommendations["silence_duration_reason"] = f"Short avg utterance ({avg_duration:.1f}s) suggests fast speech"
    elif avg_duration > 5.0:
        recommendations["silence_duration_ms"] = 800
        recommendations["silence_duration_reason"] = f"Long avg utterance ({avg_duration:.1f}s) suggests slow/hesitant speech"
    else:
        recommendations["silence_duration_ms"] = 600
        recommendations["silence_duration_reason"] = "Default silence duration is appropriate"

    # 4. AGC Recommendation
    if std_confidence > 0.2:
        recommendations["enable_agc"] = True
        recommendations["agc_reason"] = f"High confidence variance ({std_confidence:.2f}) indicates mixed speaker volumes"
    elif avg_confidence < 0.8:
        recommendations["enable_agc"] = True
        recommendations["agc_reason"] = f"Low avg confidence ({avg_confidence:.2f}) suggests quiet speakers"
    else:
        recommendations["enable_agc"] = False
        recommendations["agc_reason"] = "Consistent loud speech doesn't need AGC"

    # 5. Per-Speaker Recommendations
    for speaker, confidences in results["speaker_confidences"].items():
        avg = np.mean(confidences)
        if avg < 0.7:
            recommendations[f"boost_{speaker}"] = True
            recommendations[f"boost_{speaker}_reason"] = f"Low avg confidence ({avg:.2f})"

    # 6. Model Upgrade Recommendation
    if avg_confidence < 0.75:
        recommendations["whisper_model"] = "medium"
        recommendations["whisper_model_reason"] = f"Low confidence ({avg_confidence:.2f}) suggests model upgrade needed"
    else:
        recommendations["whisper_model"] = "small"
        recommendations["whisper_model_reason"] = "Current model performs well"

    # 7. DB-sourced recommendations (quality_metrics table)
    # Use breakdown counts to give more precise advice:
    #   asr_empty   → WhisperX VAD found no speech → audio too quiet → AGC
    #   asr_rejected → transcript was garbage → noise/background audio → VAD/model
    rejection_rate = db.get("rejection_rate", 0.0)
    avg_rms_rejected = db.get("avg_rms_rejected")
    short_rejection_rate = db.get("short_rejection_rate", 0.0)
    asr_empty_count = db.get("asr_empty_count", 0)
    asr_rejected_count = db.get("asr_rejected_count", 0)
    total = db.get("total", 0)

    if rejection_rate > 0.15:
        # Check if the majority of failures are empty-segment (quiet audio) vs garbage output
        empty_dominates = asr_empty_count > asr_rejected_count
        if avg_rms_rejected is not None and avg_rms_rejected < 500:
            recommendations["agc_db_reason"] = (
                f"DB: {rejection_rate:.1%} rejection rate with low RMS on rejected segments "
                f"({avg_rms_rejected:.0f}) — microphone input is too quiet. Enable AGC."
            )
            recommendations["enable_agc"] = True
            recommendations["agc_reason"] = recommendations["agc_db_reason"]
        elif empty_dominates and total > 0:
            # Mostly no-segment failures at normal RMS: WhisperX VAD too aggressive
            recommendations["agc_db_reason"] = (
                f"DB: {rejection_rate:.1%} rejection rate — {asr_empty_count} of {total} segments "
                f"produced no output from WhisperX VAD (audio present but not recognised as speech). "
                f"Enable AGC or check for background noise masking speech."
            )
            recommendations["enable_agc"] = True
            recommendations["agc_reason"] = recommendations["agc_db_reason"]
        else:
            recommendations["vad_threshold_db_reason"] = (
                f"DB: {rejection_rate:.1%} rejection rate ({asr_rejected_count} garbage transcripts) — "
                f"raise speech_threshold to filter more borderline audio before ASR, "
                f"or upgrade to a larger Whisper model for better noise robustness."
            )

    if short_rejection_rate > 0.3:
        recommendations["min_speech_ms_db_reason"] = (
            f"DB: {short_rejection_rate:.1%} of short segments (<1s) are rejected — "
            f"raise min_speech_duration_ms to skip very short noise bursts."
        )

    return recommendations


def print_report(results: Dict, recommendations: Dict, db_stats: Dict | None = None):
    """Print analysis report with recommendations."""

    print("\n" + "=" * 70)
    print("  HABLA AUTO-TUNING ANALYSIS")
    print("=" * 70)

    print(f"\nDataset Statistics:")
    print(f"  Sessions analyzed: {results['session_count']}")
    print(f"  Total segments: {results['total_segments']}")

    if results["total_segments"] == 0:
        print("\n  No recordings found. Enable recording and use Habla to collect data:")
        print("    export SAVE_AUDIO_RECORDINGS=1")
        print("    cd habla && uvicorn server.main:app --host 0.0.0.0 --port 8002")
        return

    confidence_scores = results["confidence_scores"]
    print(f"  Avg confidence: {np.mean(confidence_scores):.2f}")
    print(f"  Confidence std dev: {np.std(confidence_scores):.2f}")
    print(f"  Clipped onsets: {results['clipped_onsets']} ({results['clipped_onsets']/results['total_segments']:.1%})")

    durations = results["durations"]
    print(f"  Avg segment duration: {np.mean(durations):.1f}s")

    print(f"\nSpeaker Statistics:")
    for speaker, confidences in results["speaker_confidences"].items():
        avg = np.mean(confidences)
        count = len(confidences)
        print(f"  {speaker}: {count} segments, avg confidence {avg:.2f}")

    print("\n" + "-" * 70)
    print("  RECOMMENDED SETTINGS")
    print("-" * 70)

    # Group recommendations by category
    vad_params = {}
    agc_params = {}
    model_params = {}
    speaker_params = {}

    for key, value in recommendations.items():
        if key.endswith("_reason"):
            continue

        if "speech_threshold" in key or "padding" in key or "silence" in key:
            vad_params[key] = value
        elif "agc" in key:
            agc_params[key] = value
        elif "model" in key:
            model_params[key] = value
        elif "boost_" in key:
            speaker_params[key] = value

    if vad_params:
        print("\nVAD Parameters (habla/server/pipeline/vad_buffer.py):")
        for key, value in vad_params.items():
            reason = recommendations.get(f"{key}_reason", "")
            print(f"  {key}: {value}")
            if reason:
                print(f"    → {reason}")

    if agc_params:
        print("\nAudio Compression (client settings):")
        for key, value in agc_params.items():
            reason = recommendations.get(f"{key}_reason", "")
            print(f"  {key}: {value}")
            if reason:
                print(f"    → {reason}")

    if model_params:
        print("\nModel Settings (environment variable):")
        for key, value in model_params.items():
            reason = recommendations.get(f"{key}_reason", "")
            print(f"  {key}: {value}")
            if reason:
                print(f"    → {reason}")

    if speaker_params:
        print("\nPer-Speaker Adjustments:")
        for key, value in speaker_params.items():
            reason = recommendations.get(f"{key}_reason", "")
            speaker = key.replace("boost_", "")
            print(f"  {speaker} needs volume boost: {value}")
            if reason:
                print(f"    → {reason}")

    # Low confidence segments
    low_conf = results["low_confidence_segments"]
    if low_conf:
        print(f"\nLow Confidence Segments (confidence < 0.7): {len(low_conf)} found")
        print("  Review these for accuracy (top 5):")
        for seg in low_conf[:5]:
            print(f"    {seg['confidence']:.2f}: \"{seg['transcript'][:50]}...\" [{seg['speaker']}]")

    # DB quality_metrics section
    db = db_stats or {}
    if db.get("available"):
        print("\n" + "-" * 70)
        print("  QUALITY METRICS DB (quality_metrics table)")
        print("-" * 70)
        total = db["total"]
        rejected = db["rejected"]
        rejection_rate = db["rejection_rate"]
        print(f"\n  Total segments tracked in DB: {total}")
        print(f"  Total rejected: {rejected} ({rejection_rate:.1%})")
        if db.get("asr_empty_count", 0) or db.get("asr_rejected_count", 0):
            print(f"    asr_empty (no segments from WhisperX VAD): {db.get('asr_empty_count', 0)}")
            print(f"    asr_rejected (garbage transcript):          {db.get('asr_rejected_count', 0)}")
        if db.get("avg_rms_ok") is not None:
            print(f"  Avg RMS (successful): {db['avg_rms_ok']:.0f}")
        if db.get("avg_rms_rejected") is not None:
            print(f"  Avg RMS (rejected):   {db['avg_rms_rejected']:.0f}")
        if db.get("short_rejection_rate", 0) > 0:
            print(f"  Short-segment (<1s) rejection rate: {db['short_rejection_rate']:.1%}")
        if db.get("clipped_onset_count", 0) > 0:
            print(f"  Clipped-onset segments detected: {db['clipped_onset_count']}")

        # Print DB-sourced recommendation reasons
        for key in ("agc_db_reason", "vad_threshold_db_reason", "min_speech_ms_db_reason"):
            if key in recommendations:
                print(f"\n  WARNING: {recommendations[key]}")

    print("\n" + "=" * 70)


def apply_recommendations(recommendations: Dict, config_path: Path):
    """Apply recommendations to config file (interactive)."""

    print("\n" + "=" * 70)
    print("  APPLY RECOMMENDATIONS")
    print("=" * 70)

    print("\nThis will update your configuration files with recommended settings.")
    print("Backup your current config before proceeding.")

    response = input("\nApply recommendations? (yes/no): ").strip().lower()
    if response not in ("yes", "y"):
        print("Cancelled. Manual application instructions:")
        print(f"\n  Edit: {config_path}")
        print("  Update VAD settings in vad_buffer.py VADConfig class")
        print("  Set WHISPER_MODEL environment variable")
        print("  Enable AGC in client settings if recommended")
        return

    print("\nAuto-apply not yet implemented. Manual steps:")
    print("\n1. VAD Settings (habla/server/pipeline/vad_buffer.py):")
    print("   Update VADConfig defaults:")

    if "speech_threshold" in recommendations:
        print(f"     speech_threshold: float = {recommendations['speech_threshold']}")
    if "pre_speech_padding_ms" in recommendations:
        print(f"     pre_speech_padding_ms: int = {recommendations['pre_speech_padding_ms']}")
    if "silence_duration_ms" in recommendations:
        print(f"     silence_duration_ms: int = {recommendations['silence_duration_ms']}")

    print("\n2. Model Upgrade (if recommended):")
    if recommendations.get("whisper_model") == "medium":
        print("     export WHISPER_MODEL=medium")

    print("\n3. AGC Setting (client):")
    if recommendations.get("enable_agc"):
        print("     Enable 'Boost quiet speech' in settings (or enable by default)")

    print("\nRestart server after applying changes.")


def main():
    parser = argparse.ArgumentParser(description="Auto-tune Habla parameters from recordings")
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("data/audio/recordings"),
        help="Path to recordings directory (default: data/audio/recordings)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply recommendations interactively",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output recommendations as JSON",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to habla.db for quality_metrics analysis (default: auto-detect near recordings dir)",
    )

    args = parser.parse_args()

    # Locate DB — try adjacent to recordings dir, then habla/data/habla.db
    db_path = args.db
    if db_path is None:
        candidates = [
            args.recordings_dir.parent.parent / "habla.db",  # data/habla.db
            args.recordings_dir.parent / "habla.db",
            Path("data/habla.db"),
        ]
        for candidate in candidates:
            if candidate.exists():
                db_path = candidate
                break

    # Analyze recordings and DB
    results, _ = analyze_recordings(args.recordings_dir)
    db_stats = analyze_quality_metrics_db(db_path) if db_path else {"available": False}
    recommendations = compute_recommendations(results, db_stats)

    if args.json:
        # Output JSON for programmatic use
        output = {
            "results": results,
            "db_stats": db_stats,
            "recommendations": {k: v for k, v in recommendations.items() if not k.endswith("_reason")},
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable report
        print_report(results, recommendations, db_stats)

        if args.apply and recommendations:
            config_path = Path("habla/server/pipeline/vad_buffer.py")
            apply_recommendations(recommendations, config_path)


if __name__ == "__main__":
    main()
