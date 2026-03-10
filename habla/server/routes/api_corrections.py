"""Ground truth correction workflow routes.

Provides endpoints for reviewing and correcting auto-generated transcripts
before using them for Whisper fine-tuning. Corrected data is saved as
corrected_ground_truth.json alongside the original ground_truth.json.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from server.db.database import get_db
import server.routes._state as _state

logger = logging.getLogger("habla.corrections")

corrections_router = APIRouter(prefix="/api", tags=["corrections"])


class CorrectionSegment(BaseModel):
    segment_id: int
    filename: str
    transcript: str
    confidence: float = 0.0
    asr_time_seconds: float = 0.0
    translation: str = ""
    asr_corrections: str = ""
    reviewed: bool = False
    human_corrected: bool = False


class CorrectionPayload(BaseModel):
    segments: list[CorrectionSegment]


def _get_session_dir(recording_id: str) -> Path:
    """Resolve recording ID to directory using PlaybackService path safety."""
    if not _state._playback_service:
        raise HTTPException(503, "Playback service not initialized")
    session_dir = _state._playback_service._safe_session_dir(recording_id)
    if not session_dir:
        raise HTTPException(404, "Recording not found")
    return session_dir


@corrections_router.get("/recordings/{recording_id}/ground-truth")
async def get_ground_truth(recording_id: str):
    """Return segments from corrected_ground_truth.json (preferred) or ground_truth.json.

    Returns 404 if neither file exists.
    """
    session_dir = _get_session_dir(recording_id)

    # Prefer corrected over original
    corrected_path = session_dir / "corrected_ground_truth.json"
    original_path = session_dir / "ground_truth.json"

    gt_path: Optional[Path] = None
    is_corrected = False
    if corrected_path.exists():
        gt_path = corrected_path
        is_corrected = True
    elif original_path.exists():
        gt_path = original_path
    else:
        raise HTTPException(404, "No ground truth file found for this recording")

    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(500, f"Failed to read ground truth: {e}")

    return {
        "recording_id": recording_id,
        "source": "corrected" if is_corrected else "original",
        "data": data,
    }


@corrections_router.put("/recordings/{recording_id}/ground-truth")
async def save_corrected_ground_truth(recording_id: str, payload: CorrectionPayload):
    """Save corrected segments to corrected_ground_truth.json.

    Never overwrites the original ground_truth.json. Uses atomic write
    (temp file + rename) to prevent corruption.
    """
    session_dir = _get_session_dir(recording_id)

    # Read original for metadata fields we want to preserve
    original_path = session_dir / "ground_truth.json"
    original_meta = {}
    if original_path.exists():
        try:
            with open(original_path, "r", encoding="utf-8") as f:
                original_meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    segments_data = [s.model_dump() for s in payload.segments]
    reviewed_count = sum(1 for s in payload.segments if s.reviewed)
    corrected_count = sum(1 for s in payload.segments if s.human_corrected)

    corrected_doc = {
        "generated_at": original_meta.get("generated_at", ""),
        "whisper_model": original_meta.get("whisper_model", ""),
        "direction": original_meta.get("direction", ""),
        "total_segments": len(segments_data),
        "segments": segments_data,
        "corrected_at": datetime.now(timezone.utc).isoformat(),
        "correction_stats": {
            "reviewed": reviewed_count,
            "human_corrected": corrected_count,
            "total": len(segments_data),
        },
    }

    # Atomic write: temp file in same directory, then rename
    corrected_path = session_dir / "corrected_ground_truth.json"
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(session_dir), suffix=".tmp", prefix="corrected_gt_"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(corrected_doc, f, indent=2, ensure_ascii=False)
        # On Windows, target must not exist for os.rename
        if corrected_path.exists():
            corrected_path.unlink()
        os.rename(tmp_path, str(corrected_path))
    except OSError as e:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise HTTPException(500, f"Failed to save corrections: {e}")

    logger.info(
        "Saved corrections for %s: %d/%d reviewed, %d corrected",
        recording_id, reviewed_count, len(segments_data), corrected_count,
    )

    return {
        "status": "saved",
        "recording_id": recording_id,
        "reviewed": reviewed_count,
        "corrected": corrected_count,
        "total": len(segments_data),
    }


@corrections_router.get("/corrections/failed-segments")
async def get_failed_segments(
    limit: int = Query(default=50, ge=1, le=500),
    session_id: Optional[int] = Query(default=None),
):
    """Return segments that failed to produce ASR output, from the quality_metrics table.

    Includes both failure types:
      asr_empty    — WhisperX returned no speech segments (audio too quiet or non-speech)
      asr_rejected — ASR ran but produced garbage output (noise, digits-only, etc.)

    Each item includes audio conditions (RMS, VAD probability, duration) and
    a playback URL for the WAV clip if it was saved to disk. Ordered newest-first.

    Query params:
      limit — max rows returned (default 50, max 500)
      session_id — filter to a single session
    """
    db = await get_db()

    if session_id is not None:
        rows = await db.execute_fetchall(
            """SELECT qm.id, qm.session_id, qm.segment_id, qm.timestamp,
                      qm.audio_rms, qm.duration_seconds, qm.processing_time_ms,
                      qm.vad_threshold, qm.model_name, qm.clipped_onset, qm.status,
                      s.recording_dir
               FROM quality_metrics qm
               LEFT JOIN sessions s ON qm.session_id = s.id
               WHERE qm.status IN ('asr_rejected', 'asr_empty')
                 AND qm.session_id = ?
               ORDER BY qm.timestamp DESC
               LIMIT ?""",
            (session_id, limit),
        )
    else:
        rows = await db.execute_fetchall(
            """SELECT qm.id, qm.session_id, qm.segment_id, qm.timestamp,
                      qm.audio_rms, qm.duration_seconds, qm.processing_time_ms,
                      qm.vad_threshold, qm.model_name, qm.clipped_onset, qm.status,
                      s.recording_dir
               FROM quality_metrics qm
               LEFT JOIN sessions s ON qm.session_id = s.id
               WHERE qm.status IN ('asr_rejected', 'asr_empty')
               ORDER BY qm.timestamp DESC
               LIMIT ?""",
            (limit,),
        )

    recordings_dir = (
        _state._playback_service.recordings_dir
        if _state._playback_service
        else None
    )

    results = []
    for row in rows:
        r = dict(row)
        recording_dir = r.pop("recording_dir", None)
        seg_num = r.get("segment_id")

        # Try to find the WAV clip — recorded segments are named segment_NNN.wav
        audio_url = None
        if recordings_dir and recording_dir and seg_num is not None:
            clip_path = recordings_dir / recording_dir / f"segment_{seg_num:04d}.wav"
            if clip_path.exists():
                # Expose via the existing playback API
                audio_url = f"/api/recordings/{recording_dir}/segments/{seg_num:04d}"

        r["audio_url"] = audio_url
        r["recording_dir"] = recording_dir
        results.append(r)

    return {"failed_segments": results, "total": len(results)}


@corrections_router.get("/corrections/stats")
async def get_correction_stats():
    """Aggregate progress across all recordings with ground truth."""
    if not _state._playback_service:
        raise HTTPException(503, "Playback service not initialized")

    recordings_dir = _state._playback_service.recordings_dir
    if not recordings_dir.exists():
        return {
            "total_recordings": 0,
            "recordings_with_gt": 0,
            "total_segments": 0,
            "reviewed_segments": 0,
            "corrected_segments": 0,
            "total_duration_seconds": 0.0,
        }

    total_recordings = 0
    recordings_with_gt = 0
    total_segments = 0
    reviewed_segments = 0
    corrected_segments = 0
    total_duration = 0.0

    for session_dir in recordings_dir.iterdir():
        if not session_dir.is_dir():
            continue
        total_recordings += 1

        # Check for ground truth (prefer corrected)
        corrected_path = session_dir / "corrected_ground_truth.json"
        original_path = session_dir / "ground_truth.json"

        gt_path = None
        if corrected_path.exists():
            gt_path = corrected_path
        elif original_path.exists():
            gt_path = original_path

        if not gt_path:
            continue

        recordings_with_gt += 1

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        segments = data.get("segments", [])
        total_segments += len(segments)

        for seg in segments:
            if seg.get("reviewed"):
                reviewed_segments += 1
            if seg.get("human_corrected"):
                corrected_segments += 1

            # Estimate duration from WAV file size
            wav_path = session_dir / seg.get("filename", "")
            if wav_path.exists():
                try:
                    size = wav_path.stat().st_size
                    total_duration += max(0, (size - 44)) / 32000.0
                except OSError:
                    pass

    return {
        "total_recordings": total_recordings,
        "recordings_with_gt": recordings_with_gt,
        "total_segments": total_segments,
        "reviewed_segments": reviewed_segments,
        "corrected_segments": corrected_segments,
        "total_duration_seconds": round(total_duration, 1),
    }
