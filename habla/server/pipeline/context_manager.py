"""Context management — translation state, conversation context, speaker tracking, idiom scanning."""

import logging
from collections import deque
from datetime import datetime
from pathlib import Path

from server import shared_assets
from server.config import AppConfig
from server.models.schemas import FlaggedPhrase, SpeakerProfile, TranslationResult
from server.pipeline.translator import Translator
from server.services.idiom_scanner import IdiomScanner, IdiomMatch
from server.services.speaker_tracker import SpeakerTracker

logger = logging.getLogger("habla.pipeline")


class ContextManager:
    """Owns translation/context state and pure translation logic.

    Responsibilities:
    - Translator (LLM client) lifecycle
    - SpeakerTracker (in-memory speaker identity)
    - IdiomScanner (regex pattern DB)
    - Conversation context: direction, mode, topic_summary, recent_exchanges (maxlen=10)
    - Session identity (session_id — set by orchestrator on DB create/close)
    - Audio clip saving for playback
    - Async topic summary updates (fire-and-forget from orchestrator)

    Does NOT own DB access — DB operations remain in PipelineOrchestrator so that
    existing test patches on `server.pipeline.orchestrator.get_db` continue to work.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.translator = Translator(config.translator)
        self.speaker_tracker = SpeakerTracker()
        self.idiom_scanner = IdiomScanner()

        self.direction: str = config.session.direction
        self.mode: str = config.session.mode
        self.topic_summary: str = ""
        self.recent_exchanges: deque[dict] = deque(
            maxlen=shared_assets.asr_profile()["context"]["recent_exchanges"]
        )
        self.session_id: int | None = None

    def _merge_idioms(
        self, pattern_matches: list[IdiomMatch], llm_phrases: list[FlaggedPhrase]
    ) -> list[FlaggedPhrase]:
        """Merge pattern DB matches with LLM-detected phrases, deduplicating.

        Pattern DB entries take priority over LLM detections for the same phrase.
        Returns combined list with pattern DB results first.
        """
        result = []
        seen: set[str] = set()

        # Pattern DB matches first (more curated)
        for m in pattern_matches:
            key = m.canonical.lower()
            if key not in seen:
                seen.add(key)
                result.append(FlaggedPhrase(
                    phrase=m.canonical,
                    literal=m.literal,
                    meaning=m.meaning,
                    type="idiom",
                    region=m.region,
                    source="pattern_db",
                    save_worthy=True,
                    span_start=m.match_start,
                    span_end=m.match_end,
                ))

        # LLM-detected (novel ones only)
        for fp in llm_phrases:
            key = fp.phrase.lower()
            if key not in seen:
                seen.add(key)
                fp.source = "llm"
                result.append(fp)

        return result

    async def _update_topic(self, result: TranslationResult, speaker_name: str) -> None:
        """Update topic summary asynchronously (called fire-and-forget from orchestrator)."""
        try:
            new_summary = await self.translator.update_topic_summary(
                previous_summary=self.topic_summary,
                latest_source=result.corrected,
                latest_translation=result.translated,
                speaker_label=speaker_name,
            )
            if new_summary:
                self.topic_summary = new_summary
        except Exception as e:
            logger.debug(f"Topic update failed (non-critical): {e}")

    def _save_audio_clip(self, wav_path: str) -> str | None:
        """Save a copy of the WAV as an audio clip for later playback.

        Returns the relative path to the saved clip, or None if disabled/failed.
        """
        if not self.config.session.save_audio_clips or not self.session_id:
            return None
        try:
            import shutil
            clips_dir = Path("data/audio/clips") / str(self.session_id)
            clips_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            clip_path = clips_dir / f"clip_{ts}.wav"
            shutil.copy2(wav_path, clip_path)
            logger.debug(f"Saved audio clip: {clip_path}")
            return str(clip_path)
        except Exception as e:
            logger.warning(f"Failed to save audio clip: {e}")
            return None
