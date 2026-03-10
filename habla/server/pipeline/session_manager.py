"""Session lifecycle and DB persistence — extracted from orchestrator."""

import json
import logging
from datetime import datetime, UTC
from pathlib import Path

from server.config import AppConfig
from server.db.database import get_db
from server.models.schemas import Exchange, SpeakerProfile
from server.services.speaker_tracker import SpeakerTracker, SPEAKER_COLORS

logger = logging.getLogger("habla.pipeline")


class SessionManager:
    """Owns all database operations and session state persistence.

    Extracted from PipelineOrchestrator so the orchestrator stays focused on
    queue management and translation coordination.

    Responsibilities:
    - Session create/close/reset (DB rows)
    - Exchange persistence
    - Quality metrics recording
    - Idiom pattern loading from DB
    - Study summary generation (LLM + DB persist)
    - Shutdown state save/restore (JSON file)
    """

    def __init__(self, config: AppConfig, context_manager, metrics: dict):
        """
        Args:
            config: App configuration (db_path, data_dir, asr settings).
            context_manager: ContextManager instance for accessing translator,
                speaker_tracker, idiom_scanner, direction, mode, etc.
            metrics: Shared mutable metrics dict from the orchestrator.
        """
        self.config = config
        self.context_manager = context_manager
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(self) -> int:
        """Create a new session record in the database. Returns session ID."""
        try:
            db = await get_db()
            cursor = await db.execute(
                """INSERT INTO sessions (mode, direction)
                   VALUES (?, ?)""",
                (self.context_manager.mode, self.context_manager.direction),
            )
            await db.commit()
            self.context_manager.session_id = cursor.lastrowid
            self._metrics["sessions_created"] += 1
            logger.info(f"Session {self.context_manager.session_id} created")
            return self.context_manager.session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}", exc_info=True)
            return 0

    async def close_session(self):
        """Close the current session: set end time, persist topic summary and speakers."""
        if not self.context_manager.session_id:
            return
        try:
            db = await get_db()
            costs = self.context_manager.translator.get_session_costs()

            await db.execute(
                """UPDATE sessions SET ended_at = CURRENT_TIMESTAMP,
                   topic_summary = ?, speaker_count = ?,
                   llm_provider = ?, llm_input_tokens = ?,
                   llm_output_tokens = ?, llm_cost_usd = ?
                   WHERE id = ?""",
                (
                    self.context_manager.topic_summary or None,
                    len(self.context_manager.speaker_tracker.speakers),
                    self.context_manager.translator.config.provider,
                    costs["session_input_tokens"],
                    costs["session_output_tokens"],
                    costs["session_cost_usd"],
                    self.context_manager.session_id,
                ),
            )
            for sid, sp in self.context_manager.speaker_tracker.speakers.items():
                await db.execute(
                    """INSERT OR REPLACE INTO speakers
                       (id, session_id, auto_label, custom_name, role_hint, color, utterance_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (sid, self.context_manager.session_id, sp.label, sp.custom_name,
                     sp.role_hint, sp.color, sp.utterance_count),
                )
            await db.commit()
            self._metrics["sessions_closed"] += 1
            logger.info(f"Session {self.context_manager.session_id} closed ({len(self.context_manager.speaker_tracker.speakers)} speakers)")
        except Exception as e:
            logger.error(f"Failed to close session: {e}", exc_info=True)

    async def reset_session(self) -> int:
        """Close current session and start a fresh one. Returns new session ID."""
        await self.close_session()
        self.context_manager.speaker_tracker = SpeakerTracker()
        self.context_manager.recent_exchanges.clear()
        self.context_manager.topic_summary = ""
        new_id = await self.create_session()
        logger.info(f"Session reset -> new session {new_id}")
        return new_id

    # ------------------------------------------------------------------
    # Exchange and quality persistence
    # ------------------------------------------------------------------

    async def save_exchange(self, exchange: Exchange) -> int | None:
        """Persist a completed exchange to the database. Returns the exchange ID."""
        if not self.context_manager.session_id:
            return None
        try:
            db = await get_db()
            correction_json = None
            if exchange.correction_detail:
                correction_json = json.dumps(exchange.correction_detail.model_dump())
            cursor = await db.execute(
                """INSERT INTO exchanges
                   (session_id, speaker_id, direction, raw_transcript,
                    corrected_source, translation, confidence,
                    is_correction, correction_json, processing_ms, audio_path,
                    detected_language)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.context_manager.session_id,
                    exchange.speaker.id,
                    exchange.direction,
                    exchange.raw_transcript,
                    exchange.corrected_source,
                    exchange.translation,
                    exchange.confidence,
                    exchange.is_correction,
                    correction_json,
                    exchange.processing_ms,
                    exchange.audio_path,
                    exchange.detected_language,
                ),
            )
            await db.commit()
            exchange.id = cursor.lastrowid
            return exchange.id
        except Exception as e:
            logger.error(f"Failed to save exchange: {e}", exc_info=True)
            return None

    async def record_quality_metric(
        self,
        status: str,
        confidence: float | None,
        meta: dict | None = None,
        processing_ms: int = 0,
        speaker_id: str | None = None,
        duration_seconds: float = 0.0,
        clipped_onset: bool = False,
    ):
        """Record a quality metric row for auto-tuning analysis."""
        if not self.context_manager.session_id:
            return
        meta = meta or {}
        try:
            db = await get_db()
            await db.execute(
                """INSERT INTO quality_metrics
                   (session_id, segment_id, status, confidence, audio_rms,
                    duration_seconds, speaker_id, processing_time_ms,
                    vad_threshold, model_name, clipped_onset)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.context_manager.session_id,
                    meta.get("segment_id"),
                    status,
                    confidence,
                    meta.get("audio_rms"),
                    duration_seconds,
                    speaker_id,
                    processing_ms,
                    self.config.asr.vad_threshold,
                    self.config.asr.model_size,
                    clipped_onset,
                ),
            )
            await db.commit()
        except Exception as e:
            logger.warning(f"Failed to record quality metric: {e}")

    # ------------------------------------------------------------------
    # Idiom patterns from DB
    # ------------------------------------------------------------------

    async def load_db_idiom_patterns(self):
        """Load user-contributed idiom patterns from the database."""
        try:
            db = await get_db()
            rows = await db.execute_fetchall("SELECT * FROM idiom_patterns")
            if rows:
                self.context_manager.idiom_scanner.load_from_db([dict(r) for r in rows])
        except Exception as e:
            logger.warning(f"Failed to load DB idiom patterns: {e}")

    # ------------------------------------------------------------------
    # Study summary
    # ------------------------------------------------------------------

    async def generate_study_summary(self, session_id: int | None = None) -> str | None:
        """Generate and persist an LLM study summary for a session.

        Called automatically as a background task after close_session(), and on
        demand via the regenerate endpoint. Skips sessions with fewer than 3
        exchanges — not enough material to summarize.

        Args:
            session_id: Override which session to summarize. Defaults to
                        context_manager.session_id (the just-closed session).

        Returns the raw JSON string if successful, None otherwise.
        """
        from server.models.prompts import get_study_summary_prompt

        sid = session_id if session_id is not None else self.context_manager.session_id
        if not sid:
            return None

        try:
            db = await get_db()

            # Fetch exchanges
            rows = await db.execute_fetchall(
                """SELECT raw_transcript, corrected_source, translation,
                          speaker_id, is_correction, correction_json
                   FROM exchanges WHERE session_id = ?
                   ORDER BY timestamp ASC""",
                (sid,),
            )
            if len(rows) < 3:
                logger.debug("Session %s: skipping study summary (%d exchanges)", sid, len(rows))
                return None

            # Fetch topic_summary and mode for the session (needed for on-demand regenerate)
            if session_id is not None:
                meta = await db.execute_fetchall(
                    "SELECT topic_summary, mode FROM sessions WHERE id = ?", (sid,)
                )
                topic_summary = meta[0]["topic_summary"] or "" if meta else ""
                mode = meta[0]["mode"] or "conversation" if meta else "conversation"
            else:
                topic_summary = self.context_manager.topic_summary
                mode = self.config.session.mode

            # Build compact transcript for the prompt
            lines = []
            for r in rows:
                ex = dict(r)
                spk = ex.get("speaker_id") or "Speaker"
                src = ex.get("corrected_source") or ex.get("raw_transcript") or ""
                tgt = ex.get("translation") or ""
                lines.append(f"[{spk}] {src} -> {tgt}")
                if ex.get("is_correction") and ex.get("correction_json"):
                    try:
                        cd = json.loads(ex["correction_json"])
                        lines.append(
                            f"  [CORRECTION] {cd.get('wrong','')} -> "
                            f"{cd.get('right','')}: {cd.get('explanation','')}"
                        )
                    except Exception:
                        pass

            system_prompt, user_prompt = get_study_summary_prompt(
                "\n".join(lines), topic_summary, mode
            )

            raw = await self.context_manager.translator._call_llm(
                system_prompt, user_prompt, retries=1, max_tokens=1024, json_mode=True,
            )
            json.loads(raw)  # validate before persisting

            await db.execute(
                "UPDATE sessions SET study_summary = ? WHERE id = ?", (raw, sid)
            )
            await db.commit()
            logger.info("Study summary generated for session %s", sid)
            return raw

        except Exception as e:
            logger.warning("Study summary failed for session %s: %s", sid, e)
            return None

    # ------------------------------------------------------------------
    # Shutdown state persistence
    # ------------------------------------------------------------------

    async def save_shutdown_state(self):
        """Persist conversation context and metrics so state survives restarts."""
        try:
            state = {
                "shutdown_time": datetime.now(UTC).isoformat(),
                "direction": self.context_manager.direction,
                "mode": self.context_manager.mode,
                "topic_summary": self.context_manager.topic_summary,
                "recent_exchanges": list(self.context_manager.recent_exchanges),
                "speakers": {
                    sid: {
                        "auto_label": sp.label,
                        "custom_name": sp.custom_name,
                        "role_hint": sp.role_hint,
                        "utterance_count": sp.utterance_count,
                    }
                    for sid, sp in self.context_manager.speaker_tracker.speakers.items()
                },
                "translator_metrics": self.context_manager.translator.metrics,
            }
            state_path = self.config.data_dir / "last_session.json"
            state_path.write_text(json.dumps(state, indent=2, default=str))
            logger.info(f"Session state saved to {state_path}")
        except Exception as e:
            logger.warning(f"Failed to save session state: {e}")

    async def restore_shutdown_state(self):
        """Restore conversation context from last shutdown if the file exists."""
        state_path = self.config.data_dir / "last_session.json"
        if not state_path.exists():
            return
        try:
            state = json.loads(state_path.read_text())
            shutdown_time = state.get("shutdown_time", "unknown")

            if state.get("topic_summary"):
                self.context_manager.topic_summary = state["topic_summary"]

            for ex in state.get("recent_exchanges", []):
                self.context_manager.recent_exchanges.append(ex)

            for sid, sp_data in state.get("speakers", {}).items():
                idx = len(self.context_manager.speaker_tracker.speakers)
                self.context_manager.speaker_tracker.speakers[sid] = SpeakerProfile(
                    id=sid,
                    label=sp_data.get("auto_label", f"Speaker {chr(65 + idx)}"),
                    custom_name=sp_data.get("custom_name"),
                    role_hint=sp_data.get("role_hint"),
                    color=SPEAKER_COLORS[idx % len(SPEAKER_COLORS)],
                    utterance_count=sp_data.get("utterance_count", 0),
                )

            logger.info(
                f"Restored session state from {shutdown_time}: "
                f"{len(self.context_manager.recent_exchanges)} exchanges, "
                f"{len(self.context_manager.speaker_tracker.speakers)} speakers, "
                f"topic='{self.context_manager.topic_summary[:60] if self.context_manager.topic_summary else '(none)'}'"
            )
        except Exception as e:
            logger.warning(f"Failed to restore session state: {e}")
