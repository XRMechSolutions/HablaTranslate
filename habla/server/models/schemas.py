"""Data schemas for Habla."""

from pydantic import BaseModel, Field
from datetime import datetime, UTC
from typing import Optional


# --- Speaker ---

class SpeakerProfile(BaseModel):
    id: str  # "SPEAKER_00"
    label: str  # "Speaker A"
    custom_name: Optional[str] = None  # "Profesor García"
    role_hint: Optional[str] = None  # "instructor"
    color: str  # "#3B82F6"
    utterance_count: int = 0


# --- Idiom / Flagged Phrase ---

class FlaggedPhrase(BaseModel):
    phrase: str
    literal: Optional[str] = None
    meaning: str
    type: str = "idiom"  # idiom, slang, false_friend, correction, grammar_note
    region: Optional[str] = "universal"
    source: str = "llm"  # "pattern_db" or "llm"
    save_worthy: bool = True
    span_start: Optional[int] = None
    span_end: Optional[int] = None


class CorrectionDetail(BaseModel):
    wrong: str
    right: str
    explanation: str


# --- Translation Result ---

class TranslationResult(BaseModel):
    corrected: str
    translated: str
    flagged_phrases: list[FlaggedPhrase] = []
    confidence: float = 0.0
    speaker_hint: Optional[str] = None
    is_correction: bool = False
    correction_detail: Optional[CorrectionDetail] = None


# --- Exchange (one utterance through the pipeline) ---

class Exchange(BaseModel):
    id: Optional[int] = None
    session_id: int
    speaker: SpeakerProfile
    direction: str  # "es_to_en" or "en_to_es"
    raw_transcript: str
    corrected_source: Optional[str] = None
    translation: str
    flagged_phrases: list[FlaggedPhrase] = []
    confidence: float = 0.0
    is_correction: bool = False
    correction_detail: Optional[CorrectionDetail] = None
    processing_ms: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Vocab Item ---

class VocabItem(BaseModel):
    id: Optional[int] = None
    exchange_id: Optional[int] = None
    speaker_id: Optional[str] = None
    term: str
    literal: Optional[str] = None
    meaning: str
    category: str = "idiom"
    source_sentence: Optional[str] = None
    region: Optional[str] = "universal"
    ease_factor: float = 2.5
    interval_days: int = 1
    next_review: Optional[datetime] = None
    repetitions: int = 0
    lapse_count: int = 0
    times_encountered: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- WebSocket Messages ---

class WSAudioChunk(BaseModel):
    """Client → Server: audio data (binary frame, not JSON)."""
    pass  # binary frames handled separately


class WSControlMessage(BaseModel):
    """Client → Server: control commands."""
    type: str  # "toggle_direction", "set_mode", "rename_speaker", "config"
    data: dict = {}


class WSPartialTranscript(BaseModel):
    """Server → Client: partial ASR result."""
    type: str = "partial"
    text: str
    speaker_id: Optional[str] = None


class WSTranslation(BaseModel):
    """Server → Client: full translation result."""
    type: str = "translation"
    exchange_id: Optional[int] = None
    speaker: SpeakerProfile
    source: str
    corrected: str
    translated: str
    idioms: list[FlaggedPhrase] = []
    is_correction: bool = False
    correction_detail: Optional[CorrectionDetail] = None
    confidence: float = 0.0
    timestamp: str


class WSSpeakersUpdate(BaseModel):
    """Server → Client: speaker list changed."""
    type: str = "speakers_updated"
    speakers: list[SpeakerProfile] = []


class WSStatus(BaseModel):
    """Server → Client: system status."""
    type: str = "status"
    pipeline_ready: bool = True
    queue_depth: int = 0
    direction: str = "es_to_en"
    mode: str = "conversation"
    speaker_count: int = 0


