"""Speaker tracking and labeling service."""

from server.models.schemas import SpeakerProfile


SPEAKER_COLORS = [
    "#3B82F6",  # blue
    "#10B981",  # green
    "#F59E0B",  # amber
    "#EF4444",  # red
    "#8B5CF6",  # violet
    "#EC4899",  # pink
    "#06B6D4",  # cyan
    "#F97316",  # orange
]


class SpeakerTracker:
    """Tracks speaker identities within a session."""

    def __init__(self):
        self.speakers: dict[str, SpeakerProfile] = {}

    def get_or_create(self, speaker_id: str) -> SpeakerProfile:
        """Get existing speaker or create a new one."""
        if speaker_id not in self.speakers:
            idx = len(self.speakers)
            self.speakers[speaker_id] = SpeakerProfile(
                id=speaker_id,
                label=f"Speaker {chr(65 + idx)}",  # A, B, C...
                color=SPEAKER_COLORS[idx % len(SPEAKER_COLORS)],
                utterance_count=0,
            )
        return self.speakers[speaker_id]

    def record_utterance(self, speaker_id: str) -> SpeakerProfile:
        """Record that a speaker said something, return their profile."""
        profile = self.get_or_create(speaker_id)
        profile.utterance_count += 1
        return profile

    def rename(self, speaker_id: str, name: str) -> SpeakerProfile | None:
        """User assigns a custom name to a speaker."""
        if speaker_id in self.speakers:
            self.speakers[speaker_id].custom_name = name
            return self.speakers[speaker_id]
        return None

    def set_role_hint(self, speaker_id: str, role: str) -> SpeakerProfile | None:
        """Set a role hint from LLM suggestion."""
        if speaker_id in self.speakers:
            self.speakers[speaker_id].role_hint = role
            return self.speakers[speaker_id]
        return None

    def get_all(self) -> list[SpeakerProfile]:
        """Get all tracked speakers."""
        return list(self.speakers.values())

    def get_display_name(self, speaker_id: str) -> str:
        """Get the best display name for a speaker."""
        profile = self.get_or_create(speaker_id)
        return profile.custom_name or profile.label

    def get_speaker_list_summary(self) -> str:
        """Get a text summary of speakers for LLM context."""
        parts = []
        for sp in self.speakers.values():
            name = sp.custom_name or sp.label
            role = f" ({sp.role_hint})" if sp.role_hint else ""
            parts.append(f"{name}{role}: {sp.utterance_count} utterances")
        return "; ".join(parts) if parts else "(no speakers yet)"
