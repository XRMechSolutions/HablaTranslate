"""Unit tests for the SpeakerTracker service."""

import pytest
from server.services.speaker_tracker import SpeakerTracker, SPEAKER_COLORS
from server.models.schemas import SpeakerProfile


class TestSpeakerTrackerInit:
    """Test SpeakerTracker initialization."""

    def test_init_empty(self):
        """Tracker should initialize with no speakers."""
        tracker = SpeakerTracker()
        assert len(tracker.speakers) == 0
        assert tracker.get_all() == []


class TestGetOrCreate:
    """Test speaker creation and retrieval."""

    def test_create_first_speaker(self):
        """First speaker should be labeled 'Speaker A'."""
        tracker = SpeakerTracker()
        speaker = tracker.get_or_create("SPEAKER_00")

        assert speaker.id == "SPEAKER_00"
        assert speaker.label == "Speaker A"
        assert speaker.utterance_count == 0
        assert speaker.color in SPEAKER_COLORS

    def test_create_multiple_speakers(self):
        """Multiple speakers should get sequential labels."""
        tracker = SpeakerTracker()
        speaker_a = tracker.get_or_create("SPEAKER_00")
        speaker_b = tracker.get_or_create("SPEAKER_01")
        speaker_c = tracker.get_or_create("SPEAKER_02")

        assert speaker_a.label == "Speaker A"
        assert speaker_b.label == "Speaker B"
        assert speaker_c.label == "Speaker C"

    def test_get_existing_speaker(self):
        """Getting an existing speaker should return the same instance."""
        tracker = SpeakerTracker()
        speaker1 = tracker.get_or_create("SPEAKER_00")
        speaker1.custom_name = "John"

        speaker2 = tracker.get_or_create("SPEAKER_00")
        assert speaker2.custom_name == "John"
        assert speaker1 is speaker2

    def test_speaker_colors_cycle(self):
        """Speaker colors should cycle through the palette."""
        tracker = SpeakerTracker()
        num_colors = len(SPEAKER_COLORS)

        # Create more speakers than colors
        speakers = [tracker.get_or_create(f"SPEAKER_{i:02d}") for i in range(num_colors + 2)]

        # First speaker should match first color
        assert speakers[0].color == SPEAKER_COLORS[0]
        # Nth speaker should match Nth color (mod len)
        assert speakers[num_colors].color == SPEAKER_COLORS[0]
        assert speakers[num_colors + 1].color == SPEAKER_COLORS[1]


class TestRecordUtterance:
    """Test utterance counting."""

    def test_record_utterance_new_speaker(self):
        """Recording utterance for new speaker should create them."""
        tracker = SpeakerTracker()
        speaker = tracker.record_utterance("SPEAKER_00")

        assert speaker.id == "SPEAKER_00"
        assert speaker.utterance_count == 1

    def test_record_multiple_utterances(self):
        """Multiple utterances should increment count."""
        tracker = SpeakerTracker()
        tracker.record_utterance("SPEAKER_00")
        tracker.record_utterance("SPEAKER_00")
        speaker = tracker.record_utterance("SPEAKER_00")

        assert speaker.utterance_count == 3

    def test_record_different_speakers(self):
        """Different speakers should have independent counts."""
        tracker = SpeakerTracker()
        tracker.record_utterance("SPEAKER_00")
        tracker.record_utterance("SPEAKER_00")
        tracker.record_utterance("SPEAKER_01")

        speaker_a = tracker.get_or_create("SPEAKER_00")
        speaker_b = tracker.get_or_create("SPEAKER_01")

        assert speaker_a.utterance_count == 2
        assert speaker_b.utterance_count == 1


class TestRenameSpeaker:
    """Test custom speaker naming."""

    def test_rename_existing_speaker(self):
        """Renaming should update custom_name."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")

        updated = tracker.rename("SPEAKER_00", "Alice")
        assert updated is not None
        assert updated.custom_name == "Alice"
        assert updated.label == "Speaker A"  # Label unchanged

    def test_rename_nonexistent_speaker(self):
        """Renaming nonexistent speaker should return None."""
        tracker = SpeakerTracker()
        result = tracker.rename("SPEAKER_99", "Nobody")
        assert result is None

    def test_rename_updates_display_name(self):
        """Display name should use custom name after rename."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")
        tracker.rename("SPEAKER_00", "Bob")

        display_name = tracker.get_display_name("SPEAKER_00")
        assert display_name == "Bob"

    def test_rename_empty_string(self):
        """Renaming to empty string should still work."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")
        tracker.rename("SPEAKER_00", "")

        speaker = tracker.get_or_create("SPEAKER_00")
        assert speaker.custom_name == ""


class TestRoleHint:
    """Test LLM-suggested role hints."""

    def test_set_role_hint(self):
        """Setting role hint should update speaker profile."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")

        updated = tracker.set_role_hint("SPEAKER_00", "teacher")
        assert updated is not None
        assert updated.role_hint == "teacher"

    def test_set_role_hint_nonexistent(self):
        """Setting role hint for nonexistent speaker should return None."""
        tracker = SpeakerTracker()
        result = tracker.set_role_hint("SPEAKER_99", "student")
        assert result is None

    def test_role_hint_in_summary(self):
        """Role hint should appear in speaker list summary."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")
        tracker.set_role_hint("SPEAKER_00", "teacher")
        tracker.record_utterance("SPEAKER_00")

        summary = tracker.get_speaker_list_summary()
        assert "teacher" in summary
        assert "Speaker A" in summary


class TestGetDisplayName:
    """Test display name resolution."""

    def test_display_name_default(self):
        """Default display name should be the label."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")

        name = tracker.get_display_name("SPEAKER_00")
        assert name == "Speaker A"

    def test_display_name_with_custom(self):
        """Custom name should take priority over label."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")
        tracker.rename("SPEAKER_00", "Charlie")

        name = tracker.get_display_name("SPEAKER_00")
        assert name == "Charlie"

    def test_display_name_creates_speaker(self):
        """Getting display name for nonexistent speaker should create them."""
        tracker = SpeakerTracker()
        name = tracker.get_display_name("SPEAKER_00")

        assert name == "Speaker A"
        assert "SPEAKER_00" in tracker.speakers


class TestGetAll:
    """Test retrieving all speakers."""

    def test_get_all_empty(self):
        """get_all on empty tracker should return empty list."""
        tracker = SpeakerTracker()
        assert tracker.get_all() == []

    def test_get_all_multiple_speakers(self):
        """get_all should return all tracked speakers."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")
        tracker.get_or_create("SPEAKER_01")
        tracker.get_or_create("SPEAKER_02")

        all_speakers = tracker.get_all()
        assert len(all_speakers) == 3
        assert all(isinstance(s, SpeakerProfile) for s in all_speakers)

    def test_get_all_returns_copy(self):
        """get_all should return a list (not dict)."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")

        speakers = tracker.get_all()
        assert isinstance(speakers, list)


class TestSpeakerListSummary:
    """Test text summary generation."""

    def test_summary_empty(self):
        """Empty tracker should return placeholder message."""
        tracker = SpeakerTracker()
        summary = tracker.get_speaker_list_summary()
        assert "no speakers yet" in summary

    def test_summary_single_speaker(self):
        """Summary should include speaker label and count."""
        tracker = SpeakerTracker()
        tracker.record_utterance("SPEAKER_00")
        tracker.record_utterance("SPEAKER_00")

        summary = tracker.get_speaker_list_summary()
        assert "Speaker A" in summary
        assert "2 utterances" in summary

    def test_summary_with_custom_name(self):
        """Summary should use custom name when available."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")
        tracker.rename("SPEAKER_00", "David")
        tracker.record_utterance("SPEAKER_00")

        summary = tracker.get_speaker_list_summary()
        assert "David" in summary
        assert "Speaker A" not in summary

    def test_summary_with_role_hint(self):
        """Summary should include role hint."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")
        tracker.set_role_hint("SPEAKER_00", "student")
        tracker.record_utterance("SPEAKER_00")

        summary = tracker.get_speaker_list_summary()
        assert "(student)" in summary

    def test_summary_multiple_speakers(self):
        """Summary should include all speakers separated by semicolons."""
        tracker = SpeakerTracker()
        tracker.record_utterance("SPEAKER_00")
        tracker.record_utterance("SPEAKER_01")
        tracker.record_utterance("SPEAKER_01")

        summary = tracker.get_speaker_list_summary()
        assert "Speaker A" in summary
        assert "Speaker B" in summary
        assert "1 utterances" in summary or "1 utterance" in summary
        assert "2 utterances" in summary
        assert ";" in summary


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_many_speakers(self):
        """Tracker should handle many speakers (beyond A-Z)."""
        tracker = SpeakerTracker()

        # Create 30 speakers
        for i in range(30):
            speaker = tracker.get_or_create(f"SPEAKER_{i:02d}")
            assert speaker.id == f"SPEAKER_{i:02d}"

        assert len(tracker.get_all()) == 30

    def test_speaker_id_special_characters(self):
        """Speaker IDs with special characters should work."""
        tracker = SpeakerTracker()
        speaker = tracker.get_or_create("SPEAKER_00-ABC-123")
        assert speaker.id == "SPEAKER_00-ABC-123"

    def test_custom_name_special_characters(self):
        """Custom names with special characters should work."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")
        tracker.rename("SPEAKER_00", "José María")

        speaker = tracker.get_or_create("SPEAKER_00")
        assert speaker.custom_name == "José María"

    def test_zero_utterances(self):
        """Speaker with zero utterances should appear in summary."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")

        summary = tracker.get_speaker_list_summary()
        assert "0 utterances" in summary

    def test_rename_then_role_hint(self):
        """Both custom name and role hint should coexist."""
        tracker = SpeakerTracker()
        tracker.get_or_create("SPEAKER_00")
        tracker.rename("SPEAKER_00", "Emma")
        tracker.set_role_hint("SPEAKER_00", "instructor")

        summary = tracker.get_speaker_list_summary()
        assert "Emma" in summary
        assert "(instructor)" in summary

    def test_utterance_count_persistence(self):
        """Utterance count should persist across operations."""
        tracker = SpeakerTracker()
        tracker.record_utterance("SPEAKER_00")
        tracker.rename("SPEAKER_00", "Frank")
        tracker.set_role_hint("SPEAKER_00", "guide")
        tracker.record_utterance("SPEAKER_00")

        speaker = tracker.get_or_create("SPEAKER_00")
        assert speaker.utterance_count == 2
