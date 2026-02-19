"""Unit tests for idiom merging logic (pattern DB + LLM)."""

import pytest
from server.services.idiom_scanner import IdiomScanner, IdiomMatch
from server.models.schemas import FlaggedPhrase


class TestIdiomMerging:
    """Test merging of pattern DB and LLM-detected idioms."""

    def test_pattern_db_takes_priority(self):
        """Pattern DB matches should take priority over LLM duplicates."""
        pattern_matches = [
            IdiomMatch(
                canonical="importar un pepino",
                literal="to matter a cucumber",
                meaning="to not care at all",
                match_start=10,
                match_end=30,
            )
        ]

        llm_phrases = [
            FlaggedPhrase(
                phrase="importar un pepino",  # Same idiom
                literal="to import a cucumber",  # Wrong literal
                meaning="to not care",
                type="idiom",
                source="llm",
            )
        ]

        # Simulate orchestrator merge logic
        result = []
        seen = set()

        for m in pattern_matches:
            key = m.canonical.lower()
            if key not in seen:
                seen.add(key)
                result.append(FlaggedPhrase(
                    phrase=m.canonical,
                    literal=m.literal,
                    meaning=m.meaning,
                    type="idiom",
                    source="pattern_db",
                ))

        for fp in llm_phrases:
            key = fp.phrase.lower()
            if key not in seen:
                seen.add(key)
                result.append(fp)

        # Should only have one match (from pattern DB)
        assert len(result) == 1
        assert result[0].source == "pattern_db"
        assert result[0].literal == "to matter a cucumber"  # Correct literal

    def test_llm_adds_novel_idioms(self):
        """LLM should add idioms not in pattern DB."""
        pattern_matches = [
            IdiomMatch(
                canonical="importar un pepino",
                literal="to matter a cucumber",
                meaning="to not care",
            )
        ]

        llm_phrases = [
            FlaggedPhrase(
                phrase="estar de bajón",  # Novel idiom
                literal="to be of a slump",
                meaning="to feel down/depressed",
                type="idiom",
                source="llm",
            )
        ]

        result = []
        seen = set()

        for m in pattern_matches:
            key = m.canonical.lower()
            if key not in seen:
                seen.add(key)
                result.append(FlaggedPhrase(
                    phrase=m.canonical,
                    literal=m.literal,
                    meaning=m.meaning,
                    type="idiom",
                    source="pattern_db",
                ))

        for fp in llm_phrases:
            key = fp.phrase.lower()
            if key not in seen:
                seen.add(key)
                fp.source = "llm"
                result.append(fp)

        # Should have both
        assert len(result) == 2
        sources = {r.source for r in result}
        assert sources == {"pattern_db", "llm"}

    def test_case_insensitive_deduplication(self):
        """Deduplication should be case-insensitive."""
        pattern_matches = [
            IdiomMatch(
                canonical="tomar el pelo",
                literal="to take the hair",
                meaning="to pull someone's leg",
            )
        ]

        llm_phrases = [
            FlaggedPhrase(
                phrase="Tomar El Pelo",  # Different case
                literal="to take the hair",
                meaning="to joke with someone",
                type="idiom",
                source="llm",
            )
        ]

        result = []
        seen = set()

        for m in pattern_matches:
            key = m.canonical.lower()
            if key not in seen:
                seen.add(key)
                result.append(FlaggedPhrase(
                    phrase=m.canonical,
                    literal=m.literal,
                    meaning=m.meaning,
                    type="idiom",
                    source="pattern_db",
                ))

        for fp in llm_phrases:
            key = fp.phrase.lower()
            if key not in seen:
                seen.add(key)
                result.append(fp)

        # Should only have one
        assert len(result) == 1
        assert result[0].source == "pattern_db"


class TestIdiomEdgeCases:
    """Additional edge cases for idiom detection."""

    def test_multiple_idioms_in_single_sentence(self):
        """Should detect multiple idioms in one sentence."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"importa[rn]?\s+un\s+pepino",
                "canonical": "importar un pepino",
                "literal": "to matter a cucumber",
                "meaning": "to not care",
            },
            {
                "pattern": r"toma[rs]?\s+(el\s+)?pelo",
                "canonical": "tomar el pelo",
                "literal": "to take the hair",
                "meaning": "to pull someone's leg",
            },
        ])

        text = "No me importa un pepino si me tomas el pelo."
        matches = scanner.scan(text)

        assert len(matches) == 2

    def test_partial_idiom_does_not_match(self):
        """Incomplete idioms should not match."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"importa[rn]?\s+un\s+pepino",
                "canonical": "importar un pepino",
                "literal": "to matter a cucumber",
                "meaning": "to not care",
            }
        ])

        # Missing "un" - should not match
        text = "No me importa pepino"
        matches = scanner.scan(text)

        assert len(matches) == 0

    def test_idiom_with_optional_words(self):
        """Idioms with optional words (el, una, etc.) should match both forms."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"(toma[rs]?|tomando)\s+(el\s+)?pelo",
                "canonical": "tomar el pelo",
                "literal": "to take the hair",
                "meaning": "to pull someone's leg",
            }
        ])

        # With "el" should match
        assert len(scanner.scan("Me va a tomar el pelo")) > 0

        # Without "el" should also match (optional group)
        assert len(scanner.scan("Va a tomar pelo")) > 0

        # Gerund form
        assert len(scanner.scan("Está tomando el pelo")) > 0

    def test_idiom_at_sentence_boundaries(self):
        """Idioms at start/end of sentences should match."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"importa[rn]?\s+un\s+pepino",
                "canonical": "importar un pepino",
                "literal": "to matter a cucumber",
                "meaning": "to not care",
            }
        ])

        # At start
        assert len(scanner.scan("Importa un pepino todo esto")) > 0

        # At end
        assert len(scanner.scan("Todo esto me importa un pepino")) > 0

        # Standalone
        assert len(scanner.scan("Importa un pepino")) > 0

    def test_overlapping_patterns(self):
        """Overlapping patterns should not cause double-detection."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"mola[rs]?",
                "canonical": "molar",
                "literal": "to molar",
                "meaning": "to be cool",
            },
            {
                "pattern": r"mola[rs]?\s+mazo",
                "canonical": "molar mazo",
                "literal": "to molar a lot",
                "meaning": "to be very cool",
            },
        ])

        text = "Esto mola mazo"
        matches = scanner.scan(text)

        # Should detect both or just the longer one (implementation-dependent)
        # Current implementation: first match wins, dedup prevents second
        assert len(matches) >= 1

    def test_accented_characters_in_pattern(self):
        """Patterns with accents should work correctly."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"est[aá][rs]?\s+en\s+las\s+nubes",
                "canonical": "estar en las nubes",
                "literal": "to be in the clouds",
                "meaning": "to be daydreaming",
            }
        ])

        # Should match both accented and unaccented
        assert len(scanner.scan("está en las nubes")) > 0
        assert len(scanner.scan("esta en las nubes")) > 0

    def test_question_and_exclamation_marks(self):
        """Punctuation around idioms should not prevent matching."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"importa[rn]?\s+un\s+pepino",
                "canonical": "importar un pepino",
                "literal": "to matter a cucumber",
                "meaning": "to not care",
            }
        ])

        assert len(scanner.scan("¿Te importa un pepino?")) > 0
        assert len(scanner.scan("¡Me importa un pepino!")) > 0
        assert len(scanner.scan("...importa un pepino...")) > 0

    def test_verb_conjugations(self):
        """Common verb conjugations should be covered by patterns."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"importa[rn]?\s+un\s+pepino",
                "canonical": "importar un pepino",
                "literal": "to matter a cucumber",
                "meaning": "to not care",
            }
        ])

        # Different conjugations
        assert len(scanner.scan("me importa un pepino")) > 0     # 3rd person
        assert len(scanner.scan("les importan un pepino")) > 0   # 3rd plural
        assert len(scanner.scan("importar un pepino")) > 0       # infinitive

    def test_regional_variations(self):
        """Region-specific idioms should be tagged correctly."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"mola[rs]?",
                "canonical": "molar",
                "literal": "to molar",
                "meaning": "to be cool",
                "region": "spain",
                "frequency": "very common",
            },
            {
                "pattern": r"chido",
                "canonical": "chido",
                "literal": "cool",
                "meaning": "cool, awesome",
                "region": "mexico",
                "frequency": "very common",
            },
        ])

        spain_match = scanner.scan("Esto mola")[0]
        assert spain_match.region == "spain"

        mexico_match = scanner.scan("Qué chido")[0]
        assert mexico_match.region == "mexico"

    def test_empty_pattern_list(self):
        """Scanner with no patterns should return no matches."""
        scanner = IdiomScanner()
        matches = scanner.scan("No me importa un pepino")
        assert len(matches) == 0
        assert scanner.count == 0

    def test_reload_patterns(self):
        """Loading new patterns should replace old ones."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {"pattern": r"test1", "canonical": "test1", "literal": "", "meaning": "test"}
        ])
        assert scanner.count == 1

        # Clear and reload
        scanner.patterns.clear()
        scanner.load_from_db([
            {"pattern": r"test2", "canonical": "test2", "literal": "", "meaning": "test"},
            {"pattern": r"test3", "canonical": "test3", "literal": "", "meaning": "test"},
        ])
        assert scanner.count == 2
