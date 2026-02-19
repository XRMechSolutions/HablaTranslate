"""Unit tests for the IdiomScanner service."""

import pytest
import json
from pathlib import Path

from server.services.idiom_scanner import IdiomScanner, IdiomMatch, IdiomPattern, create_starter_idioms


class TestIdiomScannerInit:
    """Test IdiomScanner initialization."""

    def test_init_empty(self):
        """Scanner should initialize with no patterns."""
        scanner = IdiomScanner()
        assert scanner.count == 0
        assert scanner.patterns == []

    def test_load_from_json(self, sample_idiom_json_file):
        """Load patterns from JSON file."""
        scanner = IdiomScanner()
        scanner.load_from_json(sample_idiom_json_file)
        assert scanner.count == 2
        assert any(p.canonical == "importar un pepino" for p in scanner.patterns)

    def test_load_from_nonexistent_file(self, tmp_path):
        """Loading from nonexistent file should not crash."""
        scanner = IdiomScanner()
        scanner.load_from_json(tmp_path / "nonexistent.json")
        assert scanner.count == 0

    def test_load_from_db(self):
        """Load patterns from database rows."""
        scanner = IdiomScanner()
        rows = [
            {
                "pattern": r"importar\s+un\s+pepino",
                "canonical": "importar un pepino",
                "literal": "to matter a cucumber",
                "meaning": "to not care at all",
                "region": "universal",
                "frequency": "very common",
            },
            {
                "pattern": r"tomar\s+el\s+pelo",
                "canonical": "tomar el pelo",
                "literal": "to take the hair",
                "meaning": "to pull someone's leg",
                "region": "universal",
                "frequency": "common",
            },
        ]
        scanner.load_from_db(rows)
        assert scanner.count == 2

    def test_load_skips_bad_regex(self, tmp_path):
        """Bad regex patterns should be skipped."""
        bad_data = [
            {
                "pattern": r"[invalid(regex",  # Invalid regex
                "canonical": "test",
                "meaning": "test",
            },
            {
                "pattern": r"valid\s+pattern",
                "canonical": "valid pattern",
                "meaning": "valid",
            },
        ]
        file_path = tmp_path / "bad_idioms.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(bad_data, f)

        scanner = IdiomScanner()
        scanner.load_from_json(file_path)
        assert scanner.count == 1  # Only the valid one


class TestIdiomScanning:
    """Test idiom detection in text."""

    def test_scan_simple_match(self):
        """Detect a simple idiom."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"importa[rn]?\s+un\s+pepino",
                "canonical": "importar un pepino",
                "literal": "to matter a cucumber",
                "meaning": "to not care at all",
            }
        ])

        text = "No me importa un pepino lo que pienses."
        matches = scanner.scan(text)

        assert len(matches) == 1
        assert matches[0].canonical == "importar un pepino"
        assert matches[0].meaning == "to not care at all"

    def test_scan_case_insensitive(self):
        """Pattern matching should be case-insensitive."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"est[aá][rs]?\s+en\s+las\s+nubes",
                "canonical": "estar en las nubes",
                "literal": "to be in the clouds",
                "meaning": "to be daydreaming",
            }
        ])

        text = "Siempre ESTÁ EN LAS NUBES durante clase."
        matches = scanner.scan(text)

        assert len(matches) == 1
        assert matches[0].canonical == "estar en las nubes"

    def test_scan_multiple_matches(self):
        """Detect multiple different idioms."""
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
        canonicals = {m.canonical for m in matches}
        assert "importar un pepino" in canonicals
        assert "tomar el pelo" in canonicals

    def test_scan_no_duplicates(self):
        """Same idiom should only match once per scan."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"importa[rn]?\s+un\s+pepino",
                "canonical": "importar un pepino",
                "literal": "to matter a cucumber",
                "meaning": "to not care",
            }
        ])

        text = "Me importa un pepino esto y me importa un pepino aquello."
        matches = scanner.scan(text)

        # Should only get one match despite two occurrences
        assert len(matches) == 1

    def test_scan_no_matches(self):
        """Text without idioms should return empty list."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"importar\s+un\s+pepino",
                "canonical": "importar un pepino",
                "meaning": "to not care",
            }
        ])

        text = "Este es un texto normal sin idiomas."
        matches = scanner.scan(text)

        assert len(matches) == 0

    def test_scan_captures_position(self):
        """Matches should include start and end positions."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"meter\s+la\s+pata",
                "canonical": "meter la pata",
                "meaning": "to make a blunder",
            }
        ])

        text = "Acabo de meter la pata con mi jefe."
        matches = scanner.scan(text)

        assert len(matches) == 1
        assert matches[0].match_start >= 0
        assert matches[0].match_end > matches[0].match_start

    def test_scan_with_region_metadata(self):
        """Matches should preserve region metadata."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"mola[rs]?",
                "canonical": "molar",
                "literal": "to molar",
                "meaning": "to be cool",
                "region": "spain",
                "frequency": "very common",
            }
        ])

        text = "¡Mola mucho tu coche!"
        matches = scanner.scan(text)

        assert len(matches) == 1
        assert matches[0].region == "spain"
        assert matches[0].frequency == "very common"


class TestStarterIdioms:
    """Test the starter idiom set."""

    def test_create_starter_idioms_returns_list(self):
        """create_starter_idioms should return a list of dicts."""
        idioms = create_starter_idioms()
        assert isinstance(idioms, list)
        assert len(idioms) > 0
        assert all(isinstance(i, dict) for i in idioms)

    def test_starter_idioms_have_required_fields(self):
        """Each starter idiom should have required fields."""
        idioms = create_starter_idioms()
        for idiom in idioms:
            assert "pattern" in idiom
            assert "canonical" in idiom
            assert "meaning" in idiom

    def test_starter_idioms_compile(self):
        """All starter idiom patterns should compile as valid regex."""
        scanner = IdiomScanner()
        idioms = create_starter_idioms()
        scanner.load_from_db(idioms)

        # All patterns should have loaded (none skipped due to bad regex)
        assert scanner.count == len(idioms)

    def test_starter_idioms_detect_common_phrases(self):
        """Starter set should detect common Spanish idioms."""
        scanner = IdiomScanner()
        scanner.load_from_db(create_starter_idioms())

        # Test that scanner contains common idioms
        test_phrases = [
            "importar un pepino",
            "estar en las nubes",
            "meter la pata",
        ]

        canonicals = {p.canonical.lower() for p in scanner.patterns}
        for phrase in test_phrases:
            assert any(phrase in c for c in canonicals), f"Missing idiom: {phrase}"

        # Test actual detection with a few examples
        assert len(scanner.scan("No me importa un pepino")) > 0
        assert len(scanner.scan("Está en las nubes")) > 0
        assert len(scanner.scan("meter la pata")) > 0


class TestIdiomMatch:
    """Test IdiomMatch dataclass."""

    def test_idiom_match_defaults(self):
        """IdiomMatch should have sensible defaults."""
        match = IdiomMatch(
            canonical="test",
            literal="test literal",
            meaning="test meaning",
        )
        assert match.region == "universal"
        assert match.frequency == "common"
        assert match.match_start == 0
        assert match.match_end == 0

    def test_idiom_match_custom_values(self):
        """IdiomMatch should accept custom values."""
        match = IdiomMatch(
            canonical="molar",
            literal="to molar",
            meaning="to be cool",
            region="spain",
            frequency="very common",
            match_start=10,
            match_end=20,
        )
        assert match.region == "spain"
        assert match.frequency == "very common"
        assert match.match_start == 10
        assert match.match_end == 20


class TestIdiomPattern:
    """Test IdiomPattern dataclass."""

    def test_idiom_pattern_creation(self):
        """IdiomPattern should store compiled regex."""
        import re
        pattern = IdiomPattern(
            pattern=re.compile(r"test\s+pattern", re.IGNORECASE),
            canonical="test pattern",
            literal="test literal",
            meaning="test meaning",
        )
        assert pattern.canonical == "test pattern"
        assert pattern.pattern.search("TEST PATTERN") is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_text(self):
        """Scanning empty text should return empty list."""
        scanner = IdiomScanner()
        scanner.load_from_db(create_starter_idioms())
        matches = scanner.scan("")
        assert matches == []

    def test_very_long_text(self):
        """Scanner should handle very long text efficiently."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"needle\s+in\s+haystack",
                "canonical": "needle in haystack",
                "meaning": "test",
            }
        ])

        # 10KB of text with idiom at the end
        long_text = ("Lorem ipsum dolor sit amet. " * 500) + "needle in haystack"
        matches = scanner.scan(long_text)

        assert len(matches) == 1
        assert matches[0].canonical == "needle in haystack"

    def test_unicode_text(self):
        """Scanner should handle accented characters correctly."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"está\s+en\s+las\s+nubes",
                "canonical": "estar en las nubes",
                "meaning": "to be daydreaming",
            }
        ])

        text = "María está en las nubes."
        matches = scanner.scan(text)

        assert len(matches) == 1

    def test_special_regex_characters_in_text(self):
        """Scanner should handle text with regex special chars."""
        scanner = IdiomScanner()
        scanner.load_from_db([
            {
                "pattern": r"importa[rn]?\s+un\s+pepino",
                "canonical": "importar un pepino",
                "literal": "to matter a cucumber",
                "meaning": "to not care",
            }
        ])

        text = "¿No te importa un pepino? (realmente)"
        matches = scanner.scan(text)

        assert len(matches) == 1
