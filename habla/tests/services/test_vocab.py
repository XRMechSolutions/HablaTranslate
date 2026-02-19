"""Unit tests for the VocabService (database integration tests)."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from server.services.vocab import VocabService
from server.models.schemas import FlaggedPhrase


class MockDBCursor:
    """Mock database cursor."""

    def __init__(self, lastrowid=1):
        self.lastrowid = lastrowid
        self.rowcount = 1


class MockDB:
    """Mock database connection."""

    def __init__(self):
        self.data = {}
        self.next_id = 1

    async def execute(self, query, params=()):
        """Mock execute."""
        cursor = MockDBCursor(lastrowid=self.next_id)
        self.next_id += 1
        return cursor

    async def execute_fetchall(self, query, params=()):
        """Mock fetchall."""
        return []

    async def commit(self):
        """Mock commit."""
        pass


@pytest.fixture
def mock_db():
    """Fixture providing a mock database."""
    return MockDB()


@pytest.fixture
def vocab_service():
    """Fixture providing a VocabService instance."""
    return VocabService()


@pytest.fixture
def sample_flagged_phrase():
    """Fixture providing a sample FlaggedPhrase."""
    return FlaggedPhrase(
        phrase="importar un pepino",
        literal="to matter a cucumber",
        meaning="to not care at all",
        type="idiom",
        save_worthy=True,
        source="llm",
        region="universal",
    )


class TestSaveFromPhrase:
    """Test saving vocab from flagged phrases."""

    @pytest.mark.asyncio
    async def test_save_new_phrase(self, vocab_service, sample_flagged_phrase, mock_db):
        """Saving a new phrase should insert into database."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            # Mock empty result for existing check
            mock_db.execute_fetchall = AsyncMock(return_value=[])

            vocab_id = await vocab_service.save_from_phrase(
                sample_flagged_phrase,
                exchange_id=123,
                speaker_id="SPEAKER_00",
                source_sentence="No me importa un pepino.",
            )

            assert vocab_id > 0

    @pytest.mark.asyncio
    async def test_save_existing_phrase_increments_count(self, vocab_service, sample_flagged_phrase, mock_db):
        """Saving existing phrase should increment encounter count."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            # Mock existing phrase found
            existing_row = {"id": 42, "times_encountered": 3}
            mock_db.execute_fetchall = AsyncMock(return_value=[existing_row])
            mock_db.execute = AsyncMock(return_value=MockDBCursor())

            vocab_id = await vocab_service.save_from_phrase(sample_flagged_phrase)

            assert vocab_id == 42
            # Should have called UPDATE
            assert mock_db.execute.called

    @pytest.mark.asyncio
    async def test_save_without_optional_fields(self, vocab_service, sample_flagged_phrase, mock_db):
        """Saving phrase without optional fields should work."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_db.execute_fetchall = AsyncMock(return_value=[])

            vocab_id = await vocab_service.save_from_phrase(sample_flagged_phrase)

            assert vocab_id > 0


class TestGetAll:
    """Test retrieving all vocab items."""

    @pytest.mark.asyncio
    async def test_get_all_empty(self, vocab_service, mock_db):
        """get_all should return empty list when no vocab exists."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_db.execute_fetchall = AsyncMock(return_value=[])

            items = await vocab_service.get_all()

            assert items == []

    @pytest.mark.asyncio
    async def test_get_all_with_limit_offset(self, vocab_service, mock_db):
        """get_all should support pagination."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_rows = [
                {"id": 1, "term": "hola", "meaning": "hello"},
                {"id": 2, "term": "adiós", "meaning": "goodbye"},
            ]
            mock_db.execute_fetchall = AsyncMock(return_value=mock_rows)

            items = await vocab_service.get_all(limit=10, offset=0)

            assert len(items) == 2
            assert items[0]["term"] == "hola"

    @pytest.mark.asyncio
    async def test_get_all_with_category_filter(self, vocab_service, mock_db):
        """get_all should support category filtering."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_rows = [{"id": 1, "term": "test", "category": "idiom"}]
            mock_db.execute_fetchall = AsyncMock(return_value=mock_rows)

            items = await vocab_service.get_all(category="idiom")

            assert len(items) == 1
            # Verify query included category filter
            call_args = mock_db.execute_fetchall.call_args
            assert "category" in call_args[0][0]


class TestGetDueForReview:
    """Test spaced repetition review queries."""

    @pytest.mark.asyncio
    async def test_get_due_for_review_empty(self, vocab_service, mock_db):
        """No items due should return empty list."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_db.execute_fetchall = AsyncMock(return_value=[])

            items = await vocab_service.get_due_for_review()

            assert items == []

    @pytest.mark.asyncio
    async def test_get_due_for_review_with_due_items(self, vocab_service, mock_db):
        """Items with past review date should be returned."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            past_date = (datetime.utcnow() - timedelta(days=1)).isoformat()
            mock_rows = [
                {"id": 1, "term": "test", "next_review": past_date},
            ]
            mock_db.execute_fetchall = AsyncMock(return_value=mock_rows)

            items = await vocab_service.get_due_for_review(limit=20)

            assert len(items) == 1

    @pytest.mark.asyncio
    async def test_get_due_for_review_respects_limit(self, vocab_service, mock_db):
        """Limit parameter should be respected."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_db.execute_fetchall = AsyncMock(return_value=[])

            await vocab_service.get_due_for_review(limit=5)

            call_args = mock_db.execute_fetchall.call_args
            assert call_args[0][1][1] == 5  # Limit param


class TestRecordReview:
    """Test SM-2 spaced repetition algorithm."""

    @pytest.mark.asyncio
    async def test_record_review_not_found(self, vocab_service, mock_db):
        """Recording review for nonexistent item should return error."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_db.execute_fetchall = AsyncMock(return_value=[])

            result = await vocab_service.record_review(vocab_id=999, quality=5)

            assert result["error"] == "not found"

    @pytest.mark.asyncio
    async def test_record_review_first_success(self, vocab_service, mock_db):
        """First successful review (quality >= 3) should set interval to 1 day."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            existing = {
                "id": 1,
                "ease_factor": 2.5,
                "interval_days": 0,
                "repetitions": 0,
            }
            mock_db.execute_fetchall = AsyncMock(return_value=[existing])
            mock_db.execute = AsyncMock(return_value=MockDBCursor())

            result = await vocab_service.record_review(vocab_id=1, quality=4)

            assert result["interval_days"] == 1
            assert result["repetitions"] == 1

    @pytest.mark.asyncio
    async def test_record_review_second_success(self, vocab_service, mock_db):
        """Second successful review should set interval to 6 days."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            existing = {
                "id": 1,
                "ease_factor": 2.5,
                "interval_days": 1,
                "repetitions": 1,
            }
            mock_db.execute_fetchall = AsyncMock(return_value=[existing])
            mock_db.execute = AsyncMock(return_value=MockDBCursor())

            result = await vocab_service.record_review(vocab_id=1, quality=4)

            assert result["interval_days"] == 6
            assert result["repetitions"] == 2

    @pytest.mark.asyncio
    async def test_record_review_subsequent_success(self, vocab_service, mock_db):
        """Subsequent reviews should multiply interval by ease factor."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            existing = {
                "id": 1,
                "ease_factor": 2.5,
                "interval_days": 6,
                "repetitions": 2,
            }
            mock_db.execute_fetchall = AsyncMock(return_value=[existing])
            mock_db.execute = AsyncMock(return_value=MockDBCursor())

            result = await vocab_service.record_review(vocab_id=1, quality=4)

            assert result["interval_days"] == 15  # 6 * 2.5
            assert result["repetitions"] == 3

    @pytest.mark.asyncio
    async def test_record_review_failure_resets(self, vocab_service, mock_db):
        """Failed review (quality < 3) should reset progress."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            existing = {
                "id": 1,
                "ease_factor": 2.5,
                "interval_days": 15,
                "repetitions": 5,
            }
            mock_db.execute_fetchall = AsyncMock(return_value=[existing])
            mock_db.execute = AsyncMock(return_value=MockDBCursor())

            result = await vocab_service.record_review(vocab_id=1, quality=1)

            assert result["interval_days"] == 1
            assert result["repetitions"] == 0

    @pytest.mark.asyncio
    async def test_record_review_adjusts_ease_factor(self, vocab_service, mock_db):
        """Ease factor should adjust based on quality."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            existing = {
                "id": 1,
                "ease_factor": 2.5,
                "interval_days": 6,
                "repetitions": 2,
            }
            mock_db.execute_fetchall = AsyncMock(return_value=[existing])
            mock_db.execute = AsyncMock(return_value=MockDBCursor())

            # Quality 5 (easy) should increase ease
            result = await vocab_service.record_review(vocab_id=1, quality=5)
            assert result["ease_factor"] > 2.5

    @pytest.mark.asyncio
    async def test_record_review_ease_factor_floor(self, vocab_service, mock_db):
        """Ease factor should not go below 1.3."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            existing = {
                "id": 1,
                "ease_factor": 1.3,
                "interval_days": 6,
                "repetitions": 2,
            }
            mock_db.execute_fetchall = AsyncMock(return_value=[existing])
            mock_db.execute = AsyncMock(return_value=MockDBCursor())

            # Quality 0 (complete failure) would normally decrease ease heavily
            result = await vocab_service.record_review(vocab_id=1, quality=0)
            assert result["ease_factor"] >= 1.3


class TestDelete:
    """Test vocab deletion."""

    @pytest.mark.asyncio
    async def test_delete_existing_item(self, vocab_service, mock_db):
        """Deleting existing item should return True."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            cursor = MockDBCursor()
            cursor.rowcount = 1
            mock_db.execute = AsyncMock(return_value=cursor)

            result = await vocab_service.delete(vocab_id=1)

            assert result is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_item(self, vocab_service, mock_db):
        """Deleting nonexistent item should return False."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            cursor = MockDBCursor()
            cursor.rowcount = 0
            mock_db.execute = AsyncMock(return_value=cursor)

            result = await vocab_service.delete(vocab_id=999)

            assert result is False


class TestSearch:
    """Test full-text search."""

    @pytest.mark.asyncio
    async def test_search_empty_results(self, vocab_service, mock_db):
        """Search with no matches should return empty list."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_db.execute_fetchall = AsyncMock(return_value=[])

            results = await vocab_service.search("nonexistent")

            assert results == []

    @pytest.mark.asyncio
    async def test_search_with_results(self, vocab_service, mock_db):
        """Search should return matching vocab items."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_rows = [
                {"id": 1, "term": "importar un pepino", "meaning": "to not care"},
            ]
            mock_db.execute_fetchall = AsyncMock(return_value=mock_rows)

            results = await vocab_service.search("pepino")

            assert len(results) == 1
            assert results[0]["term"] == "importar un pepino"

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, vocab_service, mock_db):
        """Search should respect limit parameter."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_db.execute_fetchall = AsyncMock(return_value=[])

            await vocab_service.search("test", limit=5)

            call_args = mock_db.execute_fetchall.call_args
            assert call_args[0][1][1] == 5


class TestExportAnkiCSV:
    """Test Anki CSV export."""

    @pytest.mark.asyncio
    async def test_export_anki_csv_empty(self, vocab_service, mock_db):
        """Exporting empty vocab should return empty CSV."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_db.execute_fetchall = AsyncMock(return_value=[])

            csv_output = await vocab_service.export_anki_csv()

            assert csv_output == ""

    @pytest.mark.asyncio
    async def test_export_anki_csv_single_item(self, vocab_service, mock_db):
        """Export should format vocab as Anki TSV."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_rows = [
                {
                    "term": "importar un pepino",
                    "literal": "to matter a cucumber",
                    "meaning": "to not care at all",
                    "category": "idiom",
                    "source_sentence": "No me importa un pepino.",
                    "region": "universal",
                }
            ]
            mock_db.execute_fetchall = AsyncMock(return_value=mock_rows)

            csv_output = await vocab_service.export_anki_csv()

            assert "importar un pepino" in csv_output
            assert "to not care at all" in csv_output
            assert "habla" in csv_output  # Tags
            assert "\t" in csv_output  # TSV format

    @pytest.mark.asyncio
    async def test_export_anki_csv_includes_example(self, vocab_service, mock_db):
        """Export should include source sentence in back of card."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_rows = [
                {
                    "term": "test",
                    "literal": None,
                    "meaning": "test meaning",
                    "category": "idiom",
                    "source_sentence": "Example sentence here.",
                    "region": "universal",
                }
            ]
            mock_db.execute_fetchall = AsyncMock(return_value=mock_rows)

            csv_output = await vocab_service.export_anki_csv()

            assert "Example sentence here" in csv_output


class TestExportJSON:
    """Test JSON export."""

    @pytest.mark.asyncio
    async def test_export_json(self, vocab_service, mock_db):
        """Export JSON should return all vocab items."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_rows = [
                {"id": 1, "term": "hola", "meaning": "hello"},
                {"id": 2, "term": "adiós", "meaning": "goodbye"},
            ]
            mock_db.execute_fetchall = AsyncMock(return_value=mock_rows)

            result = await vocab_service.export_json()

            assert len(result) == 2
            assert result[0]["term"] == "hola"


class TestGetStats:
    """Test vocabulary statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, vocab_service, mock_db):
        """Stats on empty vocab should return zeros."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_db.execute_fetchall = AsyncMock(side_effect=[
                [{"c": 0}],  # total
                [{"c": 0}],  # due
                [],          # by category
            ])

            stats = await vocab_service.get_stats()

            assert stats["total"] == 0
            assert stats["due_for_review"] == 0
            assert stats["by_category"] == {}

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, vocab_service, mock_db):
        """Stats should aggregate vocab data."""
        with patch("server.services.vocab.get_db", return_value=mock_db):
            mock_db.execute_fetchall = AsyncMock(side_effect=[
                [{"c": 25}],  # total
                [{"c": 7}],   # due
                [             # by category
                    {"category": "idiom", "c": 15},
                    {"category": "false_friend", "c": 10},
                ],
            ])

            stats = await vocab_service.get_stats()

            assert stats["total"] == 25
            assert stats["due_for_review"] == 7
            assert stats["by_category"]["idiom"] == 15
            assert stats["by_category"]["false_friend"] == 10
