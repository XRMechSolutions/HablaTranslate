# Habla Test Suite

Comprehensive unit tests for Habla's core components.

## Running Tests

### Install test dependencies
```bash
cd habla
pip install -r requirements.txt
```

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=server --cov-report=html --cov-report=term
```

### Run specific test files
```bash
pytest tests/pipeline/test_translator.py
pytest tests/services/test_idiom_scanner.py
pytest tests/services/test_speaker_tracker.py
pytest tests/services/test_vocab.py
pytest tests/pipeline/test_vad_buffer.py
```

### Run tests by marker
```bash
pytest -m unit          # Fast unit tests only
pytest -m integration   # Integration tests (DB, file I/O)
pytest -m slow          # Slow tests (model loading)
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test class or method
```bash
pytest tests/pipeline/test_translator.py::TestRetryLogic
pytest tests/pipeline/test_translator.py::TestRetryLogic::test_is_retryable_timeout
```

## Test Coverage

### Core Components
- **translator.py** (335 lines)
  - Provider switching (Ollama, LM Studio, OpenAI)
  - Retry logic with exponential backoff
  - Fallback from cloud to local
  - JSON parsing with error recovery
  - SM-2 algorithm validation
  - Cost tracking for OpenAI

- **idiom_scanner.py** (97 tests)
  - Pattern loading from JSON and DB
  - Regex matching (case-insensitive)
  - Deduplication
  - Starter idiom set validation

- **speaker_tracker.py** (73 tests)
  - Speaker creation and labeling
  - Custom naming
  - Role hints
  - Utterance counting
  - Display name resolution

- **vocab.py** (204 tests)
  - CRUD operations
  - Spaced repetition (SM-2)
  - Full-text search
  - Anki CSV export
  - Stats aggregation

- **vad_buffer.py** (396 tests)
  - Speech segmentation
  - Silero VAD integration
  - Energy-based fallback
  - Partial audio streaming
  - ffmpeg audio decoding

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── pipeline/
│   ├── test_translator.py   # LLM translation tests
│   └── test_vad_buffer.py   # VAD and audio tests
├── services/
│   ├── test_idiom_scanner.py
│   ├── test_speaker_tracker.py
│   └── test_vocab.py
└── db/
    └── (future database tests)
```

## Writing New Tests

### Unit Test Template
```python
import pytest

class TestFeature:
    """Test description."""

    def test_basic_case(self):
        """Test a basic scenario."""
        result = function_under_test()
        assert result == expected

    @pytest.mark.asyncio
    async def test_async_case(self):
        """Test async function."""
        result = await async_function()
        assert result is not None
```

### Using Fixtures
```python
def test_with_fixture(translator_config, mock_httpx_client):
    """Fixtures are automatically injected."""
    translator = Translator(translator_config)
    translator.client = mock_httpx_client
    # Test code here
```

## Continuous Integration

Add to `.github/workflows/test.yml`:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest --cov=server --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## Notes

- Tests are isolated — no shared state between tests
- Mock external dependencies (LLM APIs, databases, models)
- Use `@pytest.mark.asyncio` for async tests
- Keep tests fast — avoid loading real ML models in unit tests
- Integration tests for database operations use in-memory SQLite
