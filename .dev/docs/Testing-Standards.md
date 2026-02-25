# Habla Testing Standards

This document defines how tests should be designed, written, and audited across the Habla project.
Use it when writing new tests and when auditing existing test coverage.

---

## 1. Test Naming Convention

**Pattern**: `test_<method_or_behavior>_<condition>_<expected_outcome>`

```
GOOD:  test_translate_success_returns_translation_result
GOOD:  test_scan_case_insensitive_matches_uppercase
GOOD:  test_call_ollama_timeout_retries_with_backoff
GOOD:  test_vad_empty_audio_returns_no_segments

BAD:   test_translate
BAD:   test_it_works
BAD:   test_translator_1
```

- Group related tests in `class TestComponentName:` classes.
- Use `@pytest.mark.parametrize` when the same logic applies across multiple inputs.
- Organize test files to mirror source: `server/services/vocab.py` -> `tests/services/test_vocab.py`.

---

## 2. Test Structure (Arrange-Act-Assert)

Every test follows the AAA pattern with clear separation:

```python
@pytest.mark.asyncio
async def test_translate_success(self, translator_config, mock_httpx_client):
    # Arrange
    translator = Translator(translator_config)
    translator._client = mock_httpx_client

    # Act
    result = await translator.translate("Hola mundo", "es_to_en", context=[])

    # Assert
    assert result.translation == "Hello world"
    assert result.confidence >= 0.8
```

- One logical assertion per test (multiple asserts on the same result are fine).
- No logic (if/else, loops) inside tests. If you need loops, use `@pytest.mark.parametrize`.
- No try/except in tests -- let pytest handle exceptions via `pytest.raises()`.

---

## 3. Assertion Quality

### 3.1 Use Specific Assertions

```python
# GOOD -- specific, informative on failure
assert len(result.flagged_phrases) == 3
assert result.translation == "Hello world"
assert result.confidence == pytest.approx(0.85, abs=0.05)

# BAD -- vague, unhelpful failure message
assert result is not None
assert len(result.flagged_phrases) > 0
```

### 3.2 Include Messages for Non-Obvious Assertions

```python
assert result.speaker_id == "A", "First detected speaker should be labeled A"
assert len(exchanges) == 10, "Context window should hold exactly 10 exchanges"
```

### 3.3 Numerical Tolerance

Use `pytest.approx` for floating-point comparisons:

```python
assert result.confidence == pytest.approx(0.85, abs=0.05)
assert audio_duration == pytest.approx(3.2, abs=0.1)
```

| Measurement Type | Tolerance | Pattern |
|-----------------|-----------|---------|
| Confidence scores | +/-0.05 | `pytest.approx(expected, abs=0.05)` |
| Audio duration (seconds) | +/-0.1s | `pytest.approx(expected, abs=0.1)` |
| Timing measurements | +/-50ms | `pytest.approx(expected, abs=0.05)` |

---

## 4. Guarding Against Trivially-Passing Tests

A test that always passes regardless of implementation correctness is worse than no test -- it provides false confidence.

### 4.1 Red-Green Verification

When writing a new test:
1. Write the assertion first.
2. Confirm it would FAIL if the method returned a wrong value (mentally or by temporarily breaking the code).
3. A test that passes with `return None`, `return ""`, or `return []` is likely trivially true.

### 4.2 Common Patterns That Hide Bugs

```python
# BAD: Always passes if method returns any non-None value
assert result is not None

# BETTER: Assert on the actual content
assert result.translation == "Hello world"
assert result.source_language == "es"
```

```python
# BAD: Passes even if collection is empty
assert isinstance(result.flagged_phrases, list)

# BETTER: Assert exact expected count and content
assert len(result.flagged_phrases) == 2
assert result.flagged_phrases[0].phrase == "echar de menos"
```

```python
# BAD: Exception test that doesn't verify the exception type
with pytest.raises(Exception):
    await translator.translate(None, "es_to_en", [])

# BETTER: Verify specific exception type and message
with pytest.raises(ValueError, match="source text cannot be empty"):
    await translator.translate("", "es_to_en", [])
```

### 4.3 Mutation Testing Mindset

Ask: "If I changed the implementation to do X wrong, would this test catch it?"

For every pipeline test, verify:
- Swapping source/target language would cause failure.
- Returning empty translation would cause failure.
- Returning wrong speaker ID would cause failure.
- Skipping retry logic would cause failure on transient errors.

### 4.4 Boundary Value Assertions

```python
# Test AT the boundary, not just in the middle
# Context window holds 10 exchanges
context = [make_exchange() for _ in range(10)]
result = orchestrator._trim_context(context)
assert len(result) == 10, "Should keep all 10 when at limit"

context = [make_exchange() for _ in range(11)]
result = orchestrator._trim_context(context)
assert len(result) == 10, "Should trim oldest when over limit"
```

---

## 5. Test Data Sourcing

### 5.1 Fixtures in conftest.py

Shared fixtures live in `tests/conftest.py`:

- `translator_config` -- TranslatorConfig with test defaults
- `mock_httpx_client` -- Mock httpx.AsyncClient with provider-specific responses
- `sample_idiom_json_file` -- Temporary JSON file with test idiom patterns
- `sample_speaker_profiles` -- List of SpeakerProfile objects
- `sample_translation_context` -- Sample conversation history

### 5.2 Synthetic Test Data

For unit tests, create minimal data that exercises the specific behavior:

```python
# GOOD: Minimal data for the behavior under test
audio_bytes = b"\x00" * 1600  # 100ms of silence at 16kHz mono

# BAD: Loading a full audio recording when you only need silence detection
audio = load_test_recording("full_conversation.webm")
```

### 5.3 Temporary Files

When tests need file I/O, use pytest's `tmp_path` fixture:

```python
def test_idiom_scanner_loads_patterns(tmp_path):
    # Arrange
    pattern_file = tmp_path / "test_idioms.json"
    pattern_file.write_text(json.dumps([
        {"pattern": "echar de menos", "meaning": "to miss someone"}
    ]))

    # Act
    scanner = IdiomScanner(str(tmp_path))

    # Assert
    assert len(scanner.patterns) == 1
```

### 5.4 Inline Data for Small Cases

```python
@pytest.mark.parametrize("text,direction,expected_lang", [
    ("Hola mundo", "es_to_en", "es"),
    ("Hello world", "en_to_es", "en"),
    ("", "es_to_en", None),
])
def test_detect_source_language(text, direction, expected_lang):
    result = detect_language(text, direction)
    assert result == expected_lang
```

---

## 6. Error Path Testing

Every public method that can fail must have tests for its failure modes.

### 6.1 Required Error Path Tests

| Scenario | Test Pattern |
|----------|-------------|
| Empty/None arguments | `pytest.raises(ValueError)` |
| Malformed WebSocket messages | Assert error response sent back to client |
| LLM API failures (timeout, 500, connection refused) | Assert retry behavior and eventual error handling |
| Malformed LLM responses (invalid JSON, missing fields) | Assert graceful degradation |
| Missing external services (Ollama down, no HF_TOKEN) | Assert clear error message and fallback behavior |
| Invalid audio data | Assert no crash, appropriate error logged |

### 6.2 Verify Error Messages Are Useful

```python
# GOOD: Verifies the error message helps diagnose the problem
with pytest.raises(ValueError, match="unsupported direction.*'fr_to_en'"):
    await translator.translate("Bonjour", "fr_to_en", [])

# BAD: Only checks that it raises, not that the message is helpful
with pytest.raises(ValueError):
    await translator.translate("Bonjour", "fr_to_en", [])
```

---

## 7. Performance and Resource Tests

### 7.1 When to Write Performance Tests

- VAD processing latency (must keep up with real-time audio)
- Idiom pattern scanning (must complete in <10ms)
- WebSocket message round-trip time
- Translation pipeline end-to-end latency
- Memory usage during long listening sessions

### 7.2 Performance Test Pattern

```python
@pytest.mark.benchmark
def test_idiom_scanner_performance(sample_idiom_json_file):
    """Idiom scanning must complete in under 10ms for real-time use."""
    scanner = IdiomScanner(str(sample_idiom_json_file.parent))

    # Warm-up
    scanner.scan("Esta es una frase de prueba")

    import time
    start = time.perf_counter()
    for _ in range(100):
        scanner.scan("Me echa de menos cuando estoy en Espana")
    elapsed = (time.perf_counter() - start) / 100

    assert elapsed < 0.01, f"Scan took {elapsed:.4f}s, must be under 10ms"
```

### 7.3 Performance Test Requirements

- **Warm-up**: Always run the operation once before timing.
- **Mark tagging**: Use `@pytest.mark.benchmark` so they can be run separately.
- **Generous thresholds**: Set at 2-3x expected time to avoid CI flakiness.

### 7.4 VRAM Budget Verification

Not directly testable in unit tests, but document expected VRAM usage:

| Component | Expected VRAM | Loaded |
|-----------|--------------|--------|
| WhisperX Small | ~1GB | Always |
| Qwen3 4B Q3_K_M (Ollama) | ~2.5GB | Always |
| Pyannote 3.1 | 0 (CPU) | Always |
| Silero VAD | 0 (CPU) | Always |
| KV cache + overhead | ~1.5GB | Runtime |
| **Total** | **~5GB** | |

Any new GPU component must include a VRAM impact assessment before merging.

---

## 8. Async Test Patterns

### 8.1 Async Method Tests

```python
@pytest.mark.asyncio
async def test_translate_returns_result(translator_config, mock_httpx_client):
    translator = Translator(translator_config)
    translator._client = mock_httpx_client

    result = await translator.translate("Hola", "es_to_en", context=[])

    assert result.translation is not None
    assert result.direction == "es_to_en"
```

### 8.2 Testing Retry and Timeout Behavior

```python
@pytest.mark.asyncio
async def test_retry_on_timeout(translator_config):
    mock_client = AsyncMock()
    mock_client.post.side_effect = [
        httpx.TimeoutException("timeout"),
        httpx.TimeoutException("timeout"),
        mock_success_response(),
    ]
    translator = Translator(translator_config)
    translator._client = mock_client

    result = await translator.translate("Hola", "es_to_en", [])

    assert mock_client.post.call_count == 3
    assert result.translation is not None
```

### 8.3 Testing WebSocket Handlers

```python
@pytest.mark.asyncio
async def test_websocket_text_input(mock_session):
    """Text input should bypass VAD/ASR and go directly to translation."""
    message = {"type": "text_input", "text": "Hola mundo"}

    await handle_text_message(mock_session, message)

    # Verify translation was queued
    assert not mock_session.processing_queue.empty()
```

---

## 9. Mocking Strategy

### 9.1 When to Mock

| Dependency | Mock? | Rationale |
|-----------|-------|-----------|
| Ollama/LM Studio/OpenAI API | Always | External service, unpredictable |
| httpx.AsyncClient | Always | Network I/O |
| WhisperX model | Always in unit tests | GPU-dependent, slow |
| Pyannote diarization | Always in unit tests | CPU-heavy, slow |
| Silero VAD | Always in unit tests | Model loading |
| aiosqlite database | Mock or use in-memory | Depends on test scope |
| IdiomScanner (regex) | Use real | Fast, deterministic, no I/O |
| SpeakerTracker (in-memory) | Use real | Pure logic, no I/O |
| File system (idiom JSON) | Use tmp_path | Pytest provides temp dirs |

### 9.2 Mock Provider Responses

Each LLM provider has a distinct response format. Mock at the HTTP client level:

```python
# Ollama response format
mock_ollama_response = {
    "response": json.dumps({
        "translation": "Hello world",
        "corrected_text": "Hola mundo",
        "confidence": 0.9,
        "flagged_phrases": []
    })
}

# LM Studio response format (OpenAI-compatible)
mock_lmstudio_response = {
    "choices": [{"message": {"content": json.dumps({...})}}]
}

# OpenAI response format
mock_openai_response = {
    "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps({...})}]}]
}
```

### 9.3 Isolation Rules

- A unit test for `Translator` must not fail because `IdiomScanner` is broken. Mock it.
- A unit test must not require Ollama running, GPU available, or network access.
- If a test needs real file I/O or database, it is an integration test -- mark it with `@pytest.mark.integration`.
- If you find yourself mocking more than 4-5 dependencies, the class under test may have too many responsibilities.

---

## 10. Test Organization

### 10.1 File Naming

Test files mirror the source file they test:

| Source File | Test File |
|-------------|-----------|
| `server/pipeline/translator.py` | `tests/pipeline/test_translator.py` |
| `server/services/vocab.py` | `tests/services/test_vocab.py` |
| `server/services/idiom_scanner.py` | `tests/services/test_idiom_scanner.py` |
| `server/routes/api.py` | `tests/routes/test_api.py` |
| `server/routes/websocket.py` | `tests/routes/test_websocket.py` |

### 10.2 Test Categories

Use pytest marks to categorize tests:

```python
@pytest.mark.unit          # Fast, no I/O, no external services
@pytest.mark.integration   # Database, file I/O, multiple components
@pytest.mark.benchmark     # Timing-sensitive performance tests
@pytest.mark.slow          # Tests that take >5 seconds
```

Run subsets:
```bash
# All tests
pytest habla/tests/

# Only unit tests
pytest -m unit habla/tests/

# Skip slow tests
pytest -m "not slow" habla/tests/

# Specific module
pytest habla/tests/pipeline/test_translator.py

# Specific test class
pytest habla/tests/services/test_idiom_scanner.py::TestIdiomScanning
```

### 10.3 Required Coverage by Module Type

| Module Type | Minimum Test Requirements |
|-------------|--------------------------|
| Pipeline (`pipeline/`) | Provider mocking, retry logic, error recovery, JSON parsing, context handling |
| Services (`services/`) | Happy path, edge cases, error paths, data integrity |
| Routes (`routes/`) | Request validation, response format, error status codes, auth (if applicable) |
| Database (`db/`) | CRUD operations, FTS search, schema integrity, concurrent access |
| WebSocket (`routes/websocket.py`) | Message handling, session lifecycle, mode switching, error responses |
| Models (`models/`) | Pydantic validation, serialization round-trip |

---

## 11. Integration and Pipeline Tests

Unit tests prove individual components work. Integration tests prove they work together.

### 11.1 Pipeline Stages

```
Browser Audio (Opus/WebM) -> AudioDecoder (ffmpeg) -> StreamingVADBuffer (Silero VAD)
  -> WhisperX ASR -> Pyannote Diarization -> Ollama/LM Studio/OpenAI Translation
  -> IdiomScanner merge -> WebSocket response -> Browser
```

Each stage produces output consumed by the next. Integration tests verify these handoffs.

### 11.2 Integration Test Pattern

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_text_to_translation():
    """Text input bypasses audio pipeline and produces translation."""
    # Arrange -- mock only the LLM, use real orchestrator logic
    orchestrator = create_test_orchestrator(mock_llm=True)

    # Act
    result = await orchestrator.process_text("Hola mundo", "es_to_en")

    # Assert
    assert result.translation is not None
    assert result.direction == "es_to_en"
    assert result.source_text == "Hola mundo"
```

### 11.3 Contract Tests

When two components share a data boundary, verify the producer's output matches the consumer's expected input:

```python
@pytest.mark.integration
def test_idiom_scanner_output_compatible_with_translation_result():
    """IdiomScanner output must merge cleanly into TranslationResult.flagged_phrases."""
    scanner = IdiomScanner(patterns_dir)
    matches = scanner.scan("Me echa de menos")

    # Verify matches have the fields TranslationResult expects
    for match in matches:
        assert hasattr(match, "phrase")
        assert hasattr(match, "meaning")
        assert hasattr(match, "source")
```

---

## 12. Resource Cleanup

### 12.1 Async Context Managers

Tests that create async resources must clean up:

```python
@pytest.mark.asyncio
async def test_database_query():
    async with aiosqlite.connect(":memory:") as db:
        await db.execute("CREATE TABLE test (id INTEGER)")
        # ... test logic
    # Connection automatically closed
```

### 12.2 Temporary Files

Always use `tmp_path` fixture for automatic cleanup:

```python
def test_idiom_patterns_load(tmp_path):
    pattern_file = tmp_path / "idioms.json"
    pattern_file.write_text('[{"pattern": "test"}]')
    # tmp_path is cleaned up by pytest automatically
```

### 12.3 Mock Cleanup

Reset mocks between tests when using class-level fixtures:

```python
class TestTranslator:
    @pytest.fixture(autouse=True)
    def reset_mocks(self, mock_httpx_client):
        mock_httpx_client.reset_mock()
```

---

## 13. Multi-Provider Testing

Habla supports Ollama, LM Studio, and OpenAI. Tests must cover all configured providers.

### 13.1 Provider-Specific Response Formats

Each provider returns translations in a different HTTP response structure. Test that the translator correctly extracts results from each format:

```python
@pytest.mark.parametrize("provider", ["ollama", "lmstudio", "openai"])
@pytest.mark.asyncio
async def test_translate_parses_provider_response(provider, translator_config):
    translator_config.provider = provider
    translator = Translator(translator_config)
    translator._client = create_mock_client_for(provider)

    result = await translator.translate("Hola", "es_to_en", [])

    assert result.translation == "Hello"
```

### 13.2 Provider Fallback

If the primary provider fails, test that fallback behavior works:

```python
@pytest.mark.asyncio
async def test_provider_fallback_on_failure(translator_config):
    """When primary provider is unreachable, should fall back gracefully."""
    translator = Translator(translator_config)
    translator._client = create_failing_mock()

    # Should raise after exhausting retries, not hang
    with pytest.raises(TranslationError):
        await translator.translate("Hola", "es_to_en", [])
```

---

## 14. Flaky Test Policy

A flaky test is one that sometimes passes and sometimes fails with the same code. Flaky tests erode trust in the entire test suite.

### 14.1 Common Causes and Fixes

| Cause | Fix |
|-------|-----|
| Timing-dependent assertions | Use generous thresholds (2-3x expected) |
| Async race conditions | Use `asyncio.Event` or `asyncio.Queue` for synchronization |
| Test order dependency | Each test creates its own state; use fixtures for shared setup |
| Floating-point comparison | Use `pytest.approx()` with explicit tolerance |
| External service dependency | Always mock external services in unit tests |

### 14.2 Handling Flaky Tests

1. **Investigate immediately** -- a flaky test usually signals a real problem (race condition, resource leak, or environment assumption).
2. **Never delete a flaky test without understanding why it flakes.** Fix the flakiness or the underlying code.
3. **Temporary quarantine**: Mark with `@pytest.mark.skip(reason="Flaky: <issue description>")` and create a task to fix it.

---

## 15. What NOT to Test

### 15.1 Do Not Test

| Category | Example | Why Skip |
|----------|---------|----------|
| Pydantic auto-validation | Field type checking | Pydantic tests this |
| FastAPI routing | "Does GET /api/vocab/ route correctly?" | Framework behavior |
| Third-party libraries | "Does httpx timeout work?" | Library has its own tests |
| Simple property access | `config.ollama_url` | No logic to break |
| Private helper methods directly | `_parse_json_response()` | Test through the public API |

### 15.2 Do Test

Even if something looks simple, test it when:
- It parses LLM output (JSON extraction from varying provider formats)
- It handles WebSocket binary/text frame routing
- It maintains conversation context state
- Getting it wrong would silently produce untranslated or garbled output
- It involves the idiom dedup merge logic

### 15.3 The Payoff Rule

If a test would take 5 minutes to write and would catch a bug that takes 2 hours to diagnose in a live classroom session, write the test. If a test takes 30 minutes and tests something that can only fail if Python's stdlib is broken, skip it.

---

## 16. Test Audit Checklist

Use this checklist when auditing existing tests. For each applicable item, report: PASS, ISSUE (with detail), or N/A.

### 16.1 Coverage

- [ ] Every public method has at least one test
- [ ] Happy path covered with representative data
- [ ] Edge cases tested (empty, zero, boundary, max/min)
- [ ] Error paths tested (None, malformed, missing, timeout, API failure)
- [ ] All three LLM providers tested where applicable

### 16.2 Assertion Quality

- [ ] Assertions verify specific values, not just non-None/non-empty
- [ ] Numerical comparisons use `pytest.approx()` with documented tolerances
- [ ] Exception tests verify type AND message content
- [ ] No assertions that would pass with a trivial/broken implementation

### 16.3 Test Isolation

- [ ] Tests do not depend on execution order
- [ ] Tests clean up temporary files and async resources
- [ ] No shared mutable state between tests
- [ ] Each test creates its own test data (or uses immutable fixtures)
- [ ] External services always mocked in unit tests

### 16.4 Naming and Organization

- [ ] Tests follow `test_<behavior>_<condition>_<outcome>` naming
- [ ] Test file mirrors source file location
- [ ] `@pytest.mark.parametrize` used where same logic applies to multiple inputs
- [ ] Tests grouped logically in classes by component/behavior

### 16.5 Async Correctness

- [ ] All async tests use `@pytest.mark.asyncio`
- [ ] Mocks use `AsyncMock` for async methods (not `MagicMock`)
- [ ] No `asyncio.run()` inside async tests
- [ ] WebSocket tests verify both send and receive paths

### 16.6 Robustness (Anti-Trivial)

- [ ] Tests would fail if the implementation returned wrong values
- [ ] Tests would fail if source/target language was swapped
- [ ] Tests would fail if retry logic was removed
- [ ] Boundary conditions tested AT the boundary (not just near it)
- [ ] No tests that only assert `is not None` or `len() > 0` without value checks

---

## Quick Reference: Running Tests

```bash
# All tests
pytest habla/tests/

# Verbose output
pytest habla/tests/ -v

# Specific test file
pytest habla/tests/pipeline/test_translator.py

# Specific test class
pytest habla/tests/services/test_idiom_scanner.py::TestIdiomScanning

# Specific test method
pytest habla/tests/pipeline/test_translator.py::TestRetryLogic::test_retry_on_timeout

# By mark
pytest -m unit habla/tests/
pytest -m "not slow" habla/tests/

# With coverage
pytest --cov=habla/server habla/tests/

# Stop on first failure
pytest habla/tests/ -x
```
