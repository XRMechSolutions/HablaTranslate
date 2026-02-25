# Habla Code Audit Checklist

This checklist defines what to review when auditing any module in the Habla codebase.
Report findings directly in console output -- no audit report file needed.

For each applicable category, report: PASS, ISSUE (with detail), or N/A.

---

## 1. Code Structure and Organization

- [ ] Follows project package conventions (`server.pipeline`, `server.services`, `server.routes`, `server.models`, `server.db`)
- [ ] Classes and functions have a single responsibility
- [ ] No dead code, unused variables, or commented-out blocks
- [ ] File and class names are descriptive and follow Python conventions (snake_case modules, PascalCase classes)
- [ ] Imports organized: stdlib, third-party, local (per PEP 8)

## 2. Documentation

- [ ] All public classes and functions have docstrings
- [ ] Parameters documented with types and units where applicable
- [ ] Complex algorithms have inline comments explaining the "why"
- [ ] Pydantic models have field descriptions for non-obvious fields

## 3. Pipeline Fidelity

Skip this section for modules outside the translation pipeline.

- [ ] Audio pipeline stages are correctly ordered: Audio -> VAD -> ASR -> Diarization -> Translation -> Client
- [ ] Partial/streaming transcription path is separate from final pipeline path
- [ ] LLM receives full conversation context (last 10 exchanges + topic summary)
- [ ] Idiom detection runs both pattern DB (fast regex) and LLM, with pattern DB taking priority in dedup
- [ ] Source language is never passed through untranslated in final output
- [ ] Direction toggle (`es_to_en` / `en_to_es`) correctly propagates to all pipeline stages

## 4. Error Handling

- [ ] Inputs validated at system boundaries (WebSocket messages, REST API params, file I/O, external API responses)
- [ ] Custom exceptions used for domain errors where appropriate
- [ ] Exceptions caught at route/handler boundary, not swallowed silently
- [ ] Error messages are informative: what failed, what was expected, context
- [ ] No bare `except Exception` blocks that hide root causes
- [ ] No code paths that silently produce wrong results
- [ ] External service failures (Ollama, LM Studio, OpenAI) handled with retry logic and clear error reporting

## 5. Logging

- [ ] Uses Python `logging` module (not `print()`)
- [ ] Log calls use appropriate levels (ERROR, WARNING, INFO, DEBUG)
- [ ] Log messages include relevant context (input values, session IDs, state)
- [ ] No sensitive data logged (API keys, user audio content)
- [ ] Errors include stack traces where useful

## 6. Performance and VRAM Budget

- [ ] GPU operations (WhisperX ASR, diarization) run in threads via `asyncio.to_thread()`
- [ ] No new GPU-resident models added without VRAM budget check (~5GB target, 12GB max)
- [ ] No blocking I/O on the async event loop
- [ ] Collections pre-sized where count is known
- [ ] No redundant API calls, model loads, or repeated calculations
- [ ] Large audio buffers released after processing, not held indefinitely
- [ ] WebSocket messages are reasonably sized (no sending full audio back to client)

## 7. Thread Safety and Async

- [ ] Shared mutable state protected with locks or avoided entirely
- [ ] `async def` used consistently for I/O-bound operations
- [ ] `asyncio.to_thread()` used for blocking CPU-bound work (ASR, diarization)
- [ ] No mixing of sync and async code that could deadlock
- [ ] Cancellation/shutdown handled gracefully (queue drain, state save)
- [ ] WebSocket session state (`ClientSession`) is per-connection, not shared

## 8. Test Coverage

- [ ] Every public method has at least one test
- [ ] Happy path tested with representative data
- [ ] Edge cases tested: empty inputs, zero-length audio, boundary values
- [ ] Error paths tested: malformed messages, missing services, null inputs, API failures
- [ ] Async methods tested with `@pytest.mark.asyncio`
- [ ] External services (Ollama, LM Studio, OpenAI) mocked at the HTTP client level
- [ ] Test data in `tests/` subdirectories or generated in fixtures
- [ ] See [Testing-Standards.md](Testing-Standards.md) for full testing requirements

## 9. WebSocket Protocol

Skip this section for modules that don't handle WebSocket communication.

- [ ] Binary frames used for audio only, text frames for JSON control/responses
- [ ] All JSON messages include a `type` field for routing
- [ ] Message types match schema definitions in `models/schemas.py`
- [ ] Connection cleanup on disconnect (stop VAD, flush buffers, release resources)
- [ ] Push-to-talk and continuous listening modes both handled correctly
- [ ] Partial transcripts clearly distinguished from final translations in messages

## 10. REST API

Skip this section for non-API modules.

- [ ] Endpoints follow REST conventions (GET for reads, POST for creates, DELETE for removes)
- [ ] Request/response models use Pydantic for validation
- [ ] Appropriate HTTP status codes returned (400 for bad input, 404 for not found, 500 for server errors)
- [ ] Query parameters validated and documented
- [ ] Pagination supported for list endpoints where appropriate

## 11. Database

Skip this section for modules that don't interact with the database.

- [ ] Uses async SQLite via `aiosqlite` (not sync sqlite3)
- [ ] WAL mode enabled for concurrent read/write
- [ ] SQL parameterized (no string interpolation for user input)
- [ ] FTS5 used for full-text search (vocab search)
- [ ] Schema changes are backward-compatible or have migration logic
- [ ] Connections properly closed on shutdown

## 12. Security and Privacy

- [ ] No hardcoded API keys, tokens, or passwords
- [ ] `HF_TOKEN` and other secrets from environment variables only
- [ ] Input sanitized before use in file operations or database queries
- [ ] WebSocket messages validated before processing
- [ ] No CORS misconfigurations exposing the API unnecessarily
- [ ] Audio data stays local (not sent to external services except configured LLM endpoints)

## 13. Configuration

- [ ] All configurable values in `config.py` via Pydantic models
- [ ] Environment variable overrides work (`HF_TOKEN`, `OLLAMA_URL`, `OLLAMA_MODEL`, etc.)
- [ ] Sensible defaults for all config values
- [ ] No magic numbers in code -- constants defined and named

## 14. Project Policy Compliance

- [ ] No emojis in code, comments, or documentation
- [ ] Existing files edited with `Edit` tool, not overwritten with `Write`
- [ ] Docker configuration maintains non-root user, HEALTHCHECK, restart policy
- [ ] VRAM budget respected (see Hardware Constraints in CLAUDE.md)
- [ ] Mobile-first client design (primary access is Android phone over Tailscale)
