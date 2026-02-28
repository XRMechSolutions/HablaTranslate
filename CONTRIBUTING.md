# Contributing to Habla

Thanks for your interest in contributing to Habla! This document covers everything you need to know to report bugs, suggest features, and submit code changes.

## Reporting Bugs

Open a [GitHub Issue](../../issues/new) with:

1. **What happened** vs **what you expected**
2. **Steps to reproduce** (include audio/browser details if relevant)
3. **Environment** — OS, GPU, Python version, browser
4. **Logs** — relevant lines from `data/habla.log` or `data/habla_errors.log`

## Suggesting Features

Open an issue tagged `enhancement`. Describe the use case and why it matters. Check [TASKS.md](TASKS.md) first — it may already be planned.

## Development Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.4+ and 12+ GB VRAM
- ffmpeg installed and in PATH
- Ollama running with `qwen3:4b` pulled
- HuggingFace token (free, for Pyannote diarization)

See [SETUP.md](SETUP.md) for the full walkthrough.

### Install

```bash
cd habla
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Run the Server

```bash
cd habla
uvicorn server.main:app --host 0.0.0.0 --port 8002
```

### Run Tests

```bash
cd habla
pytest                          # standard tests (skips slow + benchmark)
pytest -m slow                  # slow tests only (30s+ operations)
pytest -m benchmark             # benchmark tests (require live services)
pytest -m ""                    # all tests
pytest tests/test_config.py     # single module
```

The test suite has ~668 tests. All must pass before submitting a PR.

## Code Style

- **Python** — follow existing conventions in the codebase. No linter is enforced yet, but keep it clean and consistent.
- **JavaScript** — vanilla JS, no frameworks. Match the existing style in `client/js/`.
- **HTML/CSS** — mobile-first, dark theme. Match `index.html` patterns.

### Guidelines

- Keep changes focused — one feature or fix per PR.
- Don't refactor unrelated code in a feature PR.
- Add tests for new backend functionality. Follow the conventions in [Testing Standards](.dev/docs/Testing-Standards.md).
- Don't add dependencies without discussion. The VRAM budget is tight (~5 GB of 12 GB).
- Avoid GPU-resident additions without checking the [VRAM budget](ARCHITECTURE.md).

## Submitting a Pull Request

1. Fork the repo and create a feature branch from `master`
2. Make your changes
3. Run the full test suite: `cd habla && pytest`
4. Write a clear PR description: what changed and why
5. Reference any related issues

## Architecture Overview

Read [ARCHITECTURE.md](ARCHITECTURE.md) for the file tree, module dependencies, and technology table. The key pipeline flow:

```
Browser Audio (Opus) -> WebSocket -> ffmpeg -> Silero VAD
  -> WhisperX ASR -> Pyannote Diarization -> Ollama LLM -> WebSocket -> Browser
```

## Testing Standards

Before writing tests, read [Testing Standards](.dev/docs/Testing-Standards.md). Key points:

- Test names describe the scenario: `test_<function>_<scenario>_<expected>`
- No trivial pass tests (asserting only that a function returns *something*)
- Mock external services (Ollama, WhisperX) — don't require live services for unit/integration tests
- Use the fixtures in `tests/conftest.py`

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
