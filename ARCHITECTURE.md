# Habla architecture

Habla has two independent runtimes for live Spanish–English translation: a **PC-hosted service
with a browser interface**, and a **standalone Android app**. They share configuration assets
and evaluation references, but Android does not call the PC server to recognize speech.

## Runtime overview

| Layer | PC + web | Android |
|---|---|---|
| Interface | Mobile-friendly HTML/CSS/JavaScript web application | Kotlin and Jetpack Compose |
| Audio input | Browser microphone, Opus/WebM over WebSocket | Android microphone capture |
| Speech detection | Silero VAD | Silero VAD through ONNX Runtime |
| Speech recognition | WhisperX / faster-whisper, normally CUDA | whisper.cpp / ggml through JNI, quantized models |
| Translation | Ollama, LM Studio, OpenAI or Claude | ML Kit on device, or OpenAI with a user-provided key |
| Persistence | SQLite on the server | SQLite through SQLDelight on the phone |
| Speaker attribution | Optional Pyannote and speaker tracking | Not currently implemented |

## PC service and browser interface

```text
Browser microphone
  -> Opus/WebM over WebSocket
  -> ffmpeg decoding and Silero speech segmentation
  -> audio preparation and WhisperX transcription
  -> optional speaker attribution
  -> local or cloud LLM translation + conversation context
  -> persisted exchange and JSON results
  -> browser transcript, translation and vocabulary interface
```

The browser is an audio-capture and presentation client. Recognition, translation coordination
and storage run on the PC. Local LLMs are reached through Ollama or LM Studio; choosing a cloud
provider sends translation requests to that provider instead. A local LLM uses the owner's
compute and avoids per-request cloud translation charges.

The pipeline separates provisional updates from completed exchanges. During speech, partial
recognition and quick translation provide previews. Final processing produces the completed
card with optional speaker attribution, contextual translation and vocabulary annotations.
Previews are provisional and may change substantially when the final result arrives.

The LLM receives recent conversation context and a rolling topic summary to help interpret
transcripts and idioms. This is a correction aid, not a guarantee of what was actually spoken.
Pattern-based idiom detection complements LLM annotations.

The service uses FastAPI, WebSocket audio/control messages and REST endpoints for sessions,
vocabulary, settings and playback. The coordinator separates audio processing, context,
translation and persistence. Remote browser microphone access requires HTTPS; the documented
phone-to-PC setup uses Tailscale rather than exposing the development server publicly.

### Server module map

Paths below are relative to `habla/server/`.

| Area | Source | Detailed reference |
|---|---|---|
| Startup and configuration | `main.py`, `config.py` | [Deployment](docs/architecture/deployment.md) |
| Pipeline coordination | `pipeline/orchestrator.py`, `pipeline/context_manager.py`, `pipeline/session_manager.py` | [Pipeline](docs/architecture/pipeline-overview.md) |
| Decode, segmentation and recognition | `pipeline/vad_buffer.py`, `pipeline/audio_processor.py` | [Audio](docs/architecture/audio-processing.md) |
| Translation providers and prompts | `pipeline/translator.py`, `models/prompts.py` | [Translation](docs/architecture/translation.md) |
| Idioms and speaker tracking | `services/idiom_scanner.py`, `services/speaker_tracker.py` | [Annotations](docs/architecture/annotation.md) |
| Streaming and REST endpoints | `routes/websocket.py`, `routes/api_*.py` | [WebSocket](docs/architecture/websocket.md), [API](docs/architecture/api.md) |
| History and vocabulary | `db/database.py`, `services/vocab.py` | [Database](docs/architecture/database.md), [Services](docs/architecture/services.md) |
| Local LLM management | `services/lmstudio_manager.py` | [LM Studio](docs/architecture/lmstudio.md) |
| Optional speech output | `services/tts.py` | [Text-to-speech](docs/architecture/tts.md) |
| Recording and playback | `services/audio_recorder.py`, `services/playback.py` | [Audio storage](docs/architecture/audio-storage.md) |
| Browser interface | `../client/` (under `habla/`) | [Client](docs/architecture/client.md) |

## Standalone Android pipeline

```text
Phone microphone
  -> audio preparation and Silero VAD (ONNX Runtime)
  -> whisper.cpp recognition (JNI / NDK)
  -> partial previews and final transcripts
  -> ML Kit on-device translation OR user-selected cloud LLM
  -> Compose interface and local SQLDelight history
```

Audio recognition stays on the phone in both translation modes. Local translation uses
downloaded Spanish/English ML Kit language packs and does not require an LLM or API key.
The cloud option sends transcript/context text to OpenAI using the user's key. Local previews
can remain enabled while final translations use the cloud.

On-device results do not claim LLM-generated corrections, explanations or confidence scores.
The interface distinguishes these results from the richer contextual translation path.
Language packs are downloaded explicitly through Settings rather than during a translation.

### Android module map

| Module | Responsibility |
|---|---|
| [`:shared-assets`](android/shared-assets/) | Typed access to shared profiles, catalogs and language assets |
| [`:audio`](android/audio/) | Audio processing and speech-segmentation logic |
| [`:asr`](android/asr/) | Native speech-recognition integration and model handling |
| [`:translate`](android/translate/) | Translation orchestration, provider clients and on-device interfaces |
| [`:data`](android/data/) | Persistent conversation data |
| [`:app`](android/app/) | Compose UI, microphone lifecycle, pipeline coordination, ML Kit adapter and dependency wiring |

The application module composes the lower layers with Koin. Translation and audio logic are
separated from Android UI concerns; the native recognition layer is isolated behind Kotlin
interfaces. ML Kit's Android implementation is supplied by the app through the translation
module's interface.

### Mobile performance work

The Android implementation uses quantized speech models, native inference and distinct preview
and final workloads. CPU execution is the default. Vulkan GPU acceleration and alternative
preview models are experimental: a faster isolated kernel does not necessarily improve the
whole conversation pipeline or sustained performance.

Galaxy A56 measurements cover device parity, inference sweeps and sustained workloads; see
[android/benchmarks/](android/benchmarks/). End-to-end translation latency includes speech
segmentation, language detection, recognition, queueing and translation, so it should not be
equated with a single inference timing. Results from one device do not establish broad device
compatibility.

## Shared assets and evaluation

[`shared/`](shared/) contains language definitions, idiom patterns, prompts, provider/model
catalogs and audio/recognition profiles. Python reads these assets; Android packages them through
its shared-assets module. The runtimes use different inference libraries but can be compared
against common recordings and reference outputs.

Evaluation separates processing parity, device performance and recognition quality. Matching
Python outputs helps detect porting regressions; it does not establish correctness against
human transcription. [Public-domain test recordings](shared/testdata/clips/sources.json) and
[manual Android checks](android/ManualTests.md) support reproducible evaluation.

## Data boundaries and deployment

- **PC/local LLM:** microphone audio travels from the browser to the configured PC; recognition,
  translation and history remain there when local providers are selected.
- **PC/cloud LLM:** translation text/context additionally travels to the selected provider.
- **Android/local:** recognition, translation and history run on the phone after model downloads.
- **Android/cloud:** recognition remains local; translation text/context goes to the provider.

Downloads and optional external services still require network access. Public Android builds
do not embed provider credentials; user-supplied keys use Android Keystore-backed storage.
Release minification is enabled, but compiled applications should not be treated as secret
containers for credentials or algorithms.

Gateway client support exists for future managed access, including on-device fallback, but
the app's authentication, entitlement and billing integration is not yet a complete released
service. It is not required for the working local mode.

For installation, start with the [README](README.md) and [PC setup guide](SETUP.md). Library,
model and attribution obligations are listed in [third-party notices](THIRD_PARTY_NOTICES.md).
