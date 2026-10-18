# Habla: live Spanish–English speech translation

Habla turns spoken conversations into live transcripts and translations, with previews while
someone is speaking and final cards when they pause. It is built for following conversations,
classes and everyday exchanges between Spanish and English.

Choose between a **PC-hosted web application with local LLM support** and a **standalone Android
application with on-device speech recognition and translation**. The Android app does not need
the PC server.

## Built to follow the conversation

Habla grew out of the difficulty of following real Spanish conversations and classes: quiet
speech, accents, incomplete sentences and idioms all make translation harder. The aim is to
preserve what a speaker means while keeping the result fast enough to follow the exchange.

In LLM mode, translation uses recent exchanges and a rolling topic summary alongside the
current transcript. That gives the model context for interpreting an ambiguous phrase,
suggesting a correction to a misheard word, or translating an idiom by its meaning. Provisional
previews help the listener keep up while final processing has more of the utterance to work with.
The simpler offline Android mode provides on-device translation without this LLM context layer.

Better translation depends on the whole path from microphone to displayed result. Habla is
built so that recordings can become repeatable tests, weaknesses can be traced to a processing
stage, and changes can be compared for both quality and speed. The goal is an increasingly
capable translator whose improvements are measured, rather than assumed from a larger model.

## Two ways to use Habla

| | PC + web interface | Standalone Android |
|---|---|---|
| Where it runs | Speech processing on your PC; interface in a desktop or phone browser | Speech processing and interface on your phone |
| Speech recognition | WhisperX, with NVIDIA GPU acceleration | Quantized Whisper through whisper.cpp and the Android NDK |
| Translation | Local LLM through Ollama or LM Studio; optional cloud providers | On-device Google ML Kit; optional OpenAI with your own key |
| Connection | Browser connects to your PC over WebSocket | Local mode works offline after model downloads |
| Speaker labels | Optional Pyannote diarization | Not currently available |
| Best fit | Local LLM experimentation and a richer PC-backed conversation workspace | Portable translation without a server or per-use cloud inference charges |

Both versions support automatic Spanish/English direction, live previews, final translations
and local history. LLM translation can use recent conversation context to suggest transcription
corrections, interpret idioms and identify vocabulary. The on-device Android translator provides
basic translation without those LLM explanations or contextual corrections.

## PC + web interface

The browser captures microphone audio and streams it to your PC. The server detects speech,
transcribes it, translates it and returns results to the web interface. You can use the browser
on the PC itself or connect from a phone over a secured network connection.

**Local LLM mode:** run Ollama or LM Studio on your own machine. Speech recognition and translation
then use your hardware, without a paid translation API. Model choice affects memory use, speed
and translation quality. Cloud translation through OpenAI or Claude is also supported; usage is
billed by the selected provider.

### Get started on Windows

The supported setup uses Windows, Python 3.12 (or uv), ffmpeg and an NVIDIA GPU for accelerated
speech recognition. Allow additional GPU memory for a local LLM; requirements depend on the
model and quantization you choose.

1. Run `setup.bat` from the repository root to prepare the Python environment and dependencies.
2. Configure a translation provider in `habla/.env`. For a local Ollama setup, pull the model
   with `ollama pull qwen3:4b`, keep Ollama running, and set:

   ```dotenv
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=qwen3:4b
   ```

   Alternatively, use `LLM_PROVIDER=lmstudio` with a loaded model and the local API enabled.
   Explicitly choose a local provider: the server configuration otherwise defaults to OpenAI.
3. Run `start-habla.bat`.
4. Open the configured server URL. A local HTTP setup uses `http://localhost:8002`; phone access
   requires HTTPS for browser microphone permission. The setup guide covers Tailscale and HTTPS.

Speaker labels additionally require a Hugging Face token and access to the Pyannote models.
See [SETUP.md](SETUP.md) for installation and provider configuration. A Docker configuration is
also included in [habla/](habla/); the documented desktop workflow is Windows-first.

## Standalone Android

The Android application runs its audio pipeline on the phone: microphone capture, speech
detection, audio preparation and Whisper transcription. It uses quantized models and native
processing to fit mobile hardware, with separate preview and final processing paths.

- **On-device translation:** download the speech model and Spanish/English translation packs,
  then select on-device translation in Settings. No API key or PC server is required. Translation
  is powered by Google Translate through ML Kit.
- **Optional LLM translation:** enter your own OpenAI key for context-aware translation and
  explanations. Transcribed text and conversation context are sent to the provider; audio
  recognition remains on the phone. Provider usage charges apply.
- **Local history:** transcripts and translations are stored on the device. Preview translation
  can use the on-device engine even when final translations use an LLM.

Android has been exercised on a **Samsung Galaxy A56**, including CPU/GPU experiments and live
translation in both directions. In a recorded 42-second mixed-language test on October 16, 2026,
on-device mode produced six translated final cards with **zero cloud calls**. This demonstrates
the local path; it is not a guarantee of latency or accuracy on every phone.

CPU inference remains the default. Vulkan acceleration and smaller preview models are
experimental, device-dependent options. Subscription/gateway integration is in development and
is not a generally available Pro service.

### Build Android from source

Use JDK 21 and the Android SDK/NDK versions configured by the project:

```powershell
git submodule update --init --recursive
cd android
.\gradlew.bat assemblePublicDebug
```

Install the resulting public debug APK from `android/app/build/outputs/apk/public/debug/` and
download the models in Settings. Public builds do not include a developer's provider key.
For distribution, use a properly signed release build; the debug build is for local evaluation.

## How Habla improves over time

The project includes recording, playback, comparison and tuning tools to make development a
repeatable loop: **capture a problem, inspect it, compare a change, then check for regressions**.
This is an engineering workflow, not automatic learning from everyone's conversations.

| Stage | What we can investigate and improve |
|---|---|
| Audio capture and preparation | Quiet or noisy input, gain, filtering and preservation of word beginnings |
| Speech detection and chunking | VAD thresholds, pre-speech padding and pause timing, balancing complete phrases against delay |
| Whisper recognition | Model and decoding choices, quantization, language selection and differences between recognition engines |
| Transcript filtering | Empty or noise-like outputs and clipped word beginnings, while checking that genuine speech is preserved |
| Contextual translation | Prompts, conversation history, topic summaries and idiom handling, with attention to meaning and unsupported corrections |
| Mobile execution | CPU/GPU behavior, preview workloads, queueing and sustained performance on real hardware |

Saved audio and session metadata support replay and inspection. The
[evaluation tools](habla/scripts/) include parameter recommendations for VAD and segmentation,
word-error comparisons, engine comparisons and preparation of labeled recordings for future
fine-tuning experiments. Preparing a dataset is separate from training and validating a new
model; model-generated transcripts need review before they can serve as ground truth.

The Python and Kotlin implementations share model catalogs, language definitions, prompts and
audio profiles. Reference-based tests help catch differences in audio processing, segmentation
and translation behavior as the implementations evolve. The public Android implementation
includes basic transcript filtering and clipped-onset detection. These checks do not establish
that every accepted transcript is correct or eliminate hallucinations.

The aim is to improve quality without making conversations harder to follow. A change that
speeds up recognition but clips a word, loses sentence context or degrades after several minutes
is not an unqualified improvement. [Device benchmarks](android/benchmarks/) and
[manual Android checks](android/ManualTests.md) let us examine those tradeoffs beyond a single
successful demo.

Agreement with another recognizer measures implementation parity, **not accuracy against human
transcripts**. Recognition accuracy also does not by itself measure translation quality: the
translated meaning needs separate review. See [ARCHITECTURE.md](ARCHITECTURE.md) for the processing
flows and module boundaries.

## Current limits

- Spanish and English are the supported language pair.
- Speech recognition and translation can make mistakes; LLM corrections can also be wrong.
- Phone performance depends on hardware, model size, temperature and conversation length.
- On-device translation offers a simpler experience than the contextual LLM path.
- Speaker diarization is currently a server feature.
- Initial setup needs downloads. Cloud translation needs connectivity; the PC web interface
  needs a connection to its server even when the LLM itself runs locally.

## Documentation and license

- [Setup guide](SETUP.md): PC installation and configuration.
- [Architecture](ARCHITECTURE.md): web/server and Android design.
- [Contributing](CONTRIBUTING.md): development guidance.
- [Third-party notices](THIRD_PARTY_NOTICES.md): libraries, models and attribution requirements.

The repository is currently licensed under [MIT](LICENSE), copyright Clinton Campbell.
Third-party components retain their own licenses. Public-domain test recordings are attributed
in the [test-data sources](shared/testdata/clips/sources.json).
