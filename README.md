# Habla: live Spanish ⇄ English speech translation

Habla listens to a conversation or a classroom, transcribes each speaker turn, and translates it
with an LLM that sees the recent conversation, so it can fix speech-recognition mistakes and
translate idioms by meaning instead of word for word. It flags idioms, slang and false friends as
it goes and saves them for study.

I built it because Google Translate's live mode kept failing me in Spanish classes: it passes
Spanish through untranslated, mangles idioms, and gives up on quiet or noisy speech.

There are two parts:

- **Server** (`habla/`, working): a Python/FastAPI service on a PC with an NVIDIA GPU. A phone
  or browser streams microphone audio to it over WebSocket.
- **Android app** (`android/`, in progress): the same pipeline running on the phone itself, with
  no server. Only the translation calls leave the device, using the user's own OpenAI key. See
  [Android status](#android-status).

## How it works

```
mic audio -> Silero VAD (speech segments + ~1 s partial snapshots)
  partials: bandpass + gain -> Whisper quick pass -> quick translation (small, cheap model)
  finals:   highpass/lowpass + dynaudnorm -> detect language -> Whisper (beam 5)
            -> idiom pattern scan + LLM translation (JSON) -> merge -> history
```

- **Two-speed output.** Partial transcripts and rough translations show up while someone is still
  talking. The final, corrected version replaces them when they pause.
- **Language per turn.** Each finished segment gets its own language detection (limited to the
  supported languages), so a back-and-forth conversation that alternates Spanish and English is
  transcribed correctly turn by turn.
- **Shared tuning.** Prompts, VAD/ASR constants, idiom patterns, the language list and the cloud
  model table live in [`shared/`](shared/). The server reads them at startup and the Android app
  packages the same files, so a tuning change lands on both.
- **Reference outputs.** `habla/scripts/make_goldens.py` runs public-domain test clips
  (`shared/testdata/clips/`, LibriVox) through the server and records segment boundaries,
  transcripts and detected languages. The Android port is tested against those files.

| Part | Server | Android |
|---|---|---|
| VAD | Silero (PyTorch) | Silero (ONNX Runtime) |
| ASR | WhisperX `small`, int8, CUDA | whisper.cpp `small` (q8_0), NDK |
| Diarization | Pyannote (CPU) | none in v1 |
| Translation | OpenAI by default; LM Studio, Ollama or Claude also work | OpenAI, user's own key |
| Storage | SQLite | SQLite (SQLDelight) |

## Server quick start

Needs Windows, an NVIDIA GPU (8 GB is enough for the default setup), ffmpeg on `PATH`, and either
Python 3.12 or [uv](https://docs.astral.sh/uv/).

```
setup.bat          :: creates habla\.venv, installs the pinned packages, writes habla\.env
start-habla.bat    :: starts the server
```

Then put your keys in `habla\.env` (see [`habla/.env.example`](habla/.env.example)):

- `OPENAI_API_KEY`: translation. Finals use `gpt-5.4-mini` and partials use `gpt-6-luna`. My
  estimate is about $0.01 per minute of conversation; it hasn't been measured yet. Without a key
  the server falls back to a local LM Studio or Ollama model if one is running.
- `HF_TOKEN`: optional, enables speaker labels (Pyannote).

Open `https://<your-tailscale-hostname>:8002` on a phone (Tailscale gives you HTTPS, which the
browser needs for the microphone) or `http://localhost:8002` on the PC.
[SETUP.md](SETUP.md) has the full walkthrough. A Docker setup is in `habla/docker-compose.yml`.

Tests: `cd habla && .venv\Scripts\python.exe -m pytest` (about 1,000 tests, no GPU needed).

## Android status

The app is built and its tests pass on the PC and the emulator. It hasn't been run on a real
phone yet. What it does:

- Runs fully on the phone: audio capture, VAD, Whisper and history. Only translation goes to
  OpenAI.
- **No keys ship in the app.** The user enters their own key in Settings. It's stored encrypted
  with a key from the Android Keystore. CI scans the built APKs and fails if anything shaped like
  a key turns up in a public build. A build with a random dummy key proves the scan catches it.
- Port, don't reinvent: each Kotlin component names the server function it copies and is tested
  against reference outputs the server writes.

Measured so far (PC and emulator, against the server):

| Check | Result |
|---|---|
| VAD segment boundaries vs the server | identical on all test sessions |
| Silero ONNX vs the server's PyTorch model | max probability difference 4e-6 |
| Final audio prep (filters + dynaudnorm port) vs ffmpeg 8.0.1 | bit-exact on 24 speech clips |
| Translator port vs the server (279 golden cases) | all match |
| whisper.cpp `small` q8_0 vs WhisperX, PC, 3 clips | 3.7% WER difference (q5_1: 4.4%) |
| Tests | about 1,000 JVM tests and 34 emulator tests, all passing except one smoke run that waits on regenerated reference files |

Not done yet: speed on the reference phone (Samsung Galaxy A56), a live translation run with a
real key, and the manual checks in [android/ManualTests.md](android/ManualTests.md). If Whisper
`small` can't keep a ~1 s partial cadence on the phone, the fallback is a lighter partial pass,
recorded in `shared/asr_profile.json`.

Build: open a shell in `android/` with JDK 21 and run `gradlew assemblePublicDebug`. The
whisper.cpp source is a git submodule (`git submodule update --init`).

## Things I found and fixed while building the port

Porting forced a careful read of the server, and testing the phone against it turned up real bugs:

- The VAD fed Silero raw 16-bit sample values instead of [-1, 1]. On a 30-second speech clip it
  missed 28% of speech frames (51% with background noise), which split sentences and clipped word
  onsets.
- WhisperX only detects language on its first call and then reuses it, so automatic direction
  never switched languages.
- The `beam_size: 3` setting was never passed to WhisperX, which ran with its default of 5.
- The reference-file generator recorded segment ends at the end of each audio chunk instead of
  the exact frame, up to three frames late. The phone's segmenter got it right and the test
  flagged the reference.

## Limitations

- The server is Windows-first (batch scripts, LM Studio integration). The Python code itself is
  portable, but I haven't tested it on Linux.
- Spanish and English only for now. Adding a language is one entry in `shared/languages.json` plus
  idiom patterns, but nothing else has been tested.
- Speaker labels need Pyannote and a Hugging Face token, and they're only on the server.
- Short utterances (under about 2 s) get almost no loudness normalization; that's how ffmpeg's
  `dynaudnorm` behaves with the current settings.

## Documentation

- [SETUP.md](SETUP.md): installation walkthrough
- [ARCHITECTURE.md](ARCHITECTURE.md): file tree, modules, and where to look for a given problem
- [shared/](shared/): the files both apps read

## License

[MIT](LICENSE). Test clips are public-domain LibriVox recordings
([sources](shared/testdata/clips/sources.json)).
