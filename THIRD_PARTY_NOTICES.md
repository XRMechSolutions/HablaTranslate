# Third-party notices

HablaTranslate is © 2026 Clinton Campbell and released under the [MIT License](LICENSE). It builds
on the open-source components below; each remains under its own licence and copyright.

## Shipped inside the Android app

| Component | Use | Licence |
|-----------|-----|---------|
| [whisper.cpp / ggml](https://github.com/ggml-org/whisper.cpp) (git submodule `android/third_party/whisper.cpp`, patched at build time by `android/asr/src/main/cpp/whisper_patch.cmake`) | On-device speech recognition, CPU and Vulkan builds | MIT, © The ggml authors |
| [Silero VAD](https://github.com/snakers4/silero-vad) (`silero_vad.onnx`) | Voice activity detection | MIT, © Silero Team ([licence file](android/audio/src/main/resources/com/xrmech/habla/audio/silero_vad.LICENSE.txt)) |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Runs the VAD model | MIT, © Microsoft Corporation |
| [Kotlin](https://github.com/JetBrains/kotlin), [kotlinx.coroutines](https://github.com/Kotlin/kotlinx.coroutines), [kotlinx.serialization](https://github.com/Kotlin/kotlinx.serialization) | Language and runtime libraries | Apache-2.0, © JetBrains s.r.o. and contributors |
| [AndroidX](https://developer.android.com/jetpack/androidx) (Jetpack Compose, Material 3, Activity, Core, Lifecycle, Navigation, DataStore, Annotation) | UI and app framework | Apache-2.0, © The Android Open Source Project |
| [Ktor client](https://github.com/ktorio/ktor) | HTTP client for the OpenAI API | Apache-2.0, © JetBrains s.r.o. and contributors |
| [OkHttp](https://github.com/square/okhttp) and [Okio](https://github.com/square/okio) | HTTP transport | Apache-2.0, © Square, Inc. |
| [SQLDelight](https://github.com/cashapp/sqldelight) | Local history database | Apache-2.0, © Square, Inc. |
| [Koin](https://github.com/InsertKoinIO/koin) | Dependency injection | Apache-2.0, © the Koin authors |
| [ML Kit Translation](https://developers.google.com/ml-kit/language/translation) (`com.google.mlkit:translate`) | Free on-device Spanish <-> English translation | Proprietary Google SDK under the [ML Kit Terms of Service](https://developers.google.com/ml-kit/terms); results follow the [Cloud Translation attribution requirements](https://cloud.google.com/translate/attribution) |
| "powered by Google Translate" badge (`android/app/src/main/res/drawable-*/powered_by_google_translate.png`) | Required attribution beside on-device translations | Google brand feature, unaltered from Google's attribution kit; used under Google's brand guidelines, not covered by this repository's MIT licence |

The Apache-2.0 licence text is at <https://www.apache.org/licenses/LICENSE-2.0>. Where a component
ships a NOTICE file, it applies to the distributed app as well.

## Downloaded by the app at runtime (not in this repository)

| Component | Licence |
|-----------|---------|
| ML Kit Spanish and English translation models (~30 MB each), downloaded from Google when the user taps Download in Settings | ML Kit Terms of Service |
| Whisper model weights in ggml format (`shared/asr_models.json`, from the [whisper.cpp model repository](https://huggingface.co/ggerganov/whisper.cpp) at a pinned revision) | MIT, © OpenAI (Whisper) |

## Test data

The audio clips in `shared/testdata/clips/` are public-domain LibriVox recordings; titles, readers
and source links are in [`shared/testdata/clips/sources.json`](shared/testdata/clips/sources.json).

## Server (`habla/`)

The PC server's Python dependencies (WhisperX, faster-whisper, CTranslate2, pyannote.audio, FastAPI
and others in `habla/requirements*.txt`) are installed from PyPI by the user and are not
redistributed here. Each is under its own licence; pyannote's pretrained models additionally require
accepting their conditions on Hugging Face.
