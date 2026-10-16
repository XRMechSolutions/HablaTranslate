// JNI glue for com.xrmech.habla.asr.internal.WhisperNative (design.md section 3).
//
// Rules: audio is copied with GetFloatArrayRegion (never a critical section: calls take seconds);
// segment text crosses as raw UTF-8 byte[] because a segment can end inside a multibyte character and
// NewStringUTF aborts the VM on invalid input; no C++ exception crosses JNI (all errors are rc).

#include <jni.h>
#include <android/log.h>

#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "habla_engine.h"
#include "whisper.h"

namespace {

constexpr const char* kLogTag = "HablaAsr";

jclass g_native_result_class = nullptr;  // global ref
jmethodID g_native_result_ctor = nullptr;
jclass g_byte_array_class = nullptr;     // global ref, element class of byte[][]

#ifdef NDEBUG
constexpr int kMinPriority = ANDROID_LOG_WARN;
#else
constexpr int kMinPriority = ANDROID_LOG_INFO;
#endif

thread_local int t_last_priority = ANDROID_LOG_INFO;

void log_to_logcat(ggml_log_level level, const char* text, void*) {
    int priority;
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: priority = ANDROID_LOG_DEBUG; break;
        case GGML_LOG_LEVEL_INFO:  priority = ANDROID_LOG_INFO; break;
        case GGML_LOG_LEVEL_WARN:  priority = ANDROID_LOG_WARN; break;
        case GGML_LOG_LEVEL_ERROR: priority = ANDROID_LOG_ERROR; break;
        case GGML_LOG_LEVEL_CONT:  priority = t_last_priority; break;
        default:                   priority = ANDROID_LOG_INFO; break;
    }
    t_last_priority = priority;
    if (priority >= kMinPriority && text != nullptr) {
        __android_log_write(priority, kLogTag, text);
    }
}

habla::Engine* engine_of(jlong handle) {
    return reinterpret_cast<habla::Engine*>(handle);
}

bool copy_samples(JNIEnv* env, jfloatArray samples, std::vector<float>& out) {
    if (samples == nullptr) {
        return false;
    }
    const jsize n = env->GetArrayLength(samples);
    out.resize(static_cast<size_t>(n));
    if (n > 0) {
        env->GetFloatArrayRegion(samples, 0, n, out.data());
    }
    return !env->ExceptionCheck();
}

// Builds NativeResult(rc, wallMs, texts, t0, t1, noSpeechProb, avgLogprob, textTokens). On any
// allocation failure a Java exception is pending and nullptr is returned.
jobject make_result(JNIEnv* env, const habla::TranscribeResult& r) {
    const auto n = static_cast<jsize>(r.rc == 0 ? r.segments.size() : 0);

    jobjectArray texts = env->NewObjectArray(n, g_byte_array_class, nullptr);
    jlongArray t0 = env->NewLongArray(n);
    jlongArray t1 = env->NewLongArray(n);
    jfloatArray no_speech = env->NewFloatArray(n);
    jfloatArray avg_logprob = env->NewFloatArray(n);
    jintArray text_tokens = env->NewIntArray(n);
    if (texts == nullptr || t0 == nullptr || t1 == nullptr || no_speech == nullptr ||
        avg_logprob == nullptr || text_tokens == nullptr) {
        return nullptr;
    }

    if (n > 0) {
        std::vector<jlong> t0v(static_cast<size_t>(n));
        std::vector<jlong> t1v(static_cast<size_t>(n));
        std::vector<jfloat> nsv(static_cast<size_t>(n));
        std::vector<jfloat> alv(static_cast<size_t>(n));
        std::vector<jint> ttv(static_cast<size_t>(n));
        for (jsize i = 0; i < n; ++i) {
            const habla::Segment& seg = r.segments[static_cast<size_t>(i)];
            const auto len = static_cast<jsize>(seg.text.size());
            jbyteArray bytes = env->NewByteArray(len);
            if (bytes == nullptr) {
                return nullptr;
            }
            if (len > 0) {
                env->SetByteArrayRegion(bytes, 0, len, reinterpret_cast<const jbyte*>(seg.text.data()));
            }
            env->SetObjectArrayElement(texts, i, bytes);
            env->DeleteLocalRef(bytes);
            t0v[static_cast<size_t>(i)] = seg.t0;
            t1v[static_cast<size_t>(i)] = seg.t1;
            nsv[static_cast<size_t>(i)] = seg.no_speech_prob;
            alv[static_cast<size_t>(i)] = seg.avg_logprob;
            ttv[static_cast<size_t>(i)] = seg.text_tokens;
        }
        env->SetLongArrayRegion(t0, 0, n, t0v.data());
        env->SetLongArrayRegion(t1, 0, n, t1v.data());
        env->SetFloatArrayRegion(no_speech, 0, n, nsv.data());
        env->SetFloatArrayRegion(avg_logprob, 0, n, alv.data());
        env->SetIntArrayRegion(text_tokens, 0, n, ttv.data());
    }

    return env->NewObject(g_native_result_class, g_native_result_ctor,
                          static_cast<jint>(r.rc), static_cast<jlong>(r.wall_ms),
                          texts, t0, t1, no_speech, avg_logprob, text_tokens,
                          static_cast<jboolean>(r.reused_detection ? JNI_TRUE : JNI_FALSE));
}

jobject make_error_result(JNIEnv* env, int rc) {
    habla::TranscribeResult r;
    r.rc = rc;
    return make_result(env, r);
}

}  // namespace

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
    jclass local = env->FindClass("com/xrmech/habla/asr/internal/NativeResult");
    if (local == nullptr) {
        return JNI_ERR;
    }
    g_native_result_class = static_cast<jclass>(env->NewGlobalRef(local));
    env->DeleteLocalRef(local);
    g_native_result_ctor = env->GetMethodID(g_native_result_class, "<init>", "(IJ[[B[J[J[F[F[IZ)V");
    if (g_native_result_ctor == nullptr) {
        return JNI_ERR;
    }
    jclass bytes = env->FindClass("[B");
    if (bytes == nullptr) {
        return JNI_ERR;
    }
    g_byte_array_class = static_cast<jclass>(env->NewGlobalRef(bytes));
    env->DeleteLocalRef(bytes);

    whisper_log_set(log_to_logcat, nullptr);
    return JNI_VERSION_1_6;
}

JNIEXPORT jlong JNICALL
Java_com_xrmech_habla_asr_internal_WhisperNative_init(JNIEnv* env, jclass, jstring model_path, jboolean flash_attn) {
    if (model_path == nullptr) {
        return 0;
    }
    const char* path = env->GetStringUTFChars(model_path, nullptr);
    if (path == nullptr) {
        return 0;
    }
    habla::Engine* engine = nullptr;
    try {
        engine = habla::Engine::create(path, flash_attn == JNI_TRUE);
    } catch (...) {
        engine = nullptr;
    }
    env->ReleaseStringUTFChars(model_path, path);
    if (engine == nullptr) {
        __android_log_write(ANDROID_LOG_ERROR, kLogTag, "engine init failed (model or state allocation)");
    }
    return reinterpret_cast<jlong>(engine);
}

JNIEXPORT void JNICALL
Java_com_xrmech_habla_asr_internal_WhisperNative_free(JNIEnv*, jclass, jlong handle) {
    delete engine_of(handle);  // delete of nullptr is a no-op
}

JNIEXPORT jint JNICALL
Java_com_xrmech_habla_asr_internal_WhisperNative_langId(JNIEnv* env, jclass, jstring code) {
    if (code == nullptr) {
        return -1;
    }
    const char* c = env->GetStringUTFChars(code, nullptr);
    if (c == nullptr) {
        return -1;
    }
    const int id = whisper_lang_id(c);
    env->ReleaseStringUTFChars(code, c);
    return id;
}

JNIEXPORT jint JNICALL
Java_com_xrmech_habla_asr_internal_WhisperNative_langMaxId(JNIEnv*, jclass) {
    return whisper_lang_max_id();
}

JNIEXPORT jstring JNICALL
Java_com_xrmech_habla_asr_internal_WhisperNative_systemInfo(JNIEnv* env, jclass) {
    // ASCII-only feature flags, safe for NewStringUTF.
    return env->NewStringUTF(whisper_print_system_info());
}

JNIEXPORT jint JNICALL
Java_com_xrmech_habla_asr_internal_WhisperNative_detectLanguage(
        JNIEnv* env, jclass, jlong handle, jfloatArray samples, jint n_threads, jfloatArray probs_out) {
    habla::Engine* engine = engine_of(handle);
    if (engine == nullptr) {
        return habla::kRcNoState;
    }
    if (probs_out == nullptr) {
        return habla::kRcBadArgument;
    }
    try {
        std::vector<float> pcm;
        if (!copy_samples(env, samples, pcm)) {
            return habla::kRcBadArgument;
        }
        const jsize n_probs = env->GetArrayLength(probs_out);
        std::vector<float> probs(static_cast<size_t>(n_probs), 0.0f);
        const int rc = engine->detect(pcm, n_threads, probs.data(), n_probs);
        if (rc >= 0) {
            env->SetFloatArrayRegion(probs_out, 0, n_probs, probs.data());
        }
        return rc;
    } catch (...) {
        return habla::kRcNativeFailure;
    }
}

JNIEXPORT jobject JNICALL
Java_com_xrmech_habla_asr_internal_WhisperNative_transcribe(
        JNIEnv* env, jclass, jlong handle, jint slot, jfloatArray samples, jint strategy, jint n_threads,
        jint beam_size, jint best_of, jint audio_ctx, jboolean single_segment, jfloat temperature,
        jfloat temperature_inc, jfloat entropy_thold, jfloat logprob_thold, jfloat no_speech_thold,
        jstring language) {
    habla::Engine* engine = engine_of(handle);
    if (engine == nullptr) {
        return make_error_result(env, habla::kRcNoState);
    }
    habla::TranscribeResult result;
    try {
        std::vector<float> pcm;
        if (!copy_samples(env, samples, pcm) || language == nullptr) {
            return make_error_result(env, habla::kRcBadArgument);
        }
        habla::DecodeParams p;
        p.strategy = strategy;
        p.n_threads = n_threads;
        p.beam_size = beam_size;
        p.best_of = best_of;
        p.audio_ctx = audio_ctx;
        p.single_segment = single_segment == JNI_TRUE;
        p.temperature = temperature;
        p.temperature_inc = temperature_inc;
        p.entropy_thold = entropy_thold;
        p.logprob_thold = logprob_thold;
        p.no_speech_thold = no_speech_thold;
        const char* lang = env->GetStringUTFChars(language, nullptr);
        if (lang == nullptr) {
            return make_error_result(env, habla::kRcBadArgument);
        }
        p.language = lang;
        env->ReleaseStringUTFChars(language, lang);

        result = engine->transcribe(slot, pcm, p);
    } catch (...) {
        result = habla::TranscribeResult();
        result.rc = habla::kRcNativeFailure;
    }
    if (result.rc != 0) {
        result.segments.clear();
    }
    return make_result(env, result);
}

JNIEXPORT void JNICALL
Java_com_xrmech_habla_asr_internal_WhisperNative_requestAbort(JNIEnv*, jclass, jlong handle, jint slot) {
    habla::Engine* engine = engine_of(handle);
    if (engine != nullptr) {
        engine->request_abort(slot);
    }
}

JNIEXPORT void JNICALL
Java_com_xrmech_habla_asr_internal_WhisperNative_clearAbort(JNIEnv*, jclass, jlong handle, jint slot) {
    habla::Engine* engine = engine_of(handle);
    if (engine != nullptr) {
        engine->clear_abort(slot);
    }
}

}  // extern "C"
