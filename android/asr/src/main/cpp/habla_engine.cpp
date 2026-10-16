#include "habla_engine.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>

#include <sched.h>

#include "whisper.h"

// Defined by the whisper.cpp patch (whisper_patch.cmake): encodes skipped on this state so far.
int64_t habla_whisper_encoder_cache_hits(const struct whisper_state* state);

namespace habla {

namespace {

// Per-call callback data: the slot's flag plus whether a callback actually acted on it. In v1.9.5 a
// false encoder_begin_callback inside the window loop only `break`s, so whisper_full returns 0 with
// no (or partial) segments; `observed` lets transcribe() report that run as aborted.
struct AbortProbe {
    std::atomic<bool>* flag;
    bool observed;
};

// ggml_abort_callback: true stops the graph at the next node.
bool abort_requested(void* user_data) {
    auto* probe = static_cast<AbortProbe*>(user_data);
    const bool stop = probe->flag->load(std::memory_order_acquire);
    if (stop) probe->observed = true;
    return stop;
}

// encoder_begin_callback: false stops before encoding the next window.
bool encoder_may_begin(whisper_context*, whisper_state*, void* user_data) {
    auto* probe = static_cast<AbortProbe*>(user_data);
    const bool stop = probe->flag->load(std::memory_order_acquire);
    if (stop) probe->observed = true;
    return !stop;
}

// ggml (GGML_OPENMP=OFF) creates its workers with pthread_create from the calling thread, and a new
// thread inherits the caller's CPU affinity. Each compute call pins the caller to the performance
// cores of its allowed set: ggml splits every op evenly across threads and waits for the slowest, so
// one worker on an efficiency core holds the whole pass back. On the A56 an app outside top-app is
// allowed cpus 0-6 (four A520 + three A720); unpinned, 4 threads ran no faster than 2. threads() caps
// the requested count at the pinned cores, since a second thread on a core stalls the same way.
long max_freq_khz(int cpu) {
    char path[96];
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
    FILE* f = std::fopen(path, "r");
    if (f == nullptr) return -1;
    long khz = -1;
    if (std::fscanf(f, "%ld", &khz) != 1) khz = -1;
    std::fclose(f);
    return khz;
}

class PerfCoreAffinity {
public:
    PerfCoreAffinity() {
        if (sched_getaffinity(0, sizeof(saved_), &saved_) != 0) return;
        long freq[CPU_SETSIZE];
        long lowest = -1;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            freq[cpu] = CPU_ISSET(cpu, &saved_) ? max_freq_khz(cpu) : -1;
            if (freq[cpu] > 0 && (lowest < 0 || freq[cpu] < lowest)) lowest = freq[cpu];
        }
        cpu_set_t perf;
        CPU_ZERO(&perf);
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (freq[cpu] > lowest) CPU_SET(cpu, &perf);
        }
        // Single cluster (or unreadable cpufreq): leave the mask alone.
        if (CPU_COUNT(&perf) == 0) return;
        pinned_ = sched_setaffinity(0, sizeof(perf), &perf) == 0;
        if (pinned_) cores_ = CPU_COUNT(&perf);
    }
    int threads(int requested) const { return pinned_ && requested > cores_ ? cores_ : requested; }
    ~PerfCoreAffinity() {
        if (pinned_) sched_setaffinity(0, sizeof(saved_), &saved_);
    }
    PerfCoreAffinity(const PerfCoreAffinity&) = delete;
    PerfCoreAffinity& operator=(const PerfCoreAffinity&) = delete;

private:
    cpu_set_t saved_{};
    bool pinned_ = false;
    int cores_ = 0;
};

// FNV-1a over the raw sample bytes: tells transcribe() whether detect() saw the same audio.
uint64_t samples_hash(const std::vector<float>& samples) {
    uint64_t h = 1469598103934665603ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(samples.data());
    const size_t n = samples.size() * sizeof(float);
    for (size_t i = 0; i < n; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

}  // namespace

Engine* Engine::create(const char* model_path, const char* partial_model_path, bool flash_attn,
                       bool partial_on_cpu) {
    whisper_context_params cparams = whisper_context_default_params();
#ifdef HABLA_USE_GPU
    // habla_asr_vk (D043): ggml picks the first Vulkan device and falls back to the CPU without one.
    cparams.use_gpu = true;
#else
    cparams.use_gpu = false;
#endif
    cparams.flash_attn = flash_attn;

    // Owned from the first allocation, so an early return or an exception (std::bad_alloc from
    // whisper, caught at the JNI boundary) frees whatever was already acquired.
    std::unique_ptr<Engine> engine(new Engine());
    engine->ctxs_[kSlotFinal] = whisper_init_from_file_with_params_no_state(model_path, cparams);
    if (engine->ctxs_[kSlotFinal] == nullptr) {
        return nullptr;
    }
    engine->ctxs_[kSlotPartial] = engine->ctxs_[kSlotFinal];
    whisper_context_params partial_params = cparams;
    if (partial_on_cpu && cparams.use_gpu) {
        partial_params.use_gpu = false;
        if (partial_model_path == nullptr) {
            partial_model_path = model_path;  // a CPU copy of the main model
        }
    }
    if (partial_model_path != nullptr) {
        whisper_context* partial = whisper_init_from_file_with_params_no_state(partial_model_path, partial_params);
        if (partial == nullptr) {
            return nullptr;
        }
        engine->ctxs_[kSlotPartial] = partial;
    }
    for (int slot = 0; slot < kSlotCount; ++slot) {
        engine->states_[slot] = whisper_init_state(engine->ctxs_[slot]);
        if (engine->states_[slot] == nullptr) {
            return nullptr;
        }
    }
    return engine.release();
}

Engine::~Engine() {
    for (auto& state : states_) {
        if (state != nullptr) {
            whisper_free_state(state);
            state = nullptr;
        }
    }
    if (ctxs_[kSlotPartial] != nullptr && ctxs_[kSlotPartial] != ctxs_[kSlotFinal]) {
        whisper_free(ctxs_[kSlotPartial]);
    }
    if (ctxs_[kSlotFinal] != nullptr) {
        whisper_free(ctxs_[kSlotFinal]);
    }
    ctxs_[kSlotFinal] = ctxs_[kSlotPartial] = nullptr;
}

TranscribeResult Engine::transcribe(int slot, const std::vector<float>& samples, const DecodeParams& p) {
    TranscribeResult result;
    if (!valid_slot(slot) || states_[slot] == nullptr) {
        result.rc = kRcNoState;
        return result;
    }
    whisper_state* state = states_[slot];
    whisper_context* ctx = ctxs_[slot];

    const auto strategy = p.strategy == 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;
    whisper_full_params wp = whisper_full_default_params(strategy);
    wp.n_threads = p.n_threads;
    wp.beam_search.beam_size = p.beam_size;
    wp.greedy.best_of = p.best_of;
    wp.audio_ctx = p.audio_ctx;
    wp.single_segment = p.single_segment;
    wp.temperature = p.temperature;
    wp.temperature_inc = p.temperature_inc;
    wp.entropy_thold = p.entropy_thold;
    wp.logprob_thold = p.logprob_thold;
    wp.no_speech_thold = p.no_speech_thold;
    wp.language = p.language.c_str();

    // Fixed for WhisperX parity (design.md section 5).
    wp.no_context = true;
    wp.no_timestamps = true;
    wp.suppress_blank = true;
    wp.suppress_nst = true;
    wp.detect_language = false;
    wp.translate = false;
    wp.token_timestamps = false;
    wp.tdrz_enable = false;
    wp.vad = false;
    wp.print_special = false;
    wp.print_progress = false;
    wp.print_realtime = false;
    wp.print_timestamps = false;
    wp.initial_prompt = nullptr;
    wp.prompt_tokens = nullptr;
    wp.prompt_n_tokens = 0;
    wp.offset_ms = 0;
    wp.duration_ms = 0;
    wp.max_len = 0;
    wp.max_tokens = 0;

    AbortProbe probe{&abort_[slot], false};
    wp.abort_callback = abort_requested;
    wp.abort_callback_user_data = &probe;
    wp.encoder_begin_callback = encoder_may_begin;
    wp.encoder_begin_callback_user_data = &probe;

    // Same samples as the preceding detect(): n_samples = 0 keeps its mel in the state, and the
    // patched encoder then skips the window it already encoded (same offset 0, same audio_ctx 0).
    const bool reuse = p.audio_ctx == 0 && detect_mel_valid_[slot] &&
                       samples.size() == detect_n_[slot] && samples_hash(samples) == detect_hash_[slot];
    detect_mel_valid_[slot] = false;
    const int64_t hits_before = habla_whisper_encoder_cache_hits(state);

    const PerfCoreAffinity affinity;
    wp.n_threads = affinity.threads(p.n_threads);
    const auto start = std::chrono::steady_clock::now();
    int rc = reuse ? whisper_full_with_state(ctx, state, wp, nullptr, 0)
                   : whisper_full_with_state(ctx, state, wp, samples.data(), static_cast<int>(samples.size()));
    result.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    // Counted by the patched encoder, not inferred from the request: true only if an encode was skipped.
    // On rc -7 whisper has already freed the state, so it must not be read.
    result.reused_detection = reuse && rc != kRcKvCacheFailed &&
                              habla_whisper_encoder_cache_hits(state) > hits_before;
    if (rc == 0 && probe.observed) {
        rc = kRcAborted;  // stopped at a window boundary; the text is incomplete
    }
    result.rc = rc;

    if (rc == kRcKvCacheFailed) {
        // whisper_full_with_state already freed this state; freeing it again would be a double free.
        // Null the slot first: if re-creation throws, the slot must not keep the freed pointer.
        states_[slot] = nullptr;
        states_[slot] = whisper_init_state(ctx);
        return result;
    }
    if (rc != 0) {
        return result;
    }

    const whisper_token eot = whisper_token_eot(ctx);
    const int n_segments = whisper_full_n_segments_from_state(state);
    result.segments.reserve(static_cast<size_t>(n_segments));
    for (int i = 0; i < n_segments; ++i) {
        Segment seg;
        const char* text = whisper_full_get_segment_text_from_state(state, i);
        seg.text = text != nullptr ? text : "";
        seg.t0 = whisper_full_get_segment_t0_from_state(state, i);
        seg.t1 = whisper_full_get_segment_t1_from_state(state, i);
        seg.no_speech_prob = whisper_full_get_segment_no_speech_prob_from_state(state, i);

        double plog_sum = 0.0;
        int n_text = 0;
        const int n_tokens = whisper_full_n_tokens_from_state(state, i);
        for (int t = 0; t < n_tokens; ++t) {
            const whisper_token_data data = whisper_full_get_token_data_from_state(state, i, t);
            if (data.id < eot) {
                plog_sum += data.plog;
                ++n_text;
            }
        }
        seg.text_tokens = n_text;
        seg.avg_logprob = n_text > 0 ? static_cast<float>(plog_sum / n_text)
                                     : std::numeric_limits<float>::quiet_NaN();
        result.segments.push_back(std::move(seg));
    }
    return result;
}

int Engine::detect(int slot, const std::vector<float>& samples, int n_threads, float* probs, int n_probs) {
    if (!valid_slot(slot) || states_[slot] == nullptr) {
        return kRcNoState;
    }
    whisper_state* state = states_[slot];
    whisper_context* ctx = ctxs_[slot];
    if (probs == nullptr || n_probs < whisper_lang_max_id() + 1) {
        return kRcBadArgument;
    }
    detect_mel_valid_[slot] = false;
    const PerfCoreAffinity affinity;
    n_threads = affinity.threads(n_threads);
    const int mel_rc = whisper_pcm_to_mel_with_state(
        ctx, state, samples.data(), static_cast<int>(samples.size()), n_threads);
    if (mel_rc != 0) {
        return mel_rc < 0 ? mel_rc : -1;
    }
    const int lang = whisper_lang_auto_detect_with_state(ctx, state, 0, n_threads, probs);
    if (lang >= 0) {
        detect_mel_valid_[slot] = true;
        detect_n_[slot] = samples.size();
        detect_hash_[slot] = samples_hash(samples);
    }
    return lang;
}

void Engine::request_abort(int slot) {
    if (valid_slot(slot)) {
        abort_[slot].store(true, std::memory_order_release);
    }
}

void Engine::clear_abort(int slot) {
    if (valid_slot(slot)) {
        abort_[slot].store(false, std::memory_order_release);
    }
}

}  // namespace habla
