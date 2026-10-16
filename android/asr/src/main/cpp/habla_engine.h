#pragma once

// Plain C++ whisper.cpp engine for :asr (design.md sections 3-5, 7). No JNI types here.
//
// One read-only whisper_context plus two whisper_state slots: slot 0 = FINAL (also used for language
// detection, whose encoder pass relies on the state's audio_ctx staying 0), slot 1 = PARTIAL. All
// calls except request_abort() must come from the single ASR thread.
//
// A FINAL transcribe of exactly the samples the preceding detect() saw reuses that detection's mel
// and, through the patched whisper.cpp (whisper_patch.cmake), its encoder output: one encoder pass
// per final instead of two (D042).

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

struct whisper_context;
struct whisper_state;

namespace habla {

constexpr int kSlotFinal = 0;
constexpr int kSlotPartial = 1;
constexpr int kSlotCount = 2;

// Return codes added on top of whisper_full_with_state's own (0, -2 .. -9).
constexpr int kRcNoState = -100;      // handle 0, bad slot, or the slot's state is gone
constexpr int kRcBadArgument = -101;  // invalid argument from the caller (e.g. probs array too small)
constexpr int kRcNativeFailure = -102; // a C++ exception (e.g. std::bad_alloc) was caught at the boundary
constexpr int kRcKvCacheFailed = -7;  // transcribe(): whisper freed the state itself; detect()
                                      // can also return -7 (decode failure) without freeing it
constexpr int kRcAborted = -3;        // a callback stopped the run (whisper itself returns 0 for a
                                      // stop at a window boundary, see habla_engine.cpp)

struct DecodeParams {
    int strategy = 0;          // 0 = WHISPER_SAMPLING_GREEDY, 1 = WHISPER_SAMPLING_BEAM_SEARCH
    int n_threads = 1;
    int beam_size = -1;
    int best_of = 1;
    int audio_ctx = 0;
    bool single_segment = false;
    float temperature = 0.0f;
    float temperature_inc = 0.0f;
    float entropy_thold = 0.0f;
    float logprob_thold = -1.0f;
    float no_speech_thold = 1.0f;
    std::string language;
};

struct Segment {
    std::string text;          // raw UTF-8 bytes, may end inside a multibyte character
    int64_t t0 = 0;            // centiseconds
    int64_t t1 = 0;
    float no_speech_prob = 0.0f;
    float avg_logprob = 0.0f;  // NaN when the segment has no text tokens
    int text_tokens = 0;
};

struct TranscribeResult {
    int rc = 0;
    int64_t wall_ms = 0;
    bool reused_detection = false;  // the FINAL run reused detect()'s mel and encoder output
    std::vector<Segment> segments;  // empty when rc != 0
};

class Engine {
public:
    // Loads the model without a default state and allocates both slot states. partial_model_path,
    // when non-null, loads a second (smaller) model for the PARTIAL slot only (contract change 9,
    // D045); it must share the tokenizer (any multilingual Whisper). Returns nullptr (and frees
    // everything it allocated) when a model or either state cannot be created.
    static Engine* create(const char* model_path, const char* partial_model_path, bool flash_attn);

    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    TranscribeResult transcribe(int slot, const std::vector<float>& samples, const DecodeParams& params);

    // Restricted detection input: log-mel + one decoder step on the FINAL state. Fills probs (size
    // must be >= whisper_lang_max_id() + 1). Returns the argmax language id (>= 0) or a negative rc.
    int detect(const std::vector<float>& samples, int n_threads, float* probs, int n_probs);

    void request_abort(int slot);  // any thread
    void clear_abort(int slot);    // ASR thread only

private:
    Engine() = default;

    static bool valid_slot(int slot) { return slot >= 0 && slot < kSlotCount; }

    // Context per slot; both point at one context unless a partial model was given.
    whisper_context* ctxs_[kSlotCount] = {nullptr, nullptr};
    whisper_state* states_[kSlotCount] = {nullptr, nullptr};
    std::atomic<bool> abort_[kSlotCount] = {false, false};

    // Samples of the last successful detect() whose mel is still in the FINAL state.
    bool detect_mel_valid_ = false;
    size_t detect_n_ = 0;
    uint64_t detect_hash_ = 0;
};

}  // namespace habla
