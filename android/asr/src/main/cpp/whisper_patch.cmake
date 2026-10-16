# Patched copy of the whisper.cpp submodule for :asr (D042).
#
# The submodule stays pristine. At configure time the parts the build needs are copied into the
# binary dir (one directory per submodule commit, see CMakeLists.txt; local edits inside the
# submodule are not tracked, so never edit it) and src/whisper.cpp is regenerated from the submodule's file with single-line anchored
# edits (the checkout may use CRLF, which makes a unified diff fragile). An anchor that no longer
# matches stops the configure: re-check the edit against the new whisper.cpp before bumping it.
#
# Edit: encoder-output reuse. whisper_encode_internal is wrapped so that a second encode of the same
# mel at the same offset and audio_ctx returns the cached result (embd_enc and the cross-attention KV
# stay in the state). Every new mel (pcm_to_mel, set_mel) and every failed or aborted encode clears
# the cache. With it, language detection followed by whisper_full on the same mel runs the encoder
# once instead of twice. habla_whisper_encoder_cache_hits(state) counts the skipped encodes.

function(habla_patched_whisper_source SRC_DIR OUT_DIR)
    file(COPY ${SRC_DIR}/CMakeLists.txt ${SRC_DIR}/LICENSE DESTINATION ${OUT_DIR})
    file(COPY ${SRC_DIR}/cmake ${SRC_DIR}/include ${SRC_DIR}/ggml DESTINATION ${OUT_DIR})
    # A standalone whisper.cpp configure writes bindings/javascript/package.json from this template.
    file(COPY ${SRC_DIR}/bindings/javascript/package-tmpl.json DESTINATION ${OUT_DIR}/bindings/javascript)
    file(COPY ${SRC_DIR}/src DESTINATION ${OUT_DIR} PATTERN whisper.cpp EXCLUDE)

    # Re-run the configure (and so this edit) when the submodule's file changes.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${SRC_DIR}/src/whisper.cpp)
    file(READ ${SRC_DIR}/src/whisper.cpp text)

    set(state_anchor "    whisper_mel mel;")
    set(state_patch "    whisper_mel mel;

    // habla patch (encoder-output reuse): offset and n_ctx of the encode whose output the state
    // holds for the current mel; -1 = none.
    int habla_enc_offset = -1;
    int habla_enc_n_ctx  = -1;
    int64_t habla_enc_hits = 0; // encodes skipped because the output was already there")

    set(encode_anchor "static bool whisper_encode_internal(")
    set(encode_patch "static bool whisper_encode_internal_impl(
        whisper_context & wctx,
          whisper_state & wstate,
              const int   mel_offset,
              const int   n_threads,
    ggml_abort_callback   abort_callback,
                   void * abort_callback_data);

// habla patch (encoder-output reuse): skip an encode whose output the state already holds.
static bool whisper_encode_internal(
        whisper_context & wctx,
          whisper_state & wstate,
              const int   mel_offset,
              const int   n_threads,
    ggml_abort_callback   abort_callback,
                   void * abort_callback_data) {
    const int n_ctx = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : wctx.model.hparams.n_audio_ctx;
    if (wstate.habla_enc_offset == mel_offset && wstate.habla_enc_n_ctx == n_ctx) {
        if (abort_callback && abort_callback(abort_callback_data)) {
            wstate.habla_enc_offset = -1;
            return false;
        }
        wstate.habla_enc_hits++; // only a hit whose output is actually used
        return true;
    }
    wstate.habla_enc_offset = -1;
    if (!whisper_encode_internal_impl(wctx, wstate, mel_offset, n_threads, abort_callback, abort_callback_data)) {
        return false;
    }
    wstate.habla_enc_offset = mel_offset;
    wstate.habla_enc_n_ctx  = n_ctx;
    return true;
}

// habla patch: lets the engine (and its tests) observe that an encode was actually skipped.
int64_t habla_whisper_encoder_cache_hits(const struct whisper_state * state) {
    return state->habla_enc_hits;
}

static bool whisper_encode_internal_impl(")

    set(pcm_anchor "int whisper_pcm_to_mel_with_state(struct whisper_context * ctx, struct whisper_state * state, const float * samples, int n_samples, int n_threads) {")
    set(pcm_patch "${pcm_anchor}
    state->habla_enc_offset = -1; // habla patch: new mel")

    set(setmel_anchor "    state->mel.n_len     = n_len;")
    set(setmel_patch "    state->habla_enc_offset = -1; // habla patch: new mel
${setmel_anchor}")

    foreach(edit state encode pcm setmel)
        string(FIND "${text}" "${${edit}_anchor}" first)
        string(FIND "${text}" "${${edit}_anchor}" last REVERSE)
        if (first EQUAL -1 OR NOT first EQUAL last)
            message(FATAL_ERROR "whisper.cpp patch: anchor '${edit}' not found exactly once in ${SRC_DIR}/src/whisper.cpp")
        endif()
        string(REPLACE "${${edit}_anchor}" "${${edit}_patch}" text "${text}")
    endforeach()

    # Rewrite only on change so an unchanged configure does not rebuild whisper.cpp.
    set(out ${OUT_DIR}/src/whisper.cpp)
    set(old "")
    if (EXISTS ${out})
        file(READ ${out} old)
    endif()
    if (NOT old STREQUAL text)
        file(WRITE ${out} "${text}")
    endif()
endfunction()
