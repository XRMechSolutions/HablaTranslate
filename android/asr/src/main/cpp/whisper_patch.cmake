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
#
# Edit (ggml-vulkan.cpp, D047): on Samsung Xclipse GPUs, scalar flash attention uses at most 4 rows
# per block. ggml's default of 8 runs the Whisper-small encoder attention at 10 GFLOPS on the A56
# (Xclipse 540, driver 24.0.560): 7.9 s of an 8.7 s encode; 4 rows took 0.79 s. Only the vk build
# compiles this file.

# Applies EDITS (names with <name>_anchor / <name>_patch variables set in the caller) to SRC_FILE and
# writes OUT_FILE only on change, so an unchanged configure does not rebuild it.
macro(habla_apply_edits SRC_FILE OUT_FILE)
    # Re-run the configure (and so these edits) when the submodule's file changes.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${SRC_FILE})
    file(READ ${SRC_FILE} text)
    foreach(edit ${ARGN})
        string(FIND "${text}" "${${edit}_anchor}" first)
        string(FIND "${text}" "${${edit}_anchor}" last REVERSE)
        if (first EQUAL -1 OR NOT first EQUAL last)
            message(FATAL_ERROR "whisper.cpp patch: anchor '${edit}' not found exactly once in ${SRC_FILE}")
        endif()
        string(REPLACE "${${edit}_anchor}" "${${edit}_patch}" text "${text}")
    endforeach()
    set(old "")
    if (EXISTS ${OUT_FILE})
        file(READ ${OUT_FILE} old)
    endif()
    if (NOT old STREQUAL text)
        file(WRITE ${OUT_FILE} "${text}")
    endif()
endmacro()

function(habla_patched_whisper_source SRC_DIR OUT_DIR)
    file(COPY ${SRC_DIR}/CMakeLists.txt ${SRC_DIR}/LICENSE DESTINATION ${OUT_DIR})
    file(COPY ${SRC_DIR}/cmake ${SRC_DIR}/include DESTINATION ${OUT_DIR})
    file(COPY ${SRC_DIR}/ggml DESTINATION ${OUT_DIR} PATTERN ggml-vulkan.cpp EXCLUDE)
    # A standalone whisper.cpp configure writes bindings/javascript/package.json from this template.
    file(COPY ${SRC_DIR}/bindings/javascript/package-tmpl.json DESTINATION ${OUT_DIR}/bindings/javascript)
    file(COPY ${SRC_DIR}/src DESTINATION ${OUT_DIR} PATTERN whisper.cpp EXCLUDE)

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

    habla_apply_edits(${SRC_DIR}/src/whisper.cpp ${OUT_DIR}/src/whisper.cpp state encode pcm setmel)

    set(fa_anchor "static vk_fa_tuning_params get_fa_tuning_params_scalar(")
    set(fa_patch "static vk_fa_tuning_params get_fa_tuning_params_scalar_impl(const vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc);

// habla patch (Samsung Xclipse, D047): at most 4 rows per scalar flash-attention block. The default 8
// collapses throughput about 10x on the Xclipse 540 (register pressure); 4 measured best of 1-16.
static vk_fa_tuning_params get_fa_tuning_params_scalar(const vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc) {
    vk_fa_tuning_params result = get_fa_tuning_params_scalar_impl(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
    if (device->vendor_id == VK_VENDOR_ID_SAMSUNG && result.block_rows > 4) {
        result.block_rows = 4;
    }
    return result;
}

static vk_fa_tuning_params get_fa_tuning_params_scalar_impl(")
    set(vk_src ggml/src/ggml-vulkan/ggml-vulkan.cpp)
    habla_apply_edits(${SRC_DIR}/${vk_src} ${OUT_DIR}/${vk_src} fa)
endfunction()
