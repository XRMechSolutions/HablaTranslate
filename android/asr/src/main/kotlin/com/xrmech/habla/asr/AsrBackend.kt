package com.xrmech.habla.asr

import com.xrmech.habla.asr.internal.NativeLibraryLoader

/**
 * Process-wide choice of the native speech backend. The GPU library (Vulkan, D043/D047) is
 * experimental: it exists only in builds made with `-Phabla.vulkan=true`, and the choice takes
 * effect at the first model load in the process, because a loaded native library cannot be
 * swapped. Changing [preferGpu] later applies after the app restarts.
 */
object AsrBackend {
    /** True when this build packages the Vulkan library. */
    val gpuBuilt: Boolean get() = BuildConfig.HABLA_VULKAN

    /** Load the GPU library at the first model load, when [gpuBuilt]; falls back to the CPU otherwise. */
    var preferGpu: Boolean
        get() = NativeLibraryLoader.preferGpu
        set(value) {
            NativeLibraryLoader.preferGpu = value
        }

    /** `vk`, `v86` or `base` once a model has loaded in this process; null before. */
    val loadedVariant: String? get() = NativeLibraryLoader.loadedVariant?.label
}
