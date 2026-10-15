package com.xrmech.habla.asr.internal

/** The two native library builds (design.md section 1). */
internal enum class CpuVariant(val libName: String) {
    /** armv8.6-a + dotprod + i8mm + fp16: the fast int8 matmul paths on the A56. */
    V86("habla_asr_v86"),

    /** armv8-a (arm64) or SSE4.2 (x86_64 emulator): runs anywhere. */
    BASE("habla_asr_base"),
    ;

    /** `"v86"` / `"base"`, as shown in `nativeInfo`. */
    val label: String get() = libName.removePrefix("habla_asr_")
}

/**
 * Picks the native build for this CPU (design.md section 2). Loading V86 on a core without its
 * features dies with an uncatchable SIGILL, so V86 needs every `Features` line of /proc/cpuinfo (one
 * per core on big.LITTLE kernels) to list dotprod, i8mm and fp16 arithmetic; anything else is BASE.
 */
internal object CpuVariantSelector {
    private val REQUIRED_FEATURES = setOf("asimddp", "i8mm", "asimdhp")
    private val WHITESPACE = Regex("[ \t]+")

    fun select(osArch: String, cpuinfo: String?): CpuVariant {
        if (osArch != "aarch64" || cpuinfo.isNullOrEmpty()) return CpuVariant.BASE
        val featureLines = cpuinfo.lineSequence()
            .map { it.split(':', limit = 2) }
            .filter { it.size == 2 && it[0].trim() == "Features" }
            .map { it[1].trim().split(WHITESPACE).toSet() }
            .toList()
        if (featureLines.isEmpty()) return CpuVariant.BASE
        return if (featureLines.all { it.containsAll(REQUIRED_FEATURES) }) CpuVariant.V86 else CpuVariant.BASE
    }
}
