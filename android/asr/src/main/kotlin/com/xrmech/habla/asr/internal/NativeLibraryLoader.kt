package com.xrmech.habla.asr.internal

import java.io.File
import java.io.IOException

/**
 * The only place `System.loadLibrary` is called (design.md section 2). Idempotent and synchronized:
 * chooses the variant with [CpuVariantSelector], falls back from v86 to base once, and caches a
 * failure so every later call rethrows the same link error.
 */
internal object NativeLibraryLoader {
    private var loaded: CpuVariant? = null
    private var failure: UnsatisfiedLinkError? = null

    /** @throws UnsatisfiedLinkError when neither library can be loaded. */
    @Synchronized
    fun load(): CpuVariant {
        loaded?.let { return it }
        failure?.let { throw it }

        val chosen = CpuVariantSelector.select(System.getProperty("os.arch").orEmpty(), readCpuinfo())
        val variant = try {
            System.loadLibrary(chosen.libName)
            chosen
        } catch (first: UnsatisfiedLinkError) {
            if (chosen == CpuVariant.BASE) {
                failure = first
                AsrLog.e("loading ${chosen.libName} failed", first)
                throw first
            }
            AsrLog.w("loading ${chosen.libName} failed, falling back to ${CpuVariant.BASE.libName}", first)
            try {
                System.loadLibrary(CpuVariant.BASE.libName)
                CpuVariant.BASE
            } catch (second: UnsatisfiedLinkError) {
                second.addSuppressed(first)
                failure = second
                AsrLog.e("loading ${CpuVariant.BASE.libName} failed too", second)
                throw second
            }
        }
        loaded = variant
        AsrLog.i("native library ${variant.libName} loaded: ${WhisperNative.systemInfo()}")
        return variant
    }

    private fun readCpuinfo(): String? = try {
        File("/proc/cpuinfo").readText(Charsets.UTF_8)
    } catch (e: IOException) {
        null
    } catch (e: SecurityException) {
        null
    }
}
