package com.xrmech.habla.asr.internal

import java.io.File
import java.io.IOException

/**
 * The only place `System.loadLibrary` is called (design.md section 2). Idempotent and synchronized:
 * chooses the variant with [CpuVariantSelector], falls back from v86 to base once, and caches a
 * failure so every later call rethrows the same link error.
 */
internal object NativeLibraryLoader {
    /**
     * Experimental (D043): try [CpuVariant.VK] before V86. Read once, at the first [load]; a missing
     * or unloadable vk library falls back to V86 as if this were false.
     */
    @Volatile
    var preferGpu: Boolean = false

    private var loaded: CpuVariant? = null
    private var failure: UnsatisfiedLinkError? = null

    /** @throws UnsatisfiedLinkError when neither library can be loaded. */
    @Synchronized
    fun load(): CpuVariant {
        loaded?.let { return it }
        failure?.let { throw it }

        val chosen = CpuVariantSelector.select(System.getProperty("os.arch").orEmpty(), readCpuinfo())
        if (preferGpu && chosen == CpuVariant.V86) {
            try {
                System.loadLibrary(CpuVariant.VK.libName)
                loaded = CpuVariant.VK
                AsrLog.i("native library ${CpuVariant.VK.libName} loaded: ${WhisperNative.systemInfo()}")
                return CpuVariant.VK
            } catch (e: UnsatisfiedLinkError) {
                AsrLog.w("loading ${CpuVariant.VK.libName} failed, using ${chosen.libName}", e)
            }
        }
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
