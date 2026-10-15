package com.xrmech.habla.asr.internal

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class CpuVariantSelectorTest {
    private fun fixture(name: String): String =
        checkNotNull(javaClass.getResource("/cpuinfo/$name")) { "missing fixture $name" }.readText()

    @Test
    fun `A56 with dotprod i8mm and fp16 on every core picks v86`() {
        assertEquals(CpuVariant.V86, CpuVariantSelector.select("aarch64", fixture("a56.txt")))
    }

    @Test
    fun `Cortex-A53-only phone picks base`() {
        assertEquals(CpuVariant.BASE, CpuVariantSelector.select("aarch64", fixture("a53.txt")))
    }

    @Test
    fun `one cluster missing i8mm picks base`() {
        assertEquals(CpuVariant.BASE, CpuVariantSelector.select("aarch64", fixture("mixed_missing_i8mm.txt")))
    }

    @Test
    fun `x86_64 emulator picks base`() {
        assertEquals(CpuVariant.BASE, CpuVariantSelector.select("x86_64", fixture("x86_64_emulator.txt")))
    }

    @Test
    fun `empty or unreadable cpuinfo picks base`() {
        assertEquals(CpuVariant.BASE, CpuVariantSelector.select("aarch64", fixture("empty.txt")))
        assertEquals(CpuVariant.BASE, CpuVariantSelector.select("aarch64", null))
    }

    @Test
    fun `non-aarch64 never picks v86 even with the features listed`() {
        assertEquals(CpuVariant.BASE, CpuVariantSelector.select("x86_64", fixture("a56.txt")))
    }

    @Test
    fun `library names`() {
        assertEquals("habla_asr_v86", CpuVariant.V86.libName)
        assertEquals("habla_asr_base", CpuVariant.BASE.libName)
        assertEquals("v86", CpuVariant.V86.label)
        assertEquals("base", CpuVariant.BASE.label)
    }
}
