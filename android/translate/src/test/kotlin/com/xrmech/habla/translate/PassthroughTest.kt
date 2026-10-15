package com.xrmech.habla.translate

import org.junit.jupiter.api.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PassthroughTest {

    @Test
    fun emptyInputsAreNotPassthrough() {
        assertFalse(Passthrough.isPassthrough("", "hola", "es_to_en"))
        assertFalse(Passthrough.isPassthrough("hola", "", "es_to_en"))
    }

    @Test
    fun punctuationOnlySourceIsNotPassthrough() {
        assertFalse(Passthrough.isPassthrough("¿?!...", "¿?!...", "es_to_en"))
    }

    @Test
    fun identicalTextIsPassthrough() {
        assertTrue(Passthrough.isPassthrough("¿Dónde está la biblioteca?", "¿Dónde está la biblioteca?", "en_to_es"))
    }

    @Test
    fun accentedWordsSurviveNormalization() {
        // "está" and "esta" must stay distinct words, as with Python's Unicode \w.
        assertFalse(Passthrough.isPassthrough("está está está está", "esta esta esta esta", "en_to_es"))
    }

    @Test
    fun overlapBoundaryIsStrict() {
        // 7 of 10 source words in the translation = 0.70 -> not passthrough; 8 of 10 -> passthrough.
        val src = "a1 a2 a3 a4 a5 a6 a7 a8 a9 a10"
        assertFalse(Passthrough.isPassthrough(src, "a1 a2 a3 a4 a5 a6 a7 x y z", "en_to_es"))
        assertTrue(Passthrough.isPassthrough(src, "a1 a2 a3 a4 a5 a6 a7 a8 y z", "en_to_es"))
    }

    @Test
    fun overlapJustBelowAndAboveSeventyPercent() {
        val src = (1..100).joinToString(" ") { "w$it" }
        val tgt69 = (1..69).joinToString(" ") { "w$it" }
        val tgt71 = (1..71).joinToString(" ") { "w$it" }
        assertFalse(Passthrough.isPassthrough(src, tgt69, "en_to_es"))
        assertTrue(Passthrough.isPassthrough(src, tgt71, "en_to_es"))
    }

    @Test
    fun spanishScriptInEnglishOutputOnlyForToEn() {
        val src = "the cat sat on the mat today"
        val tgt = "El niño comió años pequeños ñoño"
        assertTrue(Passthrough.isPassthrough(src, tgt, "es_to_en"))
        assertFalse(Passthrough.isPassthrough(src, tgt, "auto"))
        assertFalse(Passthrough.isPassthrough(src, tgt, "en_to_es"))
    }

    @Test
    fun fewSpanishCharsAreNotEnough() {
        assertFalse(Passthrough.isPassthrough("uno dos tres", "José went to the café", "es_to_en"))
    }

    @Test
    fun caseAndPunctuationIgnored() {
        assertTrue(Passthrough.isPassthrough("Hola, ¿cómo estás?", "hola cómo estás", "es_to_en"))
    }
}
