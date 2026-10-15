package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.FlaggedPhrase
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class IdiomMergeTest {

    private val hit = FlaggedPhrase(phrase = "importar un pepino", meaning = "not care", source = "pattern_db", spanStart = 3, spanEnd = 20)

    @Test
    fun patternHitsFirstThenUnseenLlmPhrases() {
        val llm = listOf(
            FlaggedPhrase(phrase = "IMPORTAR UN PEPINO", meaning = "dup"),
            FlaggedPhrase(phrase = "tío", meaning = "dude", type = "slang", source = "pattern_db"),
            FlaggedPhrase(phrase = "Tío", meaning = "dup within llm"),
        )
        val merged = IdiomMerge.merge(listOf(hit), llm)
        assertEquals(listOf("importar un pepino", "tío"), merged.map { it.phrase })
        assertEquals(hit, merged[0])
        assertEquals("llm", merged[1].source)
        assertEquals("slang", merged[1].type)
    }

    @Test
    fun duplicatePatternHitsAreDeduplicated() {
        val merged = IdiomMerge.merge(listOf(hit, hit.copy(phrase = "Importar Un Pepino")), emptyList())
        assertEquals(listOf(hit), merged)
    }

    @Test
    fun inputsAreNotMutated() {
        val llm = listOf(FlaggedPhrase(phrase = "x", meaning = "m", source = "other"))
        IdiomMerge.merge(emptyList(), llm)
        assertEquals("other", llm[0].source)
    }

    @Test
    fun emptyInputs() {
        assertEquals(emptyList(), IdiomMerge.merge(emptyList(), emptyList()))
    }
}
