package com.xrmech.habla.translate

import com.xrmech.habla.shared.IdiomPatternDef
import com.xrmech.habla.translate.model.FlaggedPhrase
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class IdiomScannerTest {

    private val assets = TestAssets.assets
    private val scanner = IdiomScanner(assets.idioms)

    @Test
    fun allShippedPatternsCompile() {
        assertEquals(emptyList(), scanner.skipped)
        assertEquals(emptyList(), assets.skippedIdioms)
        assertEquals(assets.idioms.size, scanner.count)
        assertTrue(scanner.count > 0)
    }

    @Test
    fun interfacesScanExample() {
        assertEquals(
            listOf(
                FlaggedPhrase(
                    phrase = "importar un pepino", literal = "to matter a cucumber", meaning = "to not care at all",
                    type = "idiom", region = "universal", source = "pattern_db", saveWorthy = true,
                    spanStart = 3, spanEnd = 20,
                ),
            ),
            scanner.scan("Me IMPORTA un PEPINO"),
        )
    }

    @Test
    fun upperCaseAccentedMatch() {
        val hit = scanner.scan("ESTÁ EN LAS NUBES").single()
        assertEquals("estar en las nubes", hit.phrase)
        assertEquals(0, hit.spanStart)
        assertEquals(17, hit.spanEnd)
    }

    @Test
    fun emojiBeforeIdiomUsesCodePointSpans() {
        val text = "😀😀 Me importa un pepino"
        val hit = scanner.scan(text).single()
        // Two emoji are 4 UTF-16 chars but 2 code points.
        assertEquals(6, hit.spanStart)
        assertEquals(23, hit.spanEnd)
        assertEquals("importa un pepino", text.substring(text.offsetByCodePoints(0, hit.spanStart!!), text.offsetByCodePoints(0, hit.spanEnd!!)))
    }

    @Test
    fun duplicateCanonicalsAcrossFilesEmitOnce() {
        val defs = listOf(
            IdiomPatternDef(pattern = "pan\\s+comido", canonical = "ser pan comido", meaning = "easy"),
            IdiomPatternDef(pattern = "es\\s+pan", canonical = "SER PAN COMIDO", meaning = "dup"),
        )
        val hits = IdiomScanner(defs).scan("Es pan comido")
        assertEquals(1, hits.size)
        assertEquals("easy", hits[0].meaning)
    }

    @Test
    fun badPatternSkippedWithCanonical() {
        val lines = mutableListOf<String>()
        val previous = TranslateLogging.sink
        TranslateLogging.sink = TranslateLog { _, m -> lines += m }
        val s = try {
            IdiomScanner(
                listOf(
                    IdiomPatternDef(pattern = "(?P<x>a)", canonical = "named group", meaning = "m"),
                    IdiomPatternDef(pattern = "ok", canonical = "fine", meaning = "m"),
                ),
            )
        } finally {
            TranslateLogging.sink = previous
        }
        assertEquals(listOf("named group"), s.skipped)
        assertEquals(1, s.count)
        assertTrue(lines.any { "named group" in it })
    }

    @Test
    fun twoIdiomsInOneSentenceInPatternOrder() {
        val hits = scanner.scan("Vas a meter la pata si intentas tomar el pelo.")
        assertEquals(listOf("tomar el pelo", "meter la pata"), hits.map { it.phrase })
    }
}
