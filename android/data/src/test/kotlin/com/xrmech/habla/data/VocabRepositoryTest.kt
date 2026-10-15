package com.xrmech.habla.data

import com.xrmech.habla.translate.model.FlaggedPhrase
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

@OptIn(ExperimentalCoroutinesApi::class)
class VocabRepositoryTest {
    private val t = TestDb()
    private val clock = MutableClock()

    @AfterEach
    fun close() = t.close()

    private fun TestScope.repo() = SqlVocabRepository(t.db, UnconfinedTestDispatcher(testScheduler), clock)

    private fun phrase(term: String, meaning: String = "watch out") =
        FlaggedPhrase(phrase = term, literal = "throw waters", meaning = meaning, type = "slang", region = "mexico")

    @Test
    fun `insert stores every field`() = runTest {
        val repo = repo()
        repo.save(phrase("echar aguas"), null, "¡Echa aguas, güey!")
        val row = repo.all().first().single()
        assertEquals("echar aguas", row.term)
        assertEquals("throw waters", row.literal)
        assertEquals("watch out", row.meaning)
        assertEquals("slang", row.category)
        assertEquals("mexico", row.region)
        assertEquals("¡Echa aguas, güey!", row.sourceSentence)
        assertNull(row.exchangeId)
        assertEquals(clock.instant(), row.createdAt)
        assertEquals(1, row.timesEncountered)
    }

    @Test
    fun `null region becomes universal`() = runTest {
        val repo = repo()
        repo.save(FlaggedPhrase(phrase = "chido", meaning = "cool", region = null), null, null)
        assertEquals("universal", repo.all().first().single().region)
    }

    @Test
    fun `same term in another ASCII case increments and keeps the meaning`() = runTest {
        val repo = repo()
        repo.save(phrase("echar aguas"), null, null)
        repo.save(phrase("echar aguas", meaning = "other meaning"), null, null)
        repo.save(phrase("Echar aguas", meaning = "third meaning"), null, null)
        val row = repo.all().first().single()
        assertEquals(3, row.timesEncountered)
        assertEquals("watch out", row.meaning)
        assertEquals("echar aguas", row.term)
    }

    @Test
    fun `source sentence is replaced only by a non-blank one`() = runTest {
        val repo = repo()
        repo.save(phrase("echar aguas"), null, "first")
        repo.save(phrase("echar aguas"), null, "   ")
        assertEquals("first", repo.all().first().single().sourceSentence)
        repo.save(phrase("echar aguas"), null, null)
        assertEquals("first", repo.all().first().single().sourceSentence)
        repo.save(phrase("echar aguas"), null, "second")
        assertEquals("second", repo.all().first().single().sourceSentence)
        assertEquals(4, repo.all().first().single().timesEncountered)
    }

    @Test
    fun `non-ASCII case stays distinct`() = runTest {
        val repo = repo()
        repo.save(phrase("Ñ"), null, null)
        repo.save(phrase("ñ"), null, null)
        repo.save(phrase("Échale"), null, null)
        repo.save(phrase("échale"), null, null)
        assertEquals(4, repo.all().first().size)
    }

    @Test
    fun `blank term is rejected`() = runTest {
        val repo = repo()
        assertThrows<IllegalArgumentException> { repo.save(phrase("   "), null, null) }
        assertEquals(0, repo.all().first().size)
    }

    @Test
    fun `all is ordered created_at desc then id desc`() = runTest {
        val repo = repo()
        repo.save(phrase("a"), null, null)
        repo.save(phrase("b"), null, null)
        clock.advanceSeconds(1)
        repo.save(phrase("c"), null, null)
        assertEquals(listOf("c", "b", "a"), repo.all().first().map { it.term })
    }

    @Test
    fun `unknown exchange id is a DataException`() = runTest {
        assertThrows<DataException> { repo().save(phrase("x"), 999, null) }
    }
}
