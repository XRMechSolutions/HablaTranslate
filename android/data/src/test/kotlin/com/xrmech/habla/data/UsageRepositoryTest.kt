package com.xrmech.habla.data

import app.cash.turbine.test
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import com.xrmech.habla.translate.model.TokenUsage
import com.xrmech.habla.translate.model.UsageKind
import java.io.File
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir

@OptIn(ExperimentalCoroutinesApi::class)
class UsageRepositoryTest {
    @TempDir
    lateinit var filesDir: File

    private val t = TestDb()
    private val clock = MutableClock()

    @AfterEach
    fun close() = t.close()

    private fun TestScope.repo() = SqlUsageRepository(t.db, UnconfinedTestDispatcher(testScheduler), clock)

    private fun TestScope.history() = SqlHistoryRepository(t.db, filesDir, UnconfinedTestDispatcher(testScheduler), clock)

    private val usage = TokenUsage(inputTokens = 1200, cachedInputTokens = 300, outputTokens = 80, reasoningTokens = 40, model = "gpt-5.4-mini")

    @Test
    fun `record stores the row and the total sums cost`() = runTest {
        val repo = repo()
        val sid = history().startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        repo.record(sid, usage, UsageKind.FINAL, 0.0125)
        repo.record(sid, usage, UsageKind.QUICK, 0.0025)
        assertEquals(0.015, repo.totalCostUsd().first(), 1e-12)
        val row = t.rows("SELECT session_id, at, model, kind, input_tokens, cached_input_tokens, output_tokens FROM usage_log ORDER BY id") {
            listOf(it.getLong(0), it.getString(1), it.getString(2), it.getString(3), it.getLong(4), it.getLong(5), it.getLong(6))
        }.first()
        assertEquals(listOf<Any?>(sid, "2026-10-15T14:15:02.120Z", "gpt-5.4-mini", "final", 1200L, 300L, 80L), row)
    }

    @Test
    fun `kinds are lowercased`() = runTest {
        val repo = repo()
        for (k in UsageKind.entries) repo.record(null, usage, k, 0.0)
        assertEquals(listOf("final", "quick", "topic", "test"), t.rows("SELECT kind FROM usage_log ORDER BY id") { it.getString(0) })
    }

    @Test
    fun `empty total is zero and null session is allowed`() = runTest {
        val repo = repo()
        assertEquals(0.0, repo.totalCostUsd().first())
        repo.record(null, usage, UsageKind.TEST, 0.001)
        assertEquals(0.001, repo.totalCostUsd().first())
    }

    @Test
    fun `invalid cost and tokens are rejected`() = runTest {
        val repo = repo()
        for (bad in listOf(-0.01, Double.NaN, Double.POSITIVE_INFINITY)) {
            assertThrows<IllegalArgumentException> { repo.record(null, usage, UsageKind.FINAL, bad) }
        }
        assertThrows<IllegalArgumentException> { repo.record(null, usage.copy(inputTokens = -1), UsageKind.FINAL, 0.0) }
        assertThrows<IllegalArgumentException> { repo.record(null, usage.copy(cachedInputTokens = -1), UsageKind.FINAL, 0.0) }
        assertThrows<IllegalArgumentException> { repo.record(null, usage.copy(outputTokens = -1), UsageKind.FINAL, 0.0) }
        assertEquals(0.0, repo.totalCostUsd().first())
    }

    @Test
    fun `unknown session is a DataException`() = runTest {
        assertThrows<DataException> { repo().record(555, usage, UsageKind.FINAL, 0.0) }
    }

    @Test
    fun `total is unchanged after deleting the session`() = runTest {
        val repo = repo()
        val history = history()
        val sid = history.startSession(Mode.CONVERSATION, Direction.AUTO, "m")
        repo.record(sid, usage, UsageKind.FINAL, 0.31)
        history.deleteSession(sid)
        assertEquals(0.31, repo.totalCostUsd().first())
    }

    @Test
    fun `totalCostUsd re-emits on insert`() = runTest {
        val repo = repo()
        repo.totalCostUsd().test {
            assertEquals(0.0, awaitItem())
            repo.record(null, usage, UsageKind.FINAL, 0.5)
            assertEquals(0.5, awaitItem())
            repo.record(null, usage, UsageKind.FINAL, 0.25)
            assertEquals(0.75, awaitItem())
        }
    }
}
