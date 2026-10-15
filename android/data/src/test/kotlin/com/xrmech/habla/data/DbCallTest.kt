package com.xrmech.habla.data

import java.io.IOException
import java.sql.SQLException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

@OptIn(ExperimentalCoroutinesApi::class)
class DbCallTest {

    @Test
    fun `returns the block result`() = runTest {
        assertEquals(42, dbCall(UnconfinedTestDispatcher(testScheduler), "x") { 42 })
    }

    @Test
    fun `argument, missing-row and cancellation errors pass through`() = runTest {
        val d = UnconfinedTestDispatcher(testScheduler)
        assertThrows<IllegalArgumentException> { dbCall(d, "x") { throw IllegalArgumentException("bad") } }
        assertThrows<NoSuchElementException> { dbCall(d, "x") { throw NoSuchElementException("gone") } }
        assertThrows<CancellationException> { dbCall(d, "x") { throw CancellationException("stop") } }
    }

    @Test
    fun `storage failures become DataException with the cause`() = runTest {
        val d = UnconfinedTestDispatcher(testScheduler)
        val sql = SQLException("constraint failed")
        val e = assertThrows<DataException> { dbCall(d, "insert exchange") { throw sql } }
        assertSame(sql, e.cause)
        assertEquals("insert exchange failed: SQLException", e.message)
        assertThrows<DataException> { dbCall(d, "x") { throw IOException("disk full") } }
        assertThrows<DataException> { dbCall(d, "x") { throw IllegalStateException("closed") } }
    }
}
