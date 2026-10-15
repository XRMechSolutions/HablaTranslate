package com.xrmech.habla.data

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import app.cash.sqldelight.db.QueryResult
import app.cash.sqldelight.db.SqlDriver
import com.xrmech.habla.data.db.HablaDatabase
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/** The real AndroidSqliteDriver: per-connection foreign keys and cascades on a device. */
@RunWith(AndroidJUnit4::class)
class HablaDatabaseFactoryTest {
    private val context: Context = ApplicationProvider.getApplicationContext()
    private lateinit var driver: SqlDriver
    private lateinit var db: HablaDatabase

    @Before
    fun open() {
        context.deleteDatabase(DataFiles.DATABASE_NAME)
        driver = HablaDatabaseFactory.createDriver(context)
        db = HablaDatabase(driver)
    }

    @After
    fun cleanUp() {
        driver.close()
        context.deleteDatabase(DataFiles.DATABASE_NAME)
    }

    @Test
    fun foreignKeysAreOn() {
        val fk = driver.executeQuery(null, "PRAGMA foreign_keys", { cursor ->
            cursor.next()
            QueryResult.Value(cursor.getLong(0))
        }, 0).value
        assertEquals(1L, fk)
    }

    @Test
    fun deletingASessionCascadesItsExchanges() {
        val id = db.sessionQueries.transactionWithResult {
            db.sessionQueries.insertSession("2026-10-15T14:15:02.120Z", "conversation", "es_to_en", "ggml-small-q8_0")
            db.sessionQueries.lastInsertRowId().executeAsOne()
        }
        db.exchangeQueries.insertExchange(
            id, 1, "2026-10-15T14:15:05.000Z", "es_to_en", "hola", null, "hello", 0.9, 0, null, "[]",
            120, "es", 0, null,
        )
        assertEquals(1, db.exchangeQueries.selectExchangesForSession(id).executeAsList().size)
        db.sessionQueries.deleteSession(id)
        assertEquals(0, db.exchangeQueries.selectExchangesForSession(id).executeAsList().size)
    }
}
