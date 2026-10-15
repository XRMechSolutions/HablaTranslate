package com.xrmech.habla.data

import app.cash.sqldelight.db.QueryResult
import app.cash.sqldelight.db.SqlCursor
import app.cash.sqldelight.driver.jdbc.sqlite.JdbcSqliteDriver
import com.xrmech.habla.data.db.HablaDatabase
import java.time.Clock
import java.time.Duration
import java.time.Instant
import java.time.ZoneId
import java.time.ZoneOffset
import java.util.Properties

/** In-memory `habla.db` with FK enforcement on, as `HablaDatabaseFactory` configures it on Android. */
internal class TestDb : AutoCloseable {
    val driver = JdbcSqliteDriver(JdbcSqliteDriver.IN_MEMORY, Properties().apply { put("foreign_keys", "true") })
    val db: HablaDatabase

    init {
        HablaDatabase.Schema.create(driver)
        db = HablaDatabase(driver)
    }

    /** Runs a raw query and maps every row. */
    fun <T> rows(sql: String, map: (SqlCursor) -> T): List<T> =
        driver.executeQuery(null, sql, { cursor ->
            val out = mutableListOf<T>()
            while (cursor.next().value) out += map(cursor)
            QueryResult.Value(out)
        }, 0).value

    fun exec(sql: String) {
        driver.execute(null, sql, 0)
    }

    override fun close() = driver.close()
}

/** A clock tests advance by hand. */
internal class MutableClock(var now: Instant = Instant.parse("2026-10-15T14:15:02.120Z")) : Clock() {
    override fun getZone(): ZoneId = ZoneOffset.UTC

    override fun withZone(zone: ZoneId?): Clock = this

    override fun instant(): Instant = now

    fun advance(duration: Duration) {
        now = now.plus(duration)
    }

    fun advanceSeconds(seconds: Long) = advance(Duration.ofSeconds(seconds))
}
