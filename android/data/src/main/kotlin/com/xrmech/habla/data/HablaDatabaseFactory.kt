package com.xrmech.habla.data

import android.content.Context
import androidx.sqlite.db.SupportSQLiteDatabase
import app.cash.sqldelight.db.SqlDriver
import app.cash.sqldelight.driver.android.AndroidSqliteDriver
import com.xrmech.habla.data.db.HablaDatabase

/** Builds the single `habla.db` driver and database. */
object HablaDatabaseFactory {
    /**
     * Call from exactly one Koin `single` and nowhere else: two drivers on one file have separate
     * caches and `asFlow()` listeners on one never see the other's writes (KB quirk 4).
     *
     * Opens lazily; the first query creates or migrates the file. Open and version errors (for
     * example a file written by a newer app) propagate and are never caught to wipe the database.
     */
    fun create(context: Context): HablaDatabase = HablaDatabase(createDriver(context))

    /** The one driver configuration; exposed internally so the device test can read the pragma. */
    internal fun createDriver(context: Context): SqlDriver =
        AndroidSqliteDriver(
            schema = HablaDatabase.Schema,
            context = context.applicationContext,
            name = DataFiles.DATABASE_NAME,
            callback = object : AndroidSqliteDriver.Callback(HablaDatabase.Schema) {
                // SQLite disables FK enforcement per connection; onOpen runs for every connection.
                override fun onOpen(db: SupportSQLiteDatabase) {
                    super.onOpen(db)
                    db.execSQL("PRAGMA foreign_keys = ON")
                }
            },
        )
}
