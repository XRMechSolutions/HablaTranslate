package com.xrmech.habla.data

import app.cash.sqldelight.coroutines.asFlow
import app.cash.sqldelight.coroutines.mapToList
import com.xrmech.habla.data.db.HablaDatabase
import com.xrmech.habla.translate.model.FlaggedPhrase
import java.time.Clock
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow

/**
 * Port of `VocabService.save_from_phrase` / `get_all`. The upsert is select-then-update/insert in one
 * transaction because Android 10's SQLite 3.22 has no UPSERT (DM4). `term` is `COLLATE NOCASE`,
 * which folds ASCII only, exactly like the server's `LOWER(term) = LOWER(?)` (DM3).
 */
class SqlVocabRepository(
    db: HablaDatabase,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val clock: Clock = Clock.systemUTC(),
) : VocabRepository {

    private val queries = db.vocabQueries

    override suspend fun save(phrase: FlaggedPhrase, exchangeId: Long?, sourceSentence: String?) {
        dbCall(dispatcher, "save vocab") {
            require(phrase.phrase.isNotBlank()) { "vocab term is blank" }
            queries.transaction {
                val existing = queries.selectVocabByTerm(phrase.phrase).executeAsOneOrNull()
                if (existing != null) {
                    // Server parity: only the count and a non-blank source sentence change.
                    queries.incrementVocabEncounter(sourceSentence?.ifBlank { null }, existing.id)
                } else {
                    // The term is stored as given (the server does not trim).
                    queries.insertVocab(
                        exchange_id = exchangeId,
                        term = phrase.phrase,
                        literal = phrase.literal,
                        meaning = phrase.meaning,
                        category = phrase.type,
                        source_sentence = sourceSentence,
                        region = phrase.region ?: "universal",
                        created_at = Timestamps.now(clock),
                    )
                }
            }
        }
    }

    override fun all(): Flow<List<VocabRow>> =
        queries.selectAllVocab(::vocabRow).asFlow().mapToList(dispatcher).asDataFlow("read vocab")

    @Suppress("LongParameterList")
    private fun vocabRow(
        id: Long,
        exchangeId: Long?,
        term: String,
        literal: String?,
        meaning: String,
        category: String,
        sourceSentence: String?,
        region: String?,
        createdAt: String,
        timesEncountered: Long,
    ) = VocabRow(
        id = id,
        exchangeId = exchangeId,
        term = term,
        literal = literal,
        meaning = meaning,
        category = category,
        sourceSentence = sourceSentence,
        region = region,
        createdAt = Timestamps.parse(createdAt),
        timesEncountered = timesEncountered.toInt(),
    )
}
