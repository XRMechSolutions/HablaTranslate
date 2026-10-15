package com.xrmech.habla.data

import com.xrmech.habla.translate.model.FlaggedPhrase
import java.time.Instant
import kotlinx.coroutines.flow.Flow

/** Saved phrases (port of the server's `VocabService`). */
interface VocabRepository {
    /**
     * Inserts the phrase, or for an existing term (ASCII case-insensitive) increments its count and
     * replaces `source_sentence` only with a non-blank one. @throws IllegalArgumentException on a blank term.
     */
    suspend fun save(phrase: FlaggedPhrase, exchangeId: Long?, sourceSentence: String?)

    /** Every row, `created_at DESC, id DESC`. */
    fun all(): Flow<List<VocabRow>>
}

data class VocabRow(
    val id: Long,
    val exchangeId: Long?,
    val term: String,
    val literal: String?,
    val meaning: String,
    val category: String,
    val sourceSentence: String?,
    val region: String?,
    val createdAt: Instant,
    val timesEncountered: Int,
)
