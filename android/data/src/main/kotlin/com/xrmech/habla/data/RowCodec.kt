package com.xrmech.habla.data

import com.xrmech.habla.translate.model.CorrectionDetail
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.Mode
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.SerializationException

/**
 * Column values <-> model types for the repositories. A bad stored value degrades to a safe
 * default with a warning, so one corrupt row never hides a whole session.
 */
internal object RowCodec {
    private val flaggedSerializer = ListSerializer(FlaggedPhrase.serializer())

    /** Unknown strings fall back to `ES_TO_EN` (the server `_resolve_effective_direction` fallback). */
    fun direction(value: String): Direction =
        EnumCodec.decodeDirection(value) ?: Direction.ES_TO_EN.also { DataLog.w("unknown direction in row; using es_to_en") }

    fun mode(value: String): Mode =
        EnumCodec.decodeMode(value) ?: Mode.CONVERSATION.also { DataLog.w("unknown mode in row; using conversation") }

    fun encodeFlagged(flagged: List<FlaggedPhrase>): String = DataJson.encodeToString(flaggedSerializer, flagged)

    fun encodeCorrection(correction: CorrectionDetail?): String? =
        correction?.let { DataJson.encodeToString(CorrectionDetail.serializer(), it) }

    /** Null, or corrupt JSON, decodes to an empty list. */
    fun flagged(json: String?): List<FlaggedPhrase> {
        if (json == null) return emptyList()
        return try {
            DataJson.decodeFromString(flaggedSerializer, json)
        } catch (e: SerializationException) {
            DataLog.w("corrupt flagged_json; using []")
            emptyList()
        } catch (e: IllegalArgumentException) {
            DataLog.w("corrupt flagged_json; using []")
            emptyList()
        }
    }

    fun correction(json: String?): CorrectionDetail? {
        if (json == null) return null
        return try {
            DataJson.decodeFromString(CorrectionDetail.serializer(), json)
        } catch (e: SerializationException) {
            DataLog.w("corrupt correction_json; using null")
            null
        } catch (e: IllegalArgumentException) {
            DataLog.w("corrupt correction_json; using null")
            null
        }
    }

    fun bool(value: Long): Boolean = value != 0L

    fun bit(value: Boolean): Long = if (value) 1L else 0L
}

/** Applies the repository flow rule: a storage failure while re-querying terminates with [DataException]. */
internal fun <T> Flow<T>.asDataFlow(what: String): Flow<T> = catch { e ->
    if (e is CancellationException || e is DataException) throw e
    throw DataException("$what failed: ${e.javaClass.simpleName}", e)
}
