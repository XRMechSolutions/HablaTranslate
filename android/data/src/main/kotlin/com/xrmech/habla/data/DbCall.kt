package com.xrmech.habla.data

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext

/**
 * Runs blocking database work on [dispatcher] and applies the repository error rules: argument
 * errors (`IllegalArgumentException`), missing rows (`NoSuchElementException`) and cancellation
 * pass through; every other failure (SQLite, I/O) becomes [DataException].
 */
internal suspend inline fun <T> dbCall(
    dispatcher: CoroutineDispatcher,
    what: String,
    crossinline block: () -> T,
): T = withContext(dispatcher) {
    try {
        block()
    } catch (e: CancellationException) {
        throw e
    } catch (e: IllegalArgumentException) {
        throw e
    } catch (e: NoSuchElementException) {
        throw e
    } catch (e: Exception) {
        throw DataException("$what failed: ${e.javaClass.simpleName}", e)
    }
}
