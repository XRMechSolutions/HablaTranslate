package com.xrmech.habla.translate

import com.xrmech.habla.shared.AsrProfile

/**
 * Recent exchanges and the rolling topic summary for the prompt (port of the server's
 * `context_manager.py` sizes). Thread-safe. `:app` adds only `Ok` outcomes, with
 * `source = result.corrected`.
 */
class ConversationContext(profile: AsrProfile) {
    private val keep = profile.context.recentExchanges
    private val window = profile.context.maxContextExchanges
    private val lock = Any()
    private val exchanges = ArrayDeque<ContextExchange>(keep)

    @Volatile
    var topicSummary: String = ""

    /** Appends [exchange], dropping the oldest beyond `context.recent_exchanges` (10). */
    fun add(exchange: ContextExchange) {
        synchronized(lock) {
            exchanges.addLast(exchange)
            while (exchanges.size > keep) exchanges.removeFirst()
        }
    }

    /** Copy of the last `context.max_context_exchanges` (5) exchanges, oldest first. */
    fun recent(): List<ContextExchange> = synchronized(lock) { exchanges.takeLast(window) }

    internal fun size(): Int = synchronized(lock) { exchanges.size }

    fun reset() {
        synchronized(lock) {
            exchanges.clear()
            topicSummary = ""
        }
    }
}
