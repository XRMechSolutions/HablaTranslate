package com.xrmech.habla.translate

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Serialized topic-summary updates (port of the server's `_update_topic`). The server lets two
 * concurrent updates start from the same previous summary so the later write discards the
 * earlier one; holding a FIFO mutex for the whole update makes each read the previous result.
 *
 * `:app` launches [update] fire-and-forget in the session scope after each `Ok` final; cancelling
 * that scope (session stop/reset) cancels it, and a cancelled update writes nothing.
 */
class TopicTracker(private val translator: Translator, private val context: ConversationContext) {
    private val mutex = Mutex()

    /** Replaces the summary with the new non-blank one; failures keep the previous summary. */
    suspend fun update(latestSource: String, latestTranslation: String) {
        mutex.withLock {
            val previous = context.topicSummary
            // updateTopicSummary never throws except CancellationException, which propagates
            // before the write below.
            val updated = translator.updateTopicSummary(previous, latestSource, latestTranslation)
            if (updated.isNotBlank()) context.topicSummary = updated
        }
    }
}
