package com.xrmech.habla.app.pipeline

/**
 * The final worker's queue (design section 3.5). Capacity: at most [maxWaiting] (5, the server's
 * `asyncio.Queue(maxsize=5)`) waiting items plus the one running. Overflow drops the OLDEST waiting
 * item: the server blocks its producer instead, but on the phone blocking would back up into
 * capture, and the oldest segment is the most stale for a live view. Not thread-safe: control
 * dispatcher only.
 */
class FinalQueue(private val maxWaiting: Int = 5) {
    private val waiting = ArrayDeque<MergedSegment>()
    private var running = false

    var finalsDropped = 0
        private set
    var peakDepth = 0
        private set

    /** Waiting + running. */
    val depth: Int get() = waiting.size + if (running) 1 else 0
    val isIdle: Boolean get() = waiting.isEmpty() && !running

    /** Enqueues [item]; returns the dropped oldest waiting item when the queue was full. */
    fun offer(item: MergedSegment): MergedSegment? {
        val dropped = if (waiting.size >= maxWaiting) waiting.removeFirst().also { finalsDropped++ } else null
        waiting.addLast(item)
        peakDepth = maxOf(peakDepth, depth)
        return dropped
    }

    /** Next item to process (marks it running), or null when nothing waits. */
    fun take(): MergedSegment? {
        check(!running) { "previous item still running" }
        val next = waiting.removeFirstOrNull() ?: return null
        running = true
        return next
    }

    /** The running item finished (successfully or not). */
    fun done() {
        running = false
    }

    /** Drops every waiting item (stop timeout); returns how many. */
    fun clearWaiting(): Int {
        val n = waiting.size
        waiting.clear()
        return n
    }
}
