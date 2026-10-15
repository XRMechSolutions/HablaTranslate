package com.xrmech.habla.asr

import com.xrmech.habla.shared.AsrProfile

/**
 * Port of the server's `AudioProcessor._update_detected_language` (plus the reset in
 * `orchestrator.set_direction`): one misdetection cannot flip the session language; a switch needs
 * `language_vote.switch_ratio` of the last `language_vote.window` detections.
 *
 * Not thread-safe: call it from one serial dispatcher (the server calls it under `_asr_lock`).
 */
class LanguageVoter(profile: AsrProfile) {
    private val window = profile.languageVote.window
    private val switchRatio = profile.languageVote.switchRatio
    private val votes = ArrayDeque<String>(window)

    /** Server `_last_detected_language`; null until the first vote and after [reset]. */
    var current: String? = null
        private set

    fun vote(language: String) {
        require(language.isNotBlank()) { "language must not be blank" }
        if (votes.size == window) votes.removeFirst()
        votes.addLast(language)

        val now = current
        if (now == null) {
            current = language
            return
        }
        // Insertion-ordered counts + strict '>' keep the first-seen language on ties, like Python dict + max.
        val counts = LinkedHashMap<String, Int>()
        for (v in votes) counts[v] = (counts[v] ?: 0) + 1
        var majority = ""
        var majorityCount = 0
        for ((lang, count) in counts) {
            if (count > majorityCount) {
                majority = lang
                majorityCount = count
            }
        }
        val ratio = majorityCount.toDouble() / votes.size
        if (majority != now && ratio >= switchRatio) current = majority
    }

    /** Clears votes and [current] (session start, or direction set to a non-auto value). */
    fun reset() {
        votes.clear()
        current = null
    }
}
