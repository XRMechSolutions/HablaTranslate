package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.FlaggedPhrase
import java.util.Locale

/** Port of the server's `_merge_idioms`. */
object IdiomMerge {
    /**
     * Pattern hits first (more curated), deduplicated by lowercased phrase, then LLM phrases whose
     * lowercased phrase is unseen, each copied with `source = "llm"`. Inputs are not mutated.
     */
    fun merge(patternHits: List<FlaggedPhrase>, llm: List<FlaggedPhrase>): List<FlaggedPhrase> {
        val seen = HashSet<String>()
        val out = ArrayList<FlaggedPhrase>(patternHits.size + llm.size)
        for (p in patternHits) {
            if (seen.add(p.phrase.lowercase(Locale.ROOT))) out += p
        }
        for (p in llm) {
            if (seen.add(p.phrase.lowercase(Locale.ROOT))) out += p.copy(source = "llm")
        }
        return out
    }
}
