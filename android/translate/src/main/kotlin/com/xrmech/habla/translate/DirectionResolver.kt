package com.xrmech.habla.translate

import com.xrmech.habla.shared.LanguageRegistry
import com.xrmech.habla.translate.model.Direction

/**
 * Port of the server's `_resolve_effective_direction`, plus the voted-language fallback that
 * lives in its `_detect_segment_language`.
 */
object DirectionResolver {
    private val log = Log()

    /**
     * A fixed [setting] is returned unchanged. For AUTO: the first of [detected], [voted] that is
     * a registry key translates into `defaultTarget`, or, when it is the target itself, into the
     * first other registry key. No supported language, or a pair with no [Direction], -> ES_TO_EN.
     */
    fun resolve(setting: Direction, detected: String?, voted: String?, languages: LanguageRegistry): Direction {
        if (setting != Direction.AUTO) return setting
        val lang = listOf(detected, voted).firstOrNull { it != null && it in languages.languages }
            ?: return Direction.ES_TO_EN
        val defaultTarget = languages.defaultTarget
        val name = if (lang != defaultTarget) {
            "${lang}_to_$defaultTarget"
        } else {
            val other = languages.languages.keys.firstOrNull { it != lang } ?: "es"
            "${lang}_to_$other"
        }
        return Direction.fromSerialName(name) ?: run {
            log.warn("No direction for language pair $name; using es_to_en")
            Direction.ES_TO_EN
        }
    }
}
