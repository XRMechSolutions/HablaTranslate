package com.xrmech.habla.translate

import com.xrmech.habla.translate.model.CorrectionDetail
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.TranslationResult
import java.util.Locale
import java.util.regex.Pattern

internal sealed interface ParseResult {
    data class Ok(val result: TranslationResult) : ParseResult
    /** The server's pydantic `ValidationError` / `AttributeError` case. */
    data class Invalid(val reason: String) : ParseResult
}

/**
 * Port of the server's `Translator._parse_response` and its helpers (`translator.py`). The
 * server goldens are the arbiter: field rules reproduce Python truthiness plus pydantic v2 lax
 * validation, and anything pydantic would reject becomes [ParseResult.Invalid].
 */
internal object ResponseParser {

    private val RAW_TRANSLATION = Pattern.compile(
        "(?:translat(?:ed|ion))${PyText.WS}*[:=]${PyText.WS}*\"?(.+?)\"?${PyText.WS}*$",
        Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE or Pattern.MULTILINE or Pattern.UNIX_LINES,
    )
    private val BARE_QUOTED = Pattern.compile(
        "^${PyText.WS}*\"(.+)\"${PyText.WS}*$",
        Pattern.DOTALL or Pattern.UNIX_LINES,
    )
    private val BOOL_TRUE = setOf("1", "true", "t", "yes", "y", "on")
    private val BOOL_FALSE = setOf("0", "false", "f", "no", "n", "off")

    fun parse(raw: String, original: String): ParseResult {
        var cleaned = PyText.strip(raw)
        if ("<think>" in cleaned && "</think>" in cleaned) {
            cleaned = PyText.strip(cleaned.substringAfter("</think>"))
        }
        if (cleaned.lowercase(Locale.ROOT).startsWith("thinking")) {
            val brace = cleaned.indexOf('{')
            if (brace != -1) cleaned = cleaned.substring(brace)
        }
        if ("```" in cleaned) {
            cleaned = PyText.strip(cleaned.replace("```json", "").replace("```", ""))
        }

        val data = try {
            StrictJson.parse(extractFirstJson(cleaned))
        } catch (_: StrictJsonException) {
            return ParseResult.Ok(salvage(cleaned, original))
        } catch (e: StrictJsonTooDeepException) {
            return ParseResult.Invalid(e.message ?: "nesting too deep")
        }
        if (data !is PyJson.Obj) {
            return ParseResult.Ok(TranslationResult(corrected = original, translated = PyText.strip(cleaned), confidence = 0.1))
        }
        return try {
            ParseResult.Ok(fromObject(data.fields, original))
        } catch (e: InvalidField) {
            ParseResult.Invalid(e.message ?: "invalid")
        }
    }

    private fun salvage(cleaned: String, original: String): TranslationResult {
        val corrected = extractField(cleaned, "corrected")
        val translated = extractField(cleaned, "translated")
        if (corrected.isNotEmpty() || translated.isNotEmpty()) {
            return TranslationResult(
                corrected = corrected.ifEmpty { original },
                translated = translated.ifEmpty { PyText.strip(cleaned) },
                confidence = 0.3,
            )
        }
        val extracted = extractRawTranslation(cleaned)
        if (extracted.isNotEmpty()) {
            return TranslationResult(corrected = original, translated = extracted, confidence = 0.3)
        }
        return TranslationResult(corrected = original, translated = PyText.strip(cleaned), confidence = 0.1)
    }

    /** Brace-depth scanner with string/escape tracking; identical state machine to the server's. */
    fun extractFirstJson(text: String): String {
        val start = text.indexOf('{')
        if (start == -1) return text
        var inStr = false
        var escape = false
        var depth = 0
        for (i in start until text.length) {
            val ch = text[i]
            if (inStr) {
                if (escape) escape = false
                else if (ch == '\\') escape = true
                else if (ch == '"') inStr = false
            } else {
                when (ch) {
                    '"' -> inStr = true
                    '{' -> depth++
                    '}' -> {
                        depth--
                        if (depth == 0) return text.substring(start, i + 1)
                    }
                }
            }
        }
        return text.substring(start)
    }

    /** Port of `_extract_field`: first `"field": "value"` match, literal `\n` -> space, stripped. */
    fun extractField(text: String, field: String): String {
        val p = Pattern.compile(
            "\"${Pattern.quote(field)}\"${PyText.WS}*:${PyText.WS}*\"(.*?)\"",
            Pattern.DOTALL,
        )
        val m = p.matcher(text)
        if (!m.find()) return ""
        return PyText.strip(m.group(1).replace("\\n", " "))
    }

    /** Port of `_extract_raw_translation`. */
    fun extractRawTranslation(text: String): String {
        val m = RAW_TRANSLATION.matcher(text)
        if (m.find()) return PyText.strip(m.group(1)).trim('"')
        val q = BARE_QUOTED.matcher(text)
        if (q.find()) return PyText.strip(q.group(1))
        return ""
    }

    private class InvalidField(message: String) : Exception(message)

    private fun invalid(message: String): Nothing = throw InvalidField(message)

    private fun fromObject(data: Map<String, PyJson>, original: String): TranslationResult {
        val flagged = flaggedPhrases(data["flagged_phrases"])

        val cdValue = data["correction_detail"]
        val correction = if (cdValue == null || !truthy(cdValue)) {
            null
        } else when (cdValue) {
            is PyJson.Str -> CorrectionDetail(wrong = "", right = "", explanation = cdValue.value)
            is PyJson.Obj -> CorrectionDetail(
                wrong = strField(cdValue.fields, "wrong", ""),
                right = strField(cdValue.fields, "right", ""),
                explanation = strField(cdValue.fields, "explanation", ""),
            )
            else -> invalid("correction_detail is ${typeName(cdValue)}")
        }

        return TranslationResult(
            corrected = orOriginal(data["corrected"], original, "corrected"),
            translated = orOriginal(data["translated"], original, "translated"),
            flaggedPhrases = flagged,
            confidence = confidence(data["confidence"]),
            speakerHint = optStr(data["speaker_hint"], "speaker_hint"),
            isCorrection = data["is_correction"]?.let { bool(it, "is_correction") } ?: false,
            correctionDetail = correction,
        )
    }

    private fun flaggedPhrases(value: PyJson?): List<FlaggedPhrase> {
        val items: List<PyJson> = when (value) {
            null -> return emptyList()
            is PyJson.Arr -> value.items
            // Iterating "" or {} in Python yields nothing; any other str/dict yields non-dict items.
            is PyJson.Str -> if (value.value.isEmpty()) return emptyList() else invalid("flagged_phrases is a string")
            is PyJson.Obj -> if (value.fields.isEmpty()) return emptyList() else invalid("flagged_phrases is an object")
            else -> invalid("flagged_phrases is ${typeName(value)}")
        }
        return items.map { item ->
            val fp = (item as? PyJson.Obj)?.fields ?: invalid("flagged phrase is ${typeName(item)}")
            FlaggedPhrase(
                phrase = strField(fp, "phrase", ""),
                literal = optStr(fp["literal"], "literal"),
                meaning = strField(fp, "meaning", ""),
                type = strField(fp, "type", "idiom"),
                saveWorthy = fp["save_worthy"]?.let { bool(it, "save_worthy") } ?: true,
                source = "llm",
            )
        }
    }

    /** Python truthiness of a decoded JSON value. */
    private fun truthy(v: PyJson): Boolean = when (v) {
        PyJson.Null -> false
        is PyJson.Bool -> v.value
        is PyJson.Str -> v.value.isNotEmpty()
        is PyJson.Num -> v.toDouble() != 0.0
        is PyJson.Arr -> v.items.isNotEmpty()
        is PyJson.Obj -> v.fields.isNotEmpty()
    }

    /** `data.get(key) or original`, then pydantic `str`. */
    private fun orOriginal(v: PyJson?, original: String, name: String): String = when {
        v == null || !truthy(v) -> original
        v is PyJson.Str -> v.value
        else -> invalid("$name is ${typeName(v)}")
    }

    /** `d.get(key, default)` validated as pydantic `str` (null rejected). */
    private fun strField(d: Map<String, PyJson>, key: String, default: String): String = when (val v = d[key]) {
        null -> default
        is PyJson.Str -> v.value
        else -> invalid("$key is ${typeName(v)}")
    }

    /** pydantic `Optional[str]`. */
    private fun optStr(v: PyJson?, name: String): String? = when (v) {
        null, PyJson.Null -> null
        is PyJson.Str -> v.value
        else -> invalid("$name is ${typeName(v)}")
    }

    /** pydantic lax `float`; missing -> 0.5 (`data.get("confidence", 0.5)`). */
    private fun confidence(v: PyJson?): Double = when (v) {
        null -> 0.5
        is PyJson.Num -> v.toDouble()
        is PyJson.Bool -> if (v.value) 1.0 else 0.0
        is PyJson.Str -> PyText.strip(v.value).toDoubleOrNull()?.takeIf { looksNumeric(v.value) }
            ?: invalid("confidence is not a number")
        else -> invalid("confidence is ${typeName(v)}")
    }

    /** Rejects strings Java's `parseDouble` accepts but Python's `float` does not (hex, `f`/`d` suffix). */
    private fun looksNumeric(s: String): Boolean {
        val t = PyText.strip(s)
        return t.isNotEmpty() && t.all { it in '0'..'9' || it == '.' || it == '-' || it == '+' || it == 'e' || it == 'E' }
    }

    /** pydantic lax `bool`. */
    private fun bool(v: PyJson, name: String): Boolean = when (v) {
        is PyJson.Bool -> v.value
        is PyJson.Num -> when (v.toDouble()) {
            0.0 -> false
            1.0 -> true
            else -> invalid("$name is a number other than 0/1")
        }
        is PyJson.Str -> when (PyText.strip(v.value).lowercase(Locale.ROOT)) {
            in BOOL_TRUE -> true
            in BOOL_FALSE -> false
            else -> invalid("$name is not a boolean string")
        }
        else -> invalid("$name is ${typeName(v)}")
    }

    private fun typeName(v: PyJson): String = when (v) {
        PyJson.Null -> "null"
        is PyJson.Bool -> "bool"
        is PyJson.Str -> "string"
        is PyJson.Num -> "number"
        is PyJson.Arr -> "array"
        is PyJson.Obj -> "object"
    }
}
