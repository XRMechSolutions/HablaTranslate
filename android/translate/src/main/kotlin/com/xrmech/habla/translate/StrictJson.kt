package com.xrmech.habla.translate

/** A JSON value as Python's `json.loads` would produce it. */
internal sealed interface PyJson {
    /** Insertion-ordered; a duplicate key keeps its first position and takes the last value (Python dict). */
    data class Obj(val fields: Map<String, PyJson>) : PyJson
    data class Arr(val items: List<PyJson>) : PyJson
    data class Str(val value: String) : PyJson
    /** [isFloat] is true when the literal had a fraction or exponent (Python `float`, else `int`). */
    data class Num(val text: String, val isFloat: Boolean) : PyJson {
        fun toDouble(): Double = text.toDouble()
    }
    data class Bool(val value: Boolean) : PyJson
    data object Null : PyJson
}

internal class StrictJsonException(message: String) : Exception(message)

/**
 * Nesting deeper than [StrictJson.MAX_DEPTH]. Python raises `RecursionError` here (not a
 * `JSONDecodeError`), which the server's `translate` turns into an error result, so this is not a
 * salvage case.
 */
internal class StrictJsonTooDeepException(message: String) : Exception(message)

/**
 * Decoder with the acceptance of CPython `json.loads(s)` (strict mode): whitespace is only
 * space/tab/CR/LF, raw control characters inside strings are rejected, numbers follow the JSON
 * grammar, trailing data is rejected. One accepted divergence: Python's non-standard `NaN`,
 * `Infinity` and `-Infinity` literals are rejected here.
 */
internal object StrictJson {
    /**
     * Max array/object nesting. CPython 3.12 gives up near 3 000 levels; 1 000 is far above anything a
     * translation contains. Nesting is parsed with an explicit stack, so depth never touches the
     * thread's call stack (a recursive descent overflowed a Linux CI test thread below 1 000).
     */
    const val MAX_DEPTH = 1_000

    fun parse(text: String): PyJson {
        val p = Parser(text)
        p.skipWs()
        val value = p.value()
        p.skipWs()
        if (p.pos != text.length) throw StrictJsonException("Extra data at ${p.pos}")
        return value
    }

    private class Parser(val s: String) {
        var pos = 0

        /** An array or object whose closing bracket has not been read yet. */
        private sealed class Open {
            class ObjB : Open() {
                val fields = LinkedHashMap<String, PyJson>()
                var key = ""
            }

            class ArrB : Open() {
                val items = ArrayList<PyJson>()
            }
        }

        fun skipWs() {
            while (pos < s.length && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r')) pos++
        }

        fun fail(what: String): Nothing = throw StrictJsonException("$what at $pos")

        /** One complete value; arrays and objects nest on an explicit stack ([MAX_DEPTH] levels). */
        fun value(): PyJson {
            val stack = ArrayList<Open>()
            while (true) {
                var v = scalarOrOpen(stack) ?: continue
                // Attach the finished value to its container, then read ',' or the closing bracket.
                while (true) {
                    val top = stack.lastOrNull() ?: return v
                    when (top) {
                        is Open.ArrB -> top.items += v
                        is Open.ObjB -> top.fields[top.key] = v
                    }
                    skipWs()
                    if (pos >= s.length) fail("Expecting ',' delimiter")
                    val c = s[pos++]
                    if (c == (if (top is Open.ArrB) ']' else '}')) {
                        stack.removeAt(stack.size - 1)
                        v = if (top is Open.ArrB) PyJson.Arr(top.items) else PyJson.Obj((top as Open.ObjB).fields)
                        continue
                    }
                    if (c != ',') { pos--; fail("Expecting ',' delimiter") }
                    skipWs()
                    if (top is Open.ObjB) top.key = memberKey()
                    break
                }
            }
        }

        /** A scalar or an empty container, or null after opening a container whose first value comes next. */
        private fun scalarOrOpen(stack: ArrayList<Open>): PyJson? {
            if (pos >= s.length) fail("Expecting value")
            return when (s[pos]) {
                '"' -> PyJson.Str(string())
                '{' -> {
                    open(stack)
                    pos++
                    skipWs()
                    if (pos < s.length && s[pos] == '}') {
                        pos++
                        PyJson.Obj(LinkedHashMap())
                    } else {
                        stack += Open.ObjB().also { it.key = memberKey() }
                        null
                    }
                }
                '[' -> {
                    open(stack)
                    pos++
                    skipWs()
                    if (pos < s.length && s[pos] == ']') {
                        pos++
                        PyJson.Arr(ArrayList())
                    } else {
                        stack += Open.ArrB()
                        null
                    }
                }
                'n' -> literal("null", PyJson.Null)
                't' -> literal("true", PyJson.Bool(true))
                'f' -> literal("false", PyJson.Bool(false))
                else -> number()
            }
        }

        private fun open(stack: ArrayList<Open>) {
            if (stack.size + 1 > MAX_DEPTH) throw StrictJsonTooDeepException("Nesting deeper than $MAX_DEPTH at $pos")
        }

        /** `"key"` then `:`; leaves [pos] at the member's value. */
        private fun memberKey(): String {
            if (pos >= s.length || s[pos] != '"') fail("Expecting property name enclosed in double quotes")
            val key = string()
            skipWs()
            if (pos >= s.length || s[pos] != ':') fail("Expecting ':' delimiter")
            pos++
            skipWs()
            return key
        }

        fun literal(word: String, v: PyJson): PyJson {
            if (!s.startsWith(word, pos)) fail("Expecting value")
            pos += word.length
            return v
        }

        fun number(): PyJson {
            val start = pos
            if (pos < s.length && s[pos] == '-') pos++
            if (pos >= s.length || s[pos] !in '0'..'9') { pos = start; fail("Expecting value") }
            if (s[pos] == '0') pos++ else while (pos < s.length && s[pos] in '0'..'9') pos++
            var isFloat = false
            if (pos + 1 < s.length && s[pos] == '.' && s[pos + 1] in '0'..'9') {
                isFloat = true
                pos++
                while (pos < s.length && s[pos] in '0'..'9') pos++
            }
            if (pos < s.length && (s[pos] == 'e' || s[pos] == 'E')) {
                var q = pos + 1
                if (q < s.length && (s[q] == '+' || s[q] == '-')) q++
                if (q < s.length && s[q] in '0'..'9') {
                    while (q < s.length && s[q] in '0'..'9') q++
                    pos = q
                    isFloat = true
                }
            }
            return PyJson.Num(s.substring(start, pos), isFloat)
        }

        fun string(): String {
            pos++ // opening quote
            val sb = StringBuilder()
            while (true) {
                if (pos >= s.length) fail("Unterminated string")
                val c = s[pos++]
                when {
                    c == '"' -> return sb.toString()
                    c == '\\' -> {
                        if (pos >= s.length) fail("Unterminated string")
                        when (val e = s[pos++]) {
                            '"' -> sb.append('"')
                            '\\' -> sb.append('\\')
                            '/' -> sb.append('/')
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000C')
                            'n' -> sb.append('\n')
                            'r' -> sb.append('\r')
                            't' -> sb.append('\t')
                            'u' -> sb.append(hex4().toChar())
                            else -> { pos--; fail("Invalid \\escape '$e'") }
                        }
                    }
                    c < ' ' -> { pos--; fail("Invalid control character") }
                    else -> sb.append(c)
                }
            }
        }

        fun hex4(): Int {
            if (pos + 4 > s.length) fail("Invalid \\uXXXX escape")
            var v = 0
            for (i in 0 until 4) {
                val d = Character.digit(s[pos + i], 16)
                if (d < 0 || s[pos + i].code > 0x7F) fail("Invalid \\uXXXX escape")
                v = v * 16 + d
            }
            pos += 4
            return v
        }
    }
}
