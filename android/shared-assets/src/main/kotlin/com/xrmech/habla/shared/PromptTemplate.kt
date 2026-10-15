package com.xrmech.habla.shared

/**
 * Python `template.format(**values)` for the subset `shared/prompts.json` uses: `{name}` fields and
 * `{{` / `}}` escapes. Conversions, format specs, attribute/index access and positional fields are
 * rejected instead of ported, and load-time validation rejects any template that uses them, so the
 * phone can never produce different text from the server.
 */
object PromptTemplate {

    /**
     * Substitutes every `{name}` with `values[name]`. Values are inserted verbatim and never
     * re-scanned (a transcript containing `{text}` stays literal); unused keys are ignored.
     *
     * @throws IllegalArgumentException for the first syntax error or unknown placeholder, left to right.
     *   Messages list supplied keys only, never values (values carry user speech).
     */
    fun format(template: String, values: Map<String, String>): String {
        var capacity = template.length
        for (v in values.values) capacity += v.length
        val out = StringBuilder(capacity)
        scan(template, onLiteral = { out.append(it) }) { name, index ->
            val value = values[name]
                ?: throw IllegalArgumentException(
                    "Unknown placeholder '{$name}' at index $index; supplied keys: ${values.keys.sorted()}",
                )
            out.append(value)
        }
        return out.toString()
    }

    /** Distinct field names in first-seen order. Same syntax errors as [format]; never "Unknown placeholder". */
    fun fields(template: String): List<String> {
        val names = LinkedHashSet<String>()
        scan(template, onLiteral = {}) { name, _ -> names.add(name) }
        return names.toList()
    }

    private inline fun scan(
        template: String,
        onLiteral: (Char) -> Unit,
        onField: (name: String, index: Int) -> Unit,
    ) {
        var i = 0
        val n = template.length
        while (i < n) {
            val c = template[i]
            when {
                c == '{' && i + 1 < n && template[i + 1] == '{' -> { onLiteral('{'); i += 2 }
                c == '}' && i + 1 < n && template[i + 1] == '}' -> { onLiteral('}'); i += 2 }
                c == '{' -> {
                    val close = template.indexOf('}', i + 1)
                    if (close < 0) throw IllegalArgumentException("Single '{' encountered in format string at index $i")
                    val name = template.substring(i + 1, close)
                    validateName(name, i)
                    onField(name, i)
                    i = close + 1
                }
                c == '}' -> throw IllegalArgumentException("Single '}' encountered in format string at index $i")
                else -> { onLiteral(c); i++ }
            }
        }
    }

    private fun validateName(name: String, openIndex: Int) {
        val nested = name.indexOf('{')
        if (nested >= 0) {
            throw IllegalArgumentException("Unexpected '{' in field name at index ${openIndex + 1 + nested}")
        }
        if (name.isEmpty() || name.all { it in '0'..'9' }) {
            throw IllegalArgumentException("Positional field '{$name}' is not supported at index $openIndex")
        }
        if (name.any { it == '!' || it == ':' || it == '.' || it == '[' }) {
            throw IllegalArgumentException(
                "Unsupported field syntax '{$name}' at index $openIndex " +
                    "(conversion, format spec, attribute and index fields are not supported)",
            )
        }
    }
}
