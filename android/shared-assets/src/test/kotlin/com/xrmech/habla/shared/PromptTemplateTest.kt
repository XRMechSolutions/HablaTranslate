package com.xrmech.habla.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class PromptTemplateTest {

    private fun error(template: String, values: Map<String, String> = emptyMap()): String =
        assertFailsWith<IllegalArgumentException> { PromptTemplate.format(template, values) }.message!!

    @Test
    fun `should substitute fields and collapse escapes when given the interfaces example`() {
        val out = PromptTemplate.format(
            "Translate this {source_lang}: {{x}} {text}",
            mapOf("source_lang" to "Spanish", "text" to "{hola}"),
        )
        assertEquals("Translate this Spanish: {x} {hola}", out)
    }

    @Test
    fun `should return distinct fields in first-seen order when names repeat or are escaped`() {
        assertEquals(listOf("a", "c"), PromptTemplate.fields("{a}{{b}}{a}{c}"))
    }

    @Test
    fun `should insert a value literally when it contains braces`() {
        assertEquals("{b}", PromptTemplate.format("{a}", mapOf("a" to "{b}", "b" to "y")))
        assertEquals("}}{{", PromptTemplate.format("{a}", mapOf("a" to "}}{{")))
    }

    @Test
    fun `should use the untrimmed name as key when the field has spaces`() {
        assertEquals("v", PromptTemplate.format("{ a }", mapOf(" a " to "v")))
        assertEquals(listOf(" a "), PromptTemplate.fields("{ a }"))
    }

    @Test
    fun `should return empty when the template is empty`() {
        assertEquals("", PromptTemplate.format("", emptyMap()))
        assertEquals(emptyList(), PromptTemplate.fields(""))
    }

    @Test
    fun `should ignore extra keys when values is a superset`() {
        assertEquals("x", PromptTemplate.format("{a}", mapOf("a" to "x", "zzz" to "unused")))
    }

    @Test
    fun `should report the unknown placeholder first when a stray brace follows it`() {
        assertEquals("Unknown placeholder '{missing}' at index 0; supplied keys: []", error("{missing} }"))
    }

    @Test
    fun `should report each syntax error with its exact message and index`() {
        assertEquals("Single '{' encountered in format string at index 2", error("ab{"))
        assertEquals("Single '{' encountered in format string at index 0", error("{a"))
        assertEquals("Single '}' encountered in format string at index 1", error("a}"))
        assertEquals("Unexpected '{' in field name at index 2", error("{a{b}}"))
        assertEquals("Positional field '{}' is not supported at index 1", error("x{}"))
        assertEquals("Positional field '{0}' is not supported at index 0", error("{0}"))
        val unsupported = "(conversion, format spec, attribute and index fields are not supported)"
        assertEquals("Unsupported field syntax '{a!r}' at index 0 $unsupported", error("{a!r}"))
        assertEquals("Unsupported field syntax '{a:>5}' at index 0 $unsupported", error("{a:>5}"))
        assertEquals("Unsupported field syntax '{a.b}' at index 0 $unsupported", error("{a.b}"))
        assertEquals("Unsupported field syntax '{a[0]}' at index 0 $unsupported", error("{a[0]}"))
    }

    @Test
    fun `should list sorted keys and never values when a placeholder is unknown`() {
        val message = error("hi {topic_block}", mapOf("transcript" to "SECRET SPEECH", "context_block" to "PRIVATE"))
        assertEquals(
            "Unknown placeholder '{topic_block}' at index 3; supplied keys: [context_block, transcript]",
            message,
        )
        assertFalse("SECRET" in message || "PRIVATE" in message)
    }

    @Test
    fun `should raise syntax errors but not unknown placeholders when only listing fields`() {
        assertEquals(listOf("anything"), PromptTemplate.fields("{anything}"))
        assertFailsWith<IllegalArgumentException> { PromptTemplate.fields("}") }
    }

    @Test
    fun `should keep surrogate pairs intact when text contains emoji`() {
        assertEquals("😀 x 😀", PromptTemplate.format("😀 {a} 😀", mapOf("a" to "x")))
    }
}
