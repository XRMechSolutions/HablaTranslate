package com.xrmech.habla.shared

import java.io.File
import java.io.FileNotFoundException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FileAssetSourceTest {

    private val source = FileAssetSource.fromSystemProperty()

    @Test
    fun `should read a nested file when the path is relative and slash-separated`() {
        assertTrue(source.readText("idioms/index.json").contains("\"files\""))
    }

    @Test
    fun `should reject traversal and absolute paths when reading`() {
        for (path in listOf("../x", "idioms/../x", "./x", "\\x", "/x", "idioms\\spain.json", "", "idioms//spain.json")) {
            val e = assertFailsWith<IllegalArgumentException>(path) { source.readText(path) }
            assertEquals("invalid shared path '$path'", e.message)
        }
    }

    @Test
    fun `should throw FileNotFoundException with the absolute path when the file is absent`() {
        val e = assertFailsWith<FileNotFoundException> { source.readText("no_such_file.json") }
        assertTrue(e.message!!.endsWith("no_such_file.json"))
        assertTrue(File(e.message!!).isAbsolute)
    }

    @Test
    fun `should explain how to run when the system property is unset`() {
        val e = assertFailsWith<IllegalStateException> {
            FileAssetSource.fromSystemProperty("habla.test.unsetProperty")
        }
        assertEquals("system property habla.test.unsetProperty is not set; run tests through Gradle", e.message)
    }

    @Test
    fun `should name the path when the system property is not a directory`() {
        val name = "habla.test.notADirectory"
        val bogus = File(System.getProperty("habla.sharedDir"), "no_such_dir").path
        System.setProperty(name, bogus)
        try {
            val e = assertFailsWith<IllegalStateException> { FileAssetSource.fromSystemProperty(name) }
            assertEquals("$name is not a directory: $bogus", e.message)
        } finally {
            System.clearProperty(name)
        }
    }
}
