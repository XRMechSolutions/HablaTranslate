package com.xrmech.habla.data

import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.w3c.dom.Element

/**
 * The backup XML repeats `DataFiles` names as literals; this keeps them in step and pins what may
 * leave the device (DM12): never secrets or the DB journal, no training audio in cloud backup.
 */
class BackupRulesTest {
    private val secrets = "datastore/${DataFiles.SECRETS_STORE}.preferences_pb"
    private val secretsTmp = "$secrets.tmp"
    private val sessions = "${DataFiles.SESSIONS_DIR}/"
    private val journal = "${DataFiles.DATABASE_NAME}-journal"

    // Unit tests run with the module directory as the working directory.
    private fun parse(name: String): Element {
        val file = File("src/main/res/xml/$name")
        assertTrue(file.isFile, "missing ${file.absolutePath}")
        return DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(file).documentElement
    }

    private fun Element.child(tag: String): Element {
        val nodes = getElementsByTagName(tag)
        assertEquals(1, nodes.length, "expected one <$tag>")
        return nodes.item(0) as Element
    }

    /** (domain, path) of every `<exclude>` directly in this section. */
    private fun Element.excludes(): Set<Pair<String, String>> {
        val nodes = getElementsByTagName("exclude")
        return (0 until nodes.length).map {
            val e = nodes.item(it) as Element
            e.getAttribute("domain") to e.getAttribute("path")
        }.toSet()
    }

    private fun Element.includes(): Int = getElementsByTagName("include").length

    @Test
    fun `cloud backup excludes secrets, audio and journal and needs encryption`() {
        val cloud = parse("habla_data_extraction_rules.xml").child("cloud-backup")
        assertEquals("true", cloud.getAttribute("disableIfNoEncryptionCapabilities"))
        assertEquals(
            setOf("file" to secrets, "file" to secretsTmp, "file" to sessions, "database" to journal),
            cloud.excludes(),
        )
        assertEquals(0, cloud.includes())
    }

    @Test
    fun `device transfer excludes secrets and journal but keeps training audio`() {
        val transfer = parse("habla_data_extraction_rules.xml").child("device-transfer")
        val excludes = transfer.excludes()
        assertEquals(setOf("file" to secrets, "file" to secretsTmp, "database" to journal), excludes)
        assertFalse(excludes.any { it.second == sessions })
        assertEquals(0, transfer.includes())
    }

    @Test
    fun `full backup content for API 29-30 excludes all four`() {
        val root = parse("habla_backup_rules.xml")
        assertEquals("full-backup-content", root.tagName)
        assertEquals(
            setOf("file" to secrets, "file" to secretsTmp, "file" to sessions, "database" to journal),
            root.excludes(),
        )
        assertEquals(0, root.includes())
    }

    @Test
    fun `the database itself stays backed up`() {
        val all = parse("habla_data_extraction_rules.xml").excludes() + parse("habla_backup_rules.xml").excludes()
        assertFalse(all.any { it.second == DataFiles.DATABASE_NAME })
    }

    @Test
    fun `paths match the DataFiles names`() {
        assertEquals("datastore/secrets.preferences_pb", secrets)
        assertEquals("habla.db-journal", journal)
        assertEquals("sessions/", sessions)
    }
}
