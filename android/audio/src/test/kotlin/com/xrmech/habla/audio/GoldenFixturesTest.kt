package com.xrmech.habla.audio

import java.io.File
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test

class GoldenFixturesTest {

    @Test
    fun `manifest parses and names every fixture`() {
        val m = GoldenFixtures.manifest
        assertTrue(m.containsKey("tools") && m.containsKey("sessions") && m.containsKey("files"))
        assertEquals(listOf("session_es", "session_en_quiet", "session_noise"), GoldenFixtures.sessionNames)
        assertTrue(GoldenFixtures.finalPrepClips.isNotEmpty())
        assertTrue(GoldenFixtures.partialPrepClips.isNotEmpty())
    }

    @Test
    fun `every golden file matches its manifest hash`() {
        for ((name, expected) in GoldenFixtures.files) {
            assertEquals(expected, GoldenFixtures.sha256(GoldenFixtures.bytes(name)), "goldens/$name changed; ${GoldenFixtures.STALE}")
        }
    }

    @Test
    fun `every source clip matches its manifest hash`() {
        val clips = GoldenFixtures.manifest.getValue("clips") as JsonObject
        for ((name, expected) in clips) {
            val actual = GoldenFixtures.sha256(File(GoldenFixtures.clipsDir, name).readBytes())
            assertEquals(
                (expected as JsonPrimitive).content, actual,
                "shared/testdata/clips/$name changed; ${GoldenFixtures.STALE}",
            )
        }
    }

    @Test
    fun `asr profile audio sections match the ones the goldens were generated from`() {
        val expected = GoldenFixtures.manifest.getValue("profileSections") as JsonObject
        val profile = Json.parseToJsonElement(File(GoldenFixtures.sharedDir, "asr_profile.json").readText()) as JsonObject
        for ((section, value) in expected) {
            assertEquals(value, profile[section], "shared/asr_profile.json [$section] changed; ${GoldenFixtures.STALE}")
        }
    }

    @Test
    fun `composed sessions equal the server's input`() {
        for (name in GoldenFixtures.sessionNames) {
            val s = GoldenFixtures.session(name)
            assertEquals(s.length, s.pcm.size, "$name length")
            assertEquals(s.pcmSha256, GoldenFixtures.sha256(s.pcm), "$name content; ${GoldenFixtures.STALE}")
        }
    }

    @Test
    fun `composed prep inputs equal the generator's input`() {
        for (c in GoldenFixtures.finalPrepClips) {
            assertEquals(c.length, c.input.size, c.key)
            assertEquals(c.file("inputSha256"), GoldenFixtures.sha256(c.input), "${c.key}; ${GoldenFixtures.STALE}")
        }
    }

    @Test
    fun `local goldens directory is gitignored`() {
        val repo = GoldenFixtures.sharedDir.parentFile
        val target = "android/audio/src/test/resources/goldens/local/x"
        val proc = try {
            ProcessBuilder("git", "check-ignore", "-q", target).directory(repo).redirectErrorStream(true).start()
        } catch (e: java.io.IOException) {
            null
        }
        assumeTrue(proc != null, "git not on PATH")
        assertTrue(proc!!.waitFor(30, TimeUnit.SECONDS))
        assumeTrue(proc.exitValue() != 128, "not a git checkout")
        assertEquals(0, proc.exitValue(), "$target must be ignored by goldens/.gitignore")
    }
}
