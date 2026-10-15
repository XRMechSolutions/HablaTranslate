package com.xrmech.habla.data

import androidx.datastore.preferences.core.edit
import app.cash.turbine.test
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode
import java.io.File
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir

class SettingsRepositoryTest {

    @TempDir
    lateinit var tempDir: File

    private val defaults = AppSettings(
        Direction.AUTO, Mode.CONVERSATION, true, true, "gpt-5.4-mini", "gpt-6-luna", "ggml-small-q8_0", false, false,
    )

    private val file get() = tempDir.resolve("settings.preferences_pb")

    private fun TestScope.store() = testPreferencesStore(backgroundScope, file)

    @Test
    fun `empty file emits exactly the defaults`() = runTest {
        val repo = DataStoreSettingsRepository(store(), defaults)
        repo.settings.test {
            assertEquals(defaults, awaitItem())
            expectNoEvents()
        }
    }

    @Test
    fun `update round trips all nine keys`() = runTest {
        val store = store()
        val repo = DataStoreSettingsRepository(store, defaults)
        val changed = AppSettings(
            Direction.EN_TO_ES, Mode.CLASSROOM, false, false, "final-x", "quick-y", "asr-z", true, true,
        )
        repo.update { changed }
        assertEquals(changed, repo.settings.first())
        val p = store.data.first()
        assertEquals("en_to_es", p[SettingsCodec.DIRECTION])
        assertEquals("classroom", p[SettingsCodec.MODE])
        assertEquals(false, p[SettingsCodec.PARTIALS_ENABLED])
        assertEquals(false, p[SettingsCodec.PARTIAL_TRANSLATION_ENABLED])
        assertEquals("final-x", p[SettingsCodec.FINAL_MODEL])
        assertEquals("quick-y", p[SettingsCodec.QUICK_MODEL])
        assertEquals("asr-z", p[SettingsCodec.ASR_MODEL_ID])
        assertEquals(true, p[SettingsCodec.TRAINING_CAPTURE])
        assertEquals(true, p[SettingsCodec.FIRST_RUN_DONE])
        assertEquals(9, p.asMap().size)
    }

    @Test
    fun `update emits once more on settings`() = runTest {
        val repo = DataStoreSettingsRepository(store(), defaults)
        repo.settings.test {
            assertEquals(defaults, awaitItem())
            repo.update { it.copy(trainingCapture = true) }
            assertEquals(defaults.copy(trainingCapture = true), awaitItem())
            repo.update { it }
            expectNoEvents()
        }
    }

    @Test
    fun `unknown enum strings give that field's default`() = runTest {
        val store = store()
        store.edit {
            it[SettingsCodec.DIRECTION] = "classroom_v2"
            it[SettingsCodec.MODE] = "lecture"
            it[SettingsCodec.FINAL_MODEL] = "kept-model"
        }
        val s = DataStoreSettingsRepository(store, defaults).settings.first()
        assertEquals(Direction.AUTO, s.direction)
        assertEquals(Mode.CONVERSATION, s.mode)
        assertEquals("kept-model", s.finalModel)
    }

    @Test
    fun `two concurrent updates both apply`() = runTest {
        val repo = DataStoreSettingsRepository(store(), defaults)
        listOf(
            async { repo.update { it.copy(trainingCapture = true) } },
            async { repo.update { it.copy(firstRunDone = true) } },
        ).awaitAll()
        val s = repo.settings.first()
        assertEquals(true, s.trainingCapture)
        assertEquals(true, s.firstRunDone)
    }

    @Test
    fun `transform exception propagates and writes nothing`() = runTest {
        val store = store()
        val repo = DataStoreSettingsRepository(store, defaults)
        assertThrows<IllegalStateException> { repo.update { error("bad transform") } }
        assertNull(store.data.first()[SettingsCodec.DIRECTION])
    }

    @Test
    fun `random bytes in the file give defaults`() = runTest {
        file.writeBytes(ByteArray(64) { (it * 37 + 11).toByte() })
        val repo = DataStoreSettingsRepository(store(), defaults)
        assertEquals(defaults, repo.settings.first())
        repo.update { it.copy(firstRunDone = true) }
        assertEquals(true, repo.settings.first().firstRunDone)
    }

    @Test
    fun `EnumCodec round trips every constant`() {
        for (d in Direction.entries) assertEquals(d, EnumCodec.decodeDirection(EnumCodec.encode(d)))
        for (m in Mode.entries) assertEquals(m, EnumCodec.decodeMode(EnumCodec.encode(m)))
        assertEquals(listOf("es_to_en", "en_to_es", "auto"), Direction.entries.map { EnumCodec.encode(it) })
        assertEquals(listOf("conversation", "classroom"), Mode.entries.map { EnumCodec.encode(it) })
        assertNull(EnumCodec.decodeDirection("ES_TO_EN"))
        assertNull(EnumCodec.decodeMode("Conversation"))
    }
}
