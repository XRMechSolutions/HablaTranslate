package com.xrmech.habla.app.pipeline

import com.xrmech.habla.app.fakes.TestAssets
import com.xrmech.habla.asr.AsrException
import com.xrmech.habla.asr.DownloadResponse
import com.xrmech.habla.asr.ModelDownloader
import com.xrmech.habla.asr.ModelManager
import java.io.File
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir

class RecognizerHolderTest {
    @TempDir
    lateinit var dir: File

    private val offline = object : ModelDownloader {
        override suspend fun <T> open(url: String, rangeStart: Long, block: suspend (DownloadResponse) -> T): T =
            throw IOException("offline")
    }

    private fun holder(): RecognizerHolder {
        val assets = TestAssets.shared
        return RecognizerHolder(ModelManager(dir, assets.asrModels, offline), assets, Dispatchers.Unconfined, threads = 1)
    }

    @Test
    fun `should use the profile whisper_cpp threads by default`() {
        val assets = TestAssets.shared
        val h = RecognizerHolder(ModelManager(dir, assets.asrModels, offline), assets, Dispatchers.Unconfined)
        assertEquals(assets.asrProfile.whisperCpp.threads, h.threads)
    }

    @Test
    fun `should throw model missing when the verified file is absent`() = runTest {
        val e = assertThrows<AsrException> { holder().acquire(TestAssets.shared.asrModels.defaultId) }
        assertEquals(AsrException.Kind.MODEL_MISSING, e.kind)
        assertEquals("Speech model missing - download it in Settings", e.message)
    }

    @Test
    fun `should allow release twice with nothing loaded`() {
        val h = holder()
        h.release()
        h.release()
    }
}
