package com.xrmech.habla.app.state

import com.xrmech.habla.app.fakes.FakeApiKeyStore
import com.xrmech.habla.app.fakes.FakeSettingsRepository
import com.xrmech.habla.app.fakes.TestAssets
import com.xrmech.habla.translate.LlmException.Kind
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class AppSettingsStateTest {
    @Test
    fun `should be null until the first emission and then await returns it`() = runTest {
        val repo = FakeSettingsRepository(loaded = false)
        val state = AppSettingsState(repo, backgroundScope)
        runCurrent()
        assertNull(state.settings.value)
        val waiting = async { state.awaitLoaded() }
        runCurrent()
        assertFalse(waiting.isCompleted)
        repo.load()
        runCurrent()
        assertEquals(repo.current, waiting.await())
    }

    @Test
    fun `should use models json defaults before load and the user's models after`() = runTest {
        val models = TestAssets.shared.models
        val repo = FakeSettingsRepository(loaded = false)
        val state = AppSettingsState(repo, backgroundScope)
        assertEquals(models.finalDefault, state.translateSettings(models).finalModel)
        repo.update { it.copy(finalModel = "custom-final", partialTranslation = false) }
        runCurrent()
        val s = state.translateSettings(models)
        assertEquals("custom-final", s.finalModel)
        assertFalse(s.partialTranslation)
    }
}

class KeyHealthTrackerTest {
    private fun TestScope.tracker(store: FakeApiKeyStore) =
        KeyHealthTracker(store.hasKey, backgroundScope).also { runCurrent() }

    @Test
    fun `should be absent without a key and present with one`() = runTest(UnconfinedTestDispatcher()) {
        val store = FakeApiKeyStore()
        val t = tracker(store)
        assertEquals(KeyState.ABSENT, t.state.value)
        store.set("sk-proj-abcdefghijkl")
        assertEquals(KeyState.PRESENT, t.state.value)
    }

    @Test
    fun `should go invalid and no quota after such outcomes and back on success`() = runTest(UnconfinedTestDispatcher()) {
        val t = tracker(FakeApiKeyStore("sk-proj-abcdefghijkl"))
        t.report(Kind.INVALID_KEY)
        assertEquals(KeyState.INVALID, t.state.value)
        t.report(Kind.NETWORK)
        assertEquals(KeyState.INVALID, t.state.value)
        t.report(Kind.QUOTA)
        assertEquals(KeyState.NO_QUOTA, t.state.value)
        t.report(null)
        assertEquals(KeyState.PRESENT, t.state.value)
    }

    @Test
    fun `should reset to present on save and absent on clear`() = runTest(UnconfinedTestDispatcher()) {
        val store = FakeApiKeyStore("sk-proj-abcdefghijkl")
        val t = tracker(store)
        t.report(Kind.INVALID_KEY)
        t.reset()
        assertEquals(KeyState.PRESENT, t.state.value)
        t.report(Kind.QUOTA)
        store.clear()
        t.reset()
        assertEquals(KeyState.ABSENT, t.state.value)
    }

    @Test
    fun `should map hasKey unknown to unknown`() {
        assertEquals(KeyState.UNKNOWN, KeyHealth.state(null, KeyVerdict.INVALID))
        assertEquals(KeyState.ABSENT, KeyHealth.state(false, KeyVerdict.INVALID))
    }
}

class ThermalPolicyTest {
    @Test
    fun `should pause at severe and above`() {
        assertTrue(ThermalPolicy.paused(false, ThermalPolicy.SEVERE))
        assertTrue(ThermalPolicy.paused(false, 6))
    }

    @Test
    fun `should resume at moderate and below`() {
        assertFalse(ThermalPolicy.paused(true, ThermalPolicy.MODERATE))
        assertFalse(ThermalPolicy.paused(true, 0))
    }

    @Test
    fun `should follow status changes through the monitor`() = runTest(UnconfinedTestDispatcher()) {
        val statuses = MutableSharedFlow<Int>()
        val monitor = ThermalMonitor(statuses, backgroundScope)
        runCurrent()
        assertFalse(monitor.partialsPaused.value)
        statuses.emit(4)
        assertTrue(monitor.partialsPaused.value)
        statuses.emit(2)
        assertFalse(monitor.partialsPaused.value)
        assertEquals(2, monitor.status.value)
    }
}
