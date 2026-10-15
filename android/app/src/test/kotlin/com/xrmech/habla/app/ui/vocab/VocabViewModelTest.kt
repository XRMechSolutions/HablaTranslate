package com.xrmech.habla.app.ui.vocab

import app.cash.turbine.test
import com.xrmech.habla.app.fakes.FakeVocabRepository
import com.xrmech.habla.app.fakes.MainDispatcherExtension
import java.time.Instant
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.extension.RegisterExtension

class VocabViewModelTest {
    @JvmField
    @RegisterExtension
    val main = MainDispatcherExtension()

    private val repo = FakeVocabRepository().apply {
        // Repository order is created_at DESC: newest first.
        rows.value = listOf(
            FakeVocabRepository.row(3, "echar una mano", "to help out", Instant.parse("2026-10-15T12:00:00Z")),
            FakeVocabRepository.row(2, "Tomar el pelo", "to pull someone's leg", Instant.parse("2026-10-14T12:00:00Z")),
            FakeVocabRepository.row(1, "estar en las nubes", "to daydream", Instant.parse("2026-10-13T12:00:00Z")),
        )
    }

    private suspend fun VocabViewModel.settled(): VocabUiState {
        var last: VocabUiState? = null
        ui.test {
            last = awaitItem()
            while (last!!.loading) last = awaitItem()
            cancelAndIgnoreRemainingEvents()
        }
        return last!!
    }

    @Test
    fun `should show every row newest first when the filter is empty`() = runTest {
        val vm = VocabViewModel(repo)
        val s = vm.settled()
        assertEquals(listOf(3L, 2L, 1L), s.rows.map { it.id })
        assertEquals(3, s.totalCount)
    }

    @Test
    fun `should filter on the term ignoring case`() = runTest {
        val vm = VocabViewModel(repo)
        vm.setFilter("TOMAR")
        assertEquals(listOf(2L), vm.settled().rows.map { it.id })
    }

    @Test
    fun `should filter on the meaning`() = runTest {
        val vm = VocabViewModel(repo)
        vm.setFilter("daydream")
        val s = vm.settled()
        assertEquals(listOf(1L), s.rows.map { it.id })
        assertEquals(3, s.totalCount)
    }

    @Test
    fun `should show all rows again for a blank filter`() = runTest {
        val vm = VocabViewModel(repo)
        vm.setFilter("help")
        assertEquals(1, vm.settled().rows.size)
        vm.setFilter("   ")
        assertEquals(3, vm.settled().rows.size)
    }

    @Test
    fun `should return nothing when no term or meaning matches`() {
        assertEquals(emptyList<Any>(), VocabViewModel.filterRows(repo.rows.value, "zzz"))
    }
}
