package com.xrmech.habla.app.testing

import com.xrmech.habla.app.pipeline.LiveController
import com.xrmech.habla.app.pipeline.LiveState
import com.xrmech.habla.app.service.SessionControl
import com.xrmech.habla.app.ui.settings.KeyTester
import com.xrmech.habla.data.ApiKeyStore
import com.xrmech.habla.data.AppSettings
import com.xrmech.habla.data.ExchangeRow
import com.xrmech.habla.data.HistoryRepository
import com.xrmech.habla.data.NewExchange
import com.xrmech.habla.data.SessionRow
import com.xrmech.habla.data.SettingsRepository
import com.xrmech.habla.data.UsageRepository
import com.xrmech.habla.data.VocabRepository
import com.xrmech.habla.data.VocabRow
import com.xrmech.habla.translate.KeyTestResult
import com.xrmech.habla.translate.LlmException
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.Mode
import com.xrmech.habla.translate.model.TokenUsage
import com.xrmech.habla.translate.model.UsageKind
import java.time.Instant
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map

class FakeLiveController(initial: LiveState = LiveState()) : LiveController {
    override val state = MutableStateFlow(initial)
    override val lastMetricsLine = MutableStateFlow<String?>(null)
}

class FakeSessionControl : SessionControl {
    @Volatile var starts = 0
    @Volatile var stops = 0

    override fun start() {
        starts++
    }

    override fun stop() {
        stops++
    }
}

class InMemorySettings(initial: AppSettings) : SettingsRepository {
    val state = MutableStateFlow(initial)
    override val settings: Flow<AppSettings> = state

    override suspend fun update(transform: (AppSettings) -> AppSettings) {
        state.value = transform(state.value)
    }
}

class InMemoryKeyStore(initial: String? = null) : ApiKeyStore {
    val key = MutableStateFlow(initial)
    override val hasKey: Flow<Boolean> = key.map { it != null }
    override suspend fun get(): String? = key.value

    override suspend fun set(key: String) {
        val t = key.trim()
        require(t.isNotEmpty()) { "API key is blank" }
        this.key.value = t
    }

    override suspend fun clear() {
        key.value = null
    }
}

class InMemoryHistory : HistoryRepository {
    private val ids = AtomicLong(100)
    val sessionRows = MutableStateFlow<List<SessionRow>>(emptyList())
    val exchangeRows = MutableStateFlow<Map<Long, List<ExchangeRow>>>(emptyMap())
    val corrected = MutableStateFlow<Map<Long, String>>(emptyMap())
    val added = CopyOnWriteArrayList<NewExchange>()

    override suspend fun startSession(mode: Mode, direction: Direction, asrModel: String): Long {
        val id = ids.incrementAndGet()
        sessionRows.value = listOf(SessionRow(id, Instant.now(), null, mode, direction, null, null, asrModel, null, 0, 0.0)) + sessionRows.value
        return id
    }

    override suspend fun endSession(id: Long, topicSummary: String?) {
        sessionRows.value = sessionRows.value.map { if (it.id == id) it.copy(endedAt = Instant.now(), topicSummary = topicSummary) else it }
    }

    override suspend fun addExchange(e: NewExchange): Long {
        added += e
        return ids.incrementAndGet()
    }

    override suspend fun setCorrectedTranscript(exchangeId: Long, text: String) {
        val t = text.trim()
        corrected.value = corrected.value + (exchangeId to t)
        exchangeRows.value = exchangeRows.value.mapValues { (_, rows) ->
            rows.map { if (it.id == exchangeId) it.copy(correctedTranscript = t.ifEmpty { null }) else it }
        }
    }

    override fun sessions(): Flow<List<SessionRow>> = sessionRows
    override fun exchanges(sessionId: Long): Flow<List<ExchangeRow>> = exchangeRows.map { it[sessionId].orEmpty() }

    override suspend fun deleteSession(id: Long) {
        sessionRows.value = sessionRows.value.filterNot { it.id == id }
    }

    override suspend fun session(id: Long): SessionRow? = sessionRows.value.firstOrNull { it.id == id }
}

class InMemoryVocab : VocabRepository {
    val saved = CopyOnWriteArrayList<FlaggedPhrase>()
    val rows = MutableStateFlow<List<VocabRow>>(emptyList())

    override suspend fun save(phrase: FlaggedPhrase, exchangeId: Long?, sourceSentence: String?) {
        saved += phrase
    }

    override fun all(): Flow<List<VocabRow>> = rows
}

class InMemoryUsage : UsageRepository {
    val total = MutableStateFlow(0.0)
    override suspend fun record(sessionId: Long?, usage: TokenUsage, kind: UsageKind, costUsd: Double) {
        total.value += costUsd
    }

    override fun totalCostUsd(): Flow<Double> = total
}

/** Answers Test key with [result]; never touches the network. */
class FakeKeyTester(var result: KeyTestResult = KeyTestResult.Failed(LlmException("No OpenAI API key stored", LlmException.Kind.NO_KEY))) : KeyTester {
    override suspend fun test(candidate: String?): KeyTestResult = result
}
