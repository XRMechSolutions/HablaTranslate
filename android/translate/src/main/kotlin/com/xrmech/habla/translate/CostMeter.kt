package com.xrmech.habla.translate

import com.xrmech.habla.shared.ModelTable
import com.xrmech.habla.translate.model.CostTotals
import com.xrmech.habla.translate.model.TokenUsage
import com.xrmech.habla.translate.model.UsageKind
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.util.concurrent.ConcurrentHashMap

/** One recorded API call, emitted on [CostMeter.records] for `:app` to persist. */
data class CostRecord(val usage: TokenUsage, val kind: UsageKind, val costUsd: Double)

/**
 * Session and all-time cost totals (port of the server's `_track_cloud_cost` / `_costs`).
 * Safe to call from concurrent finals: totals are updated with an atomic `StateFlow.update`.
 */
class CostMeter(private val models: ModelTable) {
    private val log = Log()
    private val warnedModels = ConcurrentHashMap.newKeySet<String>()

    private val _totals = MutableStateFlow(CostTotals())
    val totals: StateFlow<CostTotals> = _totals.asStateFlow()

    // Capacity 256: `:app` drains each record into a DB insert; even a burst of concurrent finals,
    // fallbacks and partials stays far below this. Overflow drops the newest record (tryEmit
    // fails, logged WARN); totals are unaffected.
    private val _records = MutableSharedFlow<CostRecord>(replay = 0, extraBufferCapacity = 256)
    val records: SharedFlow<CostRecord> = _records.asSharedFlow()

    /**
     * USD cost of [usage]. Cached input is billed at the cached rate; reasoning tokens are already
     * inside `outputTokens`. An unpriced model costs 0.0 (one WARN per model id).
     */
    fun costOf(usage: TokenUsage): Double {
        val price = models.models[usage.model]
        if (price == null) {
            if (warnedModels.add(usage.model)) log.warn("No price for OpenAI model ${usage.model}; cost not counted")
            return 0.0
        }
        val cached = minOf(usage.cachedInputTokens, usage.inputTokens)
        return (usage.inputTokens - cached) / 1_000_000.0 * price.input +
            cached / 1_000_000.0 * price.cachedInput +
            usage.outputTokens / 1_000_000.0 * price.output
    }

    fun record(usage: TokenUsage, kind: UsageKind) {
        val cost = costOf(usage)
        _totals.update {
            it.copy(
                sessionInputTokens = it.sessionInputTokens + usage.inputTokens,
                sessionOutputTokens = it.sessionOutputTokens + usage.outputTokens,
                sessionCostUsd = it.sessionCostUsd + cost,
                totalCostUsd = it.totalCostUsd + cost,
            )
        }
        if (!_records.tryEmit(CostRecord(usage, kind, cost))) {
            log.warn("Cost record buffer full; usage for ${usage.model} not persisted")
        }
    }

    /** Zeroes the session fields; the all-time total is kept. */
    fun resetSession() {
        _totals.update { it.copy(sessionInputTokens = 0, sessionOutputTokens = 0, sessionCostUsd = 0.0) }
    }

    /** Sets the all-time total (sum of `usage_log.cost_usd`, loaded by `:app` at start). */
    fun seedTotal(totalCostUsd: Double) {
        _totals.update { it.copy(totalCostUsd = totalCostUsd) }
    }
}
