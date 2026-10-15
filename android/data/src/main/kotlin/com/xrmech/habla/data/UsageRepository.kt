package com.xrmech.habla.data

import com.xrmech.habla.translate.model.TokenUsage
import com.xrmech.habla.translate.model.UsageKind
import kotlinx.coroutines.flow.Flow

/** Per-request OpenAI usage and the lifetime spend. */
interface UsageRepository {
    /**
     * One `usage_log` row. [sessionId] may be null (the "Test key" call).
     * @throws IllegalArgumentException for a NaN, infinite or negative cost or a negative token count.
     */
    suspend fun record(sessionId: Long?, usage: TokenUsage, kind: UsageKind, costUsd: Double)

    /** `TOTAL(cost_usd)` over every row, 0.0 when empty; survives session deletion (DM5). */
    fun totalCostUsd(): Flow<Double>
}
