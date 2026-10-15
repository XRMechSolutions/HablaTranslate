package com.xrmech.habla.data

import app.cash.sqldelight.coroutines.asFlow
import app.cash.sqldelight.coroutines.mapToOne
import com.xrmech.habla.data.db.HablaDatabase
import com.xrmech.habla.translate.model.TokenUsage
import com.xrmech.habla.translate.model.UsageKind
import java.time.Clock
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow

/**
 * [UsageRepository] over `usage_log`. Reasoning tokens are not stored: OpenAI already counts them
 * inside `output_tokens`. Invalid values are rejected because they would corrupt the lifetime total.
 */
class SqlUsageRepository(
    db: HablaDatabase,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val clock: Clock = Clock.systemUTC(),
) : UsageRepository {

    private val queries = db.usageQueries

    override suspend fun record(sessionId: Long?, usage: TokenUsage, kind: UsageKind, costUsd: Double) {
        dbCall(dispatcher, "record usage") {
            require(costUsd.isFinite() && costUsd >= 0.0) { "cost must be finite and >= 0" }
            require(usage.inputTokens >= 0 && usage.cachedInputTokens >= 0 && usage.outputTokens >= 0) {
                "token counts must be >= 0"
            }
            queries.insertUsage(
                session_id = sessionId,
                at = Timestamps.now(clock),
                model = usage.model,
                kind = kind.name.lowercase(),
                input_tokens = usage.inputTokens.toLong(),
                cached_input_tokens = usage.cachedInputTokens.toLong(),
                output_tokens = usage.outputTokens.toLong(),
                cost_usd = costUsd,
            )
        }
    }

    override fun totalCostUsd(): Flow<Double> =
        queries.selectTotalCost().asFlow().mapToOne(dispatcher).asDataFlow("read total cost")
}
