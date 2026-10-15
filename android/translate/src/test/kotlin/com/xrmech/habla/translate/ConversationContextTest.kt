package com.xrmech.habla.translate

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class ConversationContextTest {

    private val profile = TestAssets.assets.asrProfile

    @Test
    fun profileSizesAreTenAndFive() {
        assertEquals(10, profile.context.recentExchanges)
        assertEquals(5, profile.context.maxContextExchanges)
    }

    @Test
    fun twelveAddsKeepTenAndRecentReturnsEightToTwelve() {
        val ctx = ConversationContext(profile)
        (1..12).forEach { ctx.add(ContextExchange("s$it", "t$it")) }
        assertEquals(10, ctx.size())
        assertEquals((8..12).map { ContextExchange("s$it", "t$it") }, ctx.recent())
    }

    @Test
    fun recentIsACopy() {
        val ctx = ConversationContext(profile)
        ctx.add(ContextExchange("a", "b"))
        val snapshot = ctx.recent()
        ctx.add(ContextExchange("c", "d"))
        assertEquals(1, snapshot.size)
    }

    @Test
    fun resetClearsExchangesAndSummary() {
        val ctx = ConversationContext(profile)
        ctx.add(ContextExchange("a", "b"))
        ctx.topicSummary = "topic"
        ctx.reset()
        assertEquals(emptyList(), ctx.recent())
        assertEquals("", ctx.topicSummary)
    }

    @Test
    fun concurrentAddsAreSafe() = runBlocking {
        val ctx = ConversationContext(profile)
        (1..1000).map { i -> async(Dispatchers.Default) { ctx.add(ContextExchange("s$i", "t$i")); ctx.recent() } }.awaitAll()
        assertEquals(10, ctx.size())
        assertEquals(5, ctx.recent().size)
    }
}
