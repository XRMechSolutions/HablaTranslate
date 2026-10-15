package com.xrmech.habla.audio

/**
 * Test VadModel that replays a probability vector (indexed by VAD call, which starts after
 * calibration) and records what it received. Past the end of the vector it returns [after].
 */
internal class ScriptedVad(
    private val probs: FloatArray,
    private val after: Float = 0f,
    private val throwAtCall: Int = -1,
) : VadModel {
    var calls = 0
        private set
    var resets = 0
        private set
    var closed = false
        private set
    val frames = ArrayList<FloatArray>()
    var recordFrames = false

    override fun speechProbability(frame: FloatArray): Float {
        check(!closed) { "ScriptedVad is closed" }
        require(frame.size == 512) { "frame must have 512 samples, had ${frame.size}" }
        val i = calls++
        if (recordFrames) frames += frame.copyOf()
        if (i == throwAtCall) throw VadException("scripted failure at call $i")
        return if (i < probs.size) probs[i] else after
    }

    override fun reset() {
        check(!closed) { "ScriptedVad is closed" }
        resets++
    }

    override fun close() {
        closed = true
    }
}
