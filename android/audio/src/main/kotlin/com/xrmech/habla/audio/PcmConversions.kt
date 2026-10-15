package com.xrmech.habla.audio

/** PCM16 to float samples in [-1, 1) as `s / 32768f` (D025; also how `:asr` feeds whisper.cpp). */
fun ShortArray.toFloatSamples(): FloatArray = FloatArray(size) { this[it] / 32768f }
