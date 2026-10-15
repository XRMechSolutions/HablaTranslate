package com.xrmech.habla.data

import android.util.Log

/**
 * The module's only log sink (tag `HablaData`). Callers pass ids, counts and exception class
 * names only: never a key, a key prefix, transcripts, translations or vocab text.
 */
internal object DataLog {
    const val TAG = "HablaData"

    fun w(message: String) {
        Log.w(TAG, message)
    }

    fun i(message: String) {
        Log.i(TAG, message)
    }
}
