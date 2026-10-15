package com.xrmech.habla.data

/** A storage failure (SQLite or file I/O) surfaced by a `:data` repository. */
class DataException(message: String, cause: Throwable? = null) : Exception(message, cause)
