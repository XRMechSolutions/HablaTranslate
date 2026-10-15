package com.xrmech.habla.shared

import java.io.File
import java.io.FileNotFoundException

/**
 * Reads the full UTF-8 text of `shared/<path>`, unmodified (no BOM stripping, no newline
 * conversion). `path` is relative and `/`-separated, e.g. `"idioms/spain.json"`.
 * Throws [FileNotFoundException] when absent, another `IOException` on read failure.
 */
fun interface AssetSource {
    fun readText(path: String): String
}

/** JVM [AssetSource] over a directory; used by every module's JVM tests to read the real `shared/`. */
class FileAssetSource(private val root: File) : AssetSource {

    override fun readText(path: String): String {
        requireValidPath(path)
        val file = File(root, path)
        if (!file.isFile) throw FileNotFoundException(file.absolutePath)
        return file.readText(Charsets.UTF_8)
    }

    private fun requireValidPath(path: String) {
        val invalid = path.isEmpty() ||
            path.startsWith('/') ||
            '\\' in path ||
            path.split('/').any { it.isEmpty() || it == "." || it == ".." }
        require(!invalid) { "invalid shared path '$path'" }
    }

    companion object {
        /** Root taken from the system property the Gradle test tasks set to `<repo>/shared`. */
        fun fromSystemProperty(name: String = "habla.sharedDir"): FileAssetSource {
            val value = System.getProperty(name)
                ?: throw IllegalStateException("system property $name is not set; run tests through Gradle")
            val dir = File(value)
            check(dir.isDirectory) { "$name is not a directory: $value" }
            return FileAssetSource(dir)
        }
    }
}

/** A `shared/` file is missing, unreadable, malformed or semantically invalid. Message names file and JSON path. */
class SharedAssetException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
