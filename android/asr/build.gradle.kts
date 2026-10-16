import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.serialization)
}

// whisper.cpp v1.9.5 (D028). Upgrading is an explicit change of this constant plus a D5 re-run.
val WHISPER_CPP_COMMIT = "d1be6fde11ac6e0407606b4e42fe72d34add8037"

android {
    namespace = "com.xrmech.habla.asr"
    compileSdk = libs.versions.android.compileSdk.get().toInt()
    ndkVersion = libs.versions.ndk.get()

    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")

        ndk {
            // armeabi-v7a and x86 serve no supported device; x86_64 exists for the emulator only.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_static", "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON")
                // Experimental GPU variant habla_asr_vk (D043): -Phabla.vulkan=true, SDK from
                // -Phabla.vulkanSdk or VULKAN_SDK. See src/main/cpp/CMakeLists.txt for the host needs.
                if (providers.gradleProperty("habla.vulkan").orNull == "true") {
                    arguments += "-DHABLA_VULKAN=ON"
                    providers.gradleProperty("habla.vulkanSdk").orNull?.let { arguments += "-DHABLA_VULKAN_SDK=$it" }
                }
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        // BuildConfig.DEBUG gates the leak WARN in WhisperCppRecognizer.
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(project(":shared-assets"))
    implementation(libs.coroutines.core)
    implementation(libs.coroutines.android)
    implementation(libs.serialization.json)
    implementation(libs.ktor.client.core)
    implementation(libs.ktor.client.okhttp)
    implementation(libs.androidx.annotation)
    constraints {
        // Ktor 3.6.0 pulls OkHttp 5.5.0, whose AAR needs compileSdk 37; 5.4.0 is the newest that
        // accepts compileSdk 36 (D034).
        implementation(libs.okhttp) { version { strictly(libs.versions.okhttp.get()) } }
    }

    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
    testImplementation(libs.coroutines.test)
    testImplementation(libs.turbine)
    testImplementation(libs.ktor.client.mock)

    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.core)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.coroutines.test)
}

// Gradle root is android/, so this is <repo>/shared (shared-assets design section 9).
val repoSharedDir: File = rootProject.projectDir.resolveSibling("shared")
check(repoSharedDir.isDirectory) { "shared assets directory not found at $repoSharedDir" }

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    inputs.dir(repoSharedDir).withPathSensitivity(PathSensitivity.RELATIVE).withPropertyName("sharedDir")
    systemProperty("habla.sharedDir", repoSharedDir.absolutePath)
}

// Fails before CMake runs, with an actionable message, when the submodule is absent or not at the pin.
val checkWhisperSubmodule = tasks.register("checkWhisperSubmodule") {
    val dir = rootProject.projectDir.resolve("third_party/whisper.cpp")
    val pin = WHISPER_CPP_COMMIT
    doLast {
        if (!dir.resolve("include/whisper.h").isFile) {
            throw GradleException(
                "whisper.cpp submodule missing: run \"git submodule update --init android/third_party/whisper.cpp\"",
            )
        }
        val proc = ProcessBuilder("git", "-C", dir.absolutePath, "rev-parse", "HEAD")
            .redirectErrorStream(true)
            .start()
        val head = proc.inputStream.bufferedReader().readText().trim()
        if (proc.waitFor() != 0 || head != pin) {
            throw GradleException(
                "whisper.cpp submodule is at '$head' but :asr pins WHISPER_CPP_COMMIT=$pin; " +
                    "run \"git -C android/third_party/whisper.cpp checkout $pin\" or change the pin deliberately",
            )
        }
    }
}
tasks.named("preBuild") { dependsOn(checkWhisperSubmodule) }

// ---- Instrumented tests (emulator): model files and test assets -----------------------------------

// Where :asr:prepareTestModels caches downloaded test models. Never inside a module; override with
// -Phabla.modelCacheDir=<dir> or HABLA_MODEL_CACHE (e.g. a shared cache outside the repo).
val testModelCacheDir: File = (findProperty("habla.modelCacheDir") as String?)?.let(::File)
    ?: System.getenv("HABLA_MODEL_CACHE")?.let(::File)
    ?: rootProject.file(".cache/models")

// Downloads the androidTest model catalog (ggml-tiny-q8_0), checks size + sha256, and adb-pushes it to
// /data/local/tmp/habla-models/ where the tests copy it from with `cat` (apps cannot read that dir).
tasks.register("prepareTestModels") {
    group = "verification"
    description = "Download, verify and adb push the androidTest Whisper models"
    val catalogFile = file("src/androidTest/assets/test_models.json")
    val cacheDir = testModelCacheDir
    val adb = androidComponents.sdkComponents.adb
    inputs.file(catalogFile)
    outputs.upToDateWhen { false }
    doLast {
        fun sha256(f: File): String {
            val md = MessageDigest.getInstance("SHA-256")
            f.inputStream().use { input ->
                val buf = ByteArray(1 shl 20)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    md.update(buf, 0, n)
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }

        @Suppress("UNCHECKED_CAST")
        val models = (groovy.json.JsonSlurper().parse(catalogFile) as Map<String, Any>)["models"] as List<Map<String, Any>>
        cacheDir.mkdirs()
        for (m in models) {
            val name = m["file"] as String
            val url = m["url"] as String
            val sha = m["sha256"] as String
            val size = (m["size_bytes"] as Number).toLong()
            val target = File(cacheDir, name)
            if (!(target.isFile && target.length() == size && sha256(target) == sha)) {
                logger.lifecycle("downloading $url")
                val tmp = File(cacheDir, "$name.download")
                try {
                    val conn = URI(url).toURL().openConnection() as HttpURLConnection
                    conn.instanceFollowRedirects = true
                    conn.setRequestProperty("User-Agent", "HablaAndroid-ModelManager")
                    conn.inputStream.use { input -> tmp.outputStream().use { input.copyTo(it, 1 shl 20) } }
                } catch (e: IOException) {
                    throw GradleException("test model download failed for $url (sha256 $sha): ${e.message}", e)
                }
                val got = sha256(tmp)
                if (tmp.length() != size || got != sha) {
                    tmp.delete()
                    throw GradleException("test model $url: expected $size bytes sha256 $sha, got ${tmp.length()} bytes sha256 $got")
                }
                Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
            val adbExe = adb.get().asFile.absolutePath
            for (cmd in listOf(
                listOf(adbExe, "shell", "mkdir", "-p", "/data/local/tmp/habla-models"),
                listOf(adbExe, "push", target.absolutePath, "/data/local/tmp/habla-models/$name"),
            )) {
                val proc = ProcessBuilder(cmd).redirectErrorStream(true).start()
                val out = proc.inputStream.bufferedReader().readText()
                if (proc.waitFor() != 0) throw GradleException("${cmd.joinToString(" ")} failed: $out")
                logger.lifecycle(out.trim())
            }
        }
    }
}

// D5/D6 on the A56 (@DeviceOnly): verifies the app catalog's small models in the model cache and
// adb-pushes them to /data/local/tmp/habla-models/, plus the golden set (JSON + post-FinalPrep WAVs)
// from HABLA_GOLDEN_DIR (default shared/testdata/goldens) to /data/local/tmp/habla-golden/asr/.
// Models must already be in the cache (D:/.cache/whisper-cpp via HABLA_MODEL_CACHE; compare_engines.py
// downloads them). -Phabla.benchModels=<id,id> limits the models. Target one device with ANDROID_SERIAL.
tasks.register("prepareDeviceBench") {
    group = "verification"
    description = "Verify and adb push the small Whisper models and the D6 golden set for the A56 suites"
    val catalogFile = repoSharedDir.resolve("asr_models.json")
    val goldenDir = System.getenv("HABLA_GOLDEN_DIR")?.let(::File) ?: repoSharedDir.resolve("testdata/goldens")
    val cacheDir = testModelCacheDir
    val only = (findProperty("habla.benchModels") as String?)?.split(',')?.map { it.trim() }?.toSet()
    val adb = androidComponents.sdkComponents.adb
    outputs.upToDateWhen { false }
    doLast {
        fun sha256(f: File): String {
            val md = MessageDigest.getInstance("SHA-256")
            f.inputStream().use { input ->
                val buf = ByteArray(1 shl 20)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    md.update(buf, 0, n)
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }
        val adbExe = adb.get().asFile.absolutePath
        fun run(vararg cmd: String) {
            val proc = ProcessBuilder(adbExe, *cmd).redirectErrorStream(true).start()
            val out = proc.inputStream.bufferedReader().readText()
            if (proc.waitFor() != 0) throw GradleException("adb ${cmd.joinToString(" ")} failed: $out")
            logger.lifecycle(out.trim())
        }

        @Suppress("UNCHECKED_CAST")
        val models = (groovy.json.JsonSlurper().parse(catalogFile) as Map<String, Any>)["models"] as List<Map<String, Any>>
        run("shell", "mkdir", "-p", "/data/local/tmp/habla-models")
        // Default: the models that may transcribe finals; partial-only ones (base) only when named.
        fun finalRole(m: Map<String, Any>) = (m["roles"] as List<*>?)?.contains("final") ?: true
        for (m in models.filter { if (only == null) finalRole(it) else it["id"] in only }) {
            val f = File(cacheDir, m["file"] as String)
            val sha = m["sha256"] as String
            if (!f.isFile || f.length() != (m["size_bytes"] as Number).toLong() || sha256(f) != sha) {
                throw GradleException("${f.absolutePath} missing or not sha256 $sha; download ${m["url"]} there first")
            }
            run("push", f.absolutePath, "/data/local/tmp/habla-models/${f.name}")
        }

        val jsons = goldenDir.listFiles { f -> f.isFile && f.name.endsWith(".json") }?.sortedBy { it.name }.orEmpty()
        if (jsons.isEmpty() || !goldenDir.resolve("wav").isDirectory) {
            throw GradleException("golden set not found at $goldenDir (needs *.json and wav/; run habla/scripts/make_goldens.py)")
        }
        val index = temporaryDir.resolve("index.txt").apply { writeText(jsons.joinToString("\n") { it.name } + "\n") }
        run("shell", "rm", "-rf", "/data/local/tmp/habla-golden/asr")
        run("shell", "mkdir", "-p", "/data/local/tmp/habla-golden/asr")
        jsons.forEach { run("push", it.absolutePath, "/data/local/tmp/habla-golden/asr/${it.name}") }
        run("push", goldenDir.resolve("wav").absolutePath, "/data/local/tmp/habla-golden/asr/")
        run("push", index.absolutePath, "/data/local/tmp/habla-golden/asr/index.txt")
    }
}

// Pushes must land before instrumentation starts when both are on one command line.
tasks.matching { it.name.startsWith("connected") && it.name.endsWith("AndroidTest") }.configureEach {
    mustRunAfter("prepareTestModels", "prepareDeviceBench")
}

// Public-domain test clips, their goldens (D033) and the shared/ JSON files, packaged only into the
// androidTest APK. Copied at build time so no second copy lives in git.
val hablaAndroidTestAssets = layout.buildDirectory.dir("generated/habla/androidTestAssets")
val syncHablaAndroidTestAssets = tasks.register<Sync>("syncHablaAndroidTestAssets") {
    from(repoSharedDir) {
        include("*.json", "idioms/*.json")
        into("shared")
    }
    from(repoSharedDir.resolve("testdata/clips")) {
        include("es_marti_02.wav", "en_aesop_01.wav")
        into("clips")
    }
    from(repoSharedDir.resolve("testdata/goldens")) {
        include("es_marti_02.json", "en_aesop_01.json")
        into("goldens")
    }
    into(hablaAndroidTestAssets)
}
android {
    sourceSets {
        getByName("androidTest") { assets.srcDir(hablaAndroidTestAssets.get().asFile) }
    }
}
tasks.matching { it.name.startsWith("merge") && it.name.endsWith("AndroidTestAssets") }.configureEach {
    dependsOn(syncHablaAndroidTestAssets)
}
