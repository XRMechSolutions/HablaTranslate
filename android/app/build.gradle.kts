import java.net.HttpURLConnection
import java.net.URI
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

// --- personal flavor key seed (D022, R5 section 6) -------------------------------------------
// The gitignored android/.env is read lazily: configuration never fails and public variants never
// resolve this provider. Only personal variants get BuildConfig.SEED_OPENAI_KEY.
val envSeedKey: Provider<String> = providers
    .fileContents(rootProject.layout.projectDirectory.file(".env")).asText
    .map { txt ->
        txt.lineSequence().map(String::trim)
            .firstOrNull { it.startsWith("OPENAI_API_KEY=") }
            ?.substringAfter('=')?.trim()?.trim('"', '\'')?.trim() ?: ""
    }
    .orElse("")

abstract class CheckPersonalEnv : DefaultTask() {
    // @Internal, not @Input: the key value must not be stored in Gradle's task history.
    @get:Internal
    abstract val seedKey: Property<String>

    init {
        outputs.upToDateWhen { false }
    }

    @TaskAction
    fun check() {
        if (seedKey.get().isBlank()) {
            throw GradleException("android/.env missing or has no OPENAI_API_KEY (D022: the personal flavor needs it)")
        }
    }
}

val checkPersonalEnv = tasks.register<CheckPersonalEnv>("checkPersonalEnv") {
    description = "Fails personal builds when android/.env has no OPENAI_API_KEY (D022)."
    group = "verification"
    seedKey.set(envSeedKey)
}

// --- release signing: gitignored android/keystore.properties, else debug-signed with a warning ---
val keystorePropsFile: File = rootProject.file("keystore.properties")
val keystoreProps: Properties? = keystorePropsFile.takeIf { it.isFile }?.let { f ->
    Properties().apply { f.inputStream().use { load(it) } }
}

val emulatorOnly: Boolean = providers.gradleProperty("habla.emulatorOnly").map { it.toBoolean() }.getOrElse(false)

// llm-gateway base URL for Pro (D051): -Phabla.gatewayUrl or HABLA_GATEWAY_URL. Empty = no Pro engine
// in this build (the picker hides it). Not a secret: the gateway authenticates users, not builds.
val gatewayUrl: String = (providers.gradleProperty("habla.gatewayUrl").orNull ?: System.getenv("HABLA_GATEWAY_URL") ?: "").trim()
require(gatewayUrl.isEmpty() || Regex("^https?://[A-Za-z0-9.-]+(:[0-9]+)?/?$").matches(gatewayUrl)) {
    "habla.gatewayUrl must be a bare http(s) origin, got '$gatewayUrl'"
}

android {
    namespace = "com.xrmech.habla.app"
    compileSdk = libs.versions.android.compileSdk.get().toInt()
    ndkVersion = libs.versions.ndk.get()

    defaultConfig {
        applicationId = "com.xrmech.habla"
        minSdk = libs.versions.android.minSdk.get().toInt()
        targetSdk = libs.versions.android.targetSdk.get().toInt()
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // Release ships arm64 only; debug adds x86_64 for the Windows emulator (CCR-4).
        // -Phabla.emulatorOnly=true packs only x86_64 (a 125 MB two-ABI debug APK does not fit a
        // nearly full emulator /data); never set it for a phone or release build.
        ndk { abiFilters += if (emulatorOnly) "x86_64" else "arm64-v8a" }
        buildConfigField("String", "GATEWAY_URL", "\"$gatewayUrl\"")
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    flavorDimensions += "dist"
    productFlavors {
        create("public") {
            dimension = "dist"
        }
        create("personal") {
            dimension = "dist"
            applicationIdSuffix = ".personal"
            versionNameSuffix = "-personal"
        }
    }

    signingConfigs {
        if (keystoreProps != null) {
            create("release") {
                storeFile = rootProject.file(keystoreProps.getProperty("storeFile"))
                storePassword = keystoreProps.getProperty("storePassword")
                keyAlias = keystoreProps.getProperty("keyAlias")
                keyPassword = keystoreProps.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        getByName("debug") {
            applicationIdSuffix = ".debug"
            ndk { abiFilters += "x86_64" }
        }
        getByName("release") {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = if (keystoreProps != null) {
                signingConfigs.getByName("release")
            } else {
                logger.warn("android/keystore.properties not found: release builds are signed with the debug key (sideload only).")
                signingConfigs.getByName("debug")
            }
        }
    }

    packaging {
        // Uncompressed, page-aligned libs (mmap) for real builds; the emulator-only build compresses
        // them so the APK fits a nearly full emulator /data.
        jniLibs.useLegacyPackaging = emulatorOnly
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    testOptions {
        // The pipeline package touches no android.* API, so strict stubs catch any leak.
        unitTests.isReturnDefaultValues = false
        unitTests.all { it.useJUnitPlatform() }
        animationsDisabled = true
    }

    lint {
        abortOnError = true
        checkReleaseBuilds = true
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

// JVM unit tests read the real shared/ assets and test clips (SharedAssets(FileAssetSource.fromSystemProperty())).
val sharedDirForTests: File = rootProject.projectDir.resolveSibling("shared")
tasks.withType<Test>().configureEach {
    inputs.dir(sharedDirForTests).withPathSensitivity(PathSensitivity.RELATIVE).withPropertyName("sharedDir")
    systemProperty("habla.sharedDir", sharedDirForTests.absolutePath)
}

androidComponents {
    onVariants(selector().withFlavor("dist" to "personal")) { variant ->
        variant.buildConfigFields?.put(
            "SEED_OPENAI_KEY",
            envSeedKey.map {
                com.android.build.api.variant.BuildConfigField(
                    "String",
                    "\"" + it.replace("\\", "\\\\").replace("\"", "\\\"") + "\"",
                    null,
                )
            },
        )
        val preBuild = "pre${variant.name.replaceFirstChar { it.uppercase() }}Build"
        tasks.matching { it.name == preBuild }.configureEach { dependsOn(checkPersonalEnv) }
    }
}

// --- merged-manifest guard (design section 2) ------------------------------------------------
// Runs on the public release merged manifest: a library that adds a permission, or a lost
// tools:node="remove" for ORT telemetry, fails `check` with the offender's name.
abstract class VerifyMergedManifest : DefaultTask() {
    @get:InputFile
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val mergedManifest: RegularFileProperty

    @get:Input
    abstract val allowedPermissions: ListProperty<String>

    @get:Input
    abstract val forbiddenComponents: ListProperty<String>

    @get:OutputFile
    abstract val report: RegularFileProperty

    @TaskAction
    fun verify() {
        val androidNs = "http://schemas.android.com/apk/res/android"
        val factory = javax.xml.parsers.DocumentBuilderFactory.newInstance().apply { isNamespaceAware = true }
        val doc = factory.newDocumentBuilder().parse(mergedManifest.get().asFile)
        val problems = mutableListOf<String>()
        val allowed = allowedPermissions.get().toSet()
        for (tag in listOf("uses-permission", "uses-permission-sdk-23")) {
            val nodes = doc.getElementsByTagName(tag)
            for (i in 0 until nodes.length) {
                val name = (nodes.item(i) as org.w3c.dom.Element).getAttributeNS(androidNs, "name")
                if (name !in allowed) problems += "unexpected permission $name"
            }
        }
        val forbidden = forbiddenComponents.get().toSet()
        val all = doc.getElementsByTagName("*")
        for (i in 0 until all.length) {
            val name = (all.item(i) as org.w3c.dom.Element).getAttributeNS(androidNs, "name")
            if (name in forbidden) problems += "forbidden component $name"
        }
        if (problems.isNotEmpty()) {
            throw GradleException("verifyMergedManifest: " + problems.joinToString("; "))
        }
        report.get().asFile.writeText("OK\n")
    }
}

val verifyMergedManifest = tasks.register<VerifyMergedManifest>("verifyMergedManifest") {
    description = "Fails if the public release manifest has an unlisted permission or ORT telemetry."
    group = "verification"
    allowedPermissions.set(
        listOf(
            "android.permission.RECORD_AUDIO",
            "android.permission.FOREGROUND_SERVICE",
            "android.permission.FOREGROUND_SERVICE_MICROPHONE",
            "android.permission.POST_NOTIFICATIONS",
            "android.permission.WAKE_LOCK",
            "android.permission.INTERNET",
            "android.permission.ACCESS_NETWORK_STATE",
        ),
    )
    forbiddenComponents.set(listOf("ai.onnxruntime.TelemetryInitializer"))
    report.set(layout.buildDirectory.file("reports/verifyMergedManifest.txt"))
}
tasks.named("check") { dependsOn(verifyMergedManifest) }

androidComponents {
    onVariants(selector().withName("publicRelease")) { variant ->
        verifyMergedManifest.configure {
            mergedManifest.set(variant.artifacts.get(com.android.build.api.artifact.SingleArtifact.MERGED_MANIFEST))
            // androidx.core's app-private signature permission (ContextCompat.registerReceiver with
            // RECEIVER_NOT_EXPORTED below API 33); grants no capability outside this app.
            allowedPermissions.add(variant.applicationId.map { "$it.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION" })
        }
    }
}

// --- androidTest assets: public-domain clips + goldens (D033) and the tiny Whisper model -------
// Generated under build/ only; nothing is committed under android/.
val androidTestClipsDir = layout.buildDirectory.dir("generated/androidTestClips")
val syncAndroidTestClips = tasks.register<Sync>("syncAndroidTestClips") {
    description = "Copies shared/testdata clips and goldens into the androidTest assets."
    group = "verification"
    from(sharedDirForTests.resolve("testdata/clips")) {
        include("*.wav")
        into("clips")
    }
    from(sharedDirForTests.resolve("testdata/goldens")) {
        include("*.json")
        into("goldens")
    }
    // The test-only catalog holding ggml-tiny-q8_0 (D037 a), so tests build a ModelCatalog for it.
    from(rootProject.layout.projectDirectory.file("asr/src/androidTest/assets/test_models.json")) {
        into("models")
    }
    into(androidTestClipsDir)
}

/**
 * Downloads one entry of the asr androidTest model catalog (D037 a: `ggml-tiny-q8_0` lives in
 * `:asr` `src/androidTest/assets/test_models.json`, not in shared/asr_models.json), verifies its
 * SHA-256 and places it under models/. A copy already in the local model cache (`.cache/models`,
 * `HABLA_MODEL_CACHE` or `-Phabla.modelCacheDir`) is reused when its hash matches.
 */
abstract class FetchTestModel : DefaultTask() {
    @get:InputFile
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val catalog: RegularFileProperty

    @get:Input
    abstract val modelId: Property<String>

    @get:Internal
    abstract val cacheDir: DirectoryProperty

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun fetch() {
        @Suppress("UNCHECKED_CAST")
        val models = (groovy.json.JsonSlurper().parse(catalog.get().asFile) as Map<String, Any>)["models"] as List<Map<String, Any>>
        val entry = models.firstOrNull { it["id"] == modelId.get() }
            ?: throw GradleException("model ${modelId.get()} not in ${catalog.get().asFile}")
        val name = entry["file"] as String
        val url = entry["url"] as String
        val sha = entry["sha256"] as String
        val outDir = outputDir.get().asFile.resolve("models").apply { mkdirs() }
        val target = outDir.resolve(name)
        if (target.isFile && sha256(target) == sha) return
        val cached = cacheDir.get().asFile.resolve(name)
        val tmp = outDir.resolve("$name.download")
        if (cached.isFile && sha256(cached) == sha) {
            cached.copyTo(tmp, overwrite = true)
        } else {
            logger.lifecycle("downloading $url")
            try {
                val conn = URI(url).toURL().openConnection() as HttpURLConnection
                conn.instanceFollowRedirects = true
                conn.setRequestProperty("User-Agent", "HablaAndroid-TestAssets")
                conn.inputStream.use { input -> tmp.outputStream().use { input.copyTo(it, 1 shl 20) } }
            } catch (e: java.io.IOException) {
                tmp.delete()
                throw GradleException("download failed for $url: ${e.message}", e)
            }
        }
        val got = sha256(tmp)
        if (got != sha) {
            tmp.delete()
            throw GradleException("SHA-256 mismatch for $url: expected $sha, got $got")
        }
        Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
    }

    private fun sha256(f: File): String {
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
}

val testModelInApk: Boolean = providers.gradleProperty("habla.testModelInApk").map { it.toBoolean() }.getOrElse(true)
val androidTestModelsDir = layout.buildDirectory.dir("generated/androidTestModels")
val fetchTestWhisperTiny = tasks.register<FetchTestModel>("fetchTestWhisperTiny") {
    description = "Downloads and verifies ggml-tiny-q8_0 for the emulator smoke test."
    group = "verification"
    catalog.set(rootProject.layout.projectDirectory.file("asr/src/androidTest/assets/test_models.json"))
    modelId.set("ggml-tiny-q8_0")
    cacheDir.set(
        ((findProperty("habla.modelCacheDir") as String?)?.let(::File)
            ?: System.getenv("HABLA_MODEL_CACHE")?.let(::File)
            ?: rootProject.file(".cache/models")),
    )
    outputDir.set(androidTestModelsDir)
}

android {
    sourceSets {
        getByName("androidTest") {
            assets.srcDir(androidTestClipsDir.get().asFile)
            // -Phabla.testModelInApk=false keeps the 43 MB model out of the test APK for an emulator
            // with little free /data; the tests then read the same pinned file from
            // /data/local/tmp/habla-models (adb push build/generated/androidTestModels/models/*).
            if (testModelInApk) assets.srcDir(androidTestModelsDir.get().asFile)
        }
    }
}
tasks.matching { it.name.startsWith("merge") && it.name.endsWith("AndroidTestAssets") }.configureEach {
    dependsOn(syncAndroidTestClips, fetchTestWhisperTiny)
}

dependencies {
    implementation(project(":shared-assets"))
    implementation(project(":audio"))
    implementation(project(":asr"))
    implementation(project(":translate"))
    implementation(project(":data"))

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.activity.compose)
    implementation(libs.core.ktx)
    implementation(libs.lifecycle.runtime.compose)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.navigation.compose)
    implementation(libs.koin.android)
    implementation(libs.koin.androidx.compose)
    implementation(libs.coroutines.android)
    implementation(libs.serialization.json)
    implementation(libs.onnxruntime.android)
    implementation(libs.mlkit.translate)
    // :asr pins OkHttp strictly (5.5.0's AAR needs compileSdk 37, D034); :translate's Ktor asks for
    // 5.5.0 without the pin, so the app graph must state the same strict version to resolve.
    implementation(libs.okhttp) { version { strictly(libs.versions.okhttp.get()) } }

    debugImplementation(libs.compose.ui.tooling)
    debugImplementation(libs.compose.ui.test.manifest)

    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
    testImplementation(libs.coroutines.test)
    testImplementation(libs.turbine)
    testImplementation(libs.mockk)
    testImplementation(libs.koin.test)

    androidTestImplementation(platform(libs.compose.bom))
    androidTestImplementation(libs.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.rules)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.uiautomator)
    androidTestImplementation(libs.koin.test)
    androidTestImplementation(libs.koin.test.junit4)
    androidTestImplementation(libs.coroutines.test)
}

// --- shared/ assets (owned by :shared-assets, design.md section 10) ---------------------------
// Packages the repo's shared/*.json into every variant as assets/shared/... with no second copy in
// git. sync (not copy) so a file deleted from shared/ also leaves the APK; testdata/ never ships.
abstract class SyncSharedAssets : DefaultTask() {
    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val sourceDir: DirectoryProperty

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @get:Inject
    abstract val fs: FileSystemOperations

    @TaskAction
    fun run() {
        fs.sync {
            from(sourceDir) {
                include("**/*.json")
                exclude("testdata/**")
            }
            into(outputDir.dir("shared"))
        }
    }
}

val repoSharedDir: File = rootProject.projectDir.resolveSibling("shared")
check(repoSharedDir.isDirectory) { "shared assets directory not found: ${repoSharedDir.absolutePath}" }
val staleSharedCopy = file("src/main/assets/shared")
check(!staleSharedCopy.exists()) {
    "${staleSharedCopy.absolutePath} must not exist: shared/ is packaged by sync<Variant>SharedAssets. " +
        "Delete the manual copy."
}

androidComponents {
    onVariants { variant ->
        val taskName = "sync${variant.name.replaceFirstChar { it.uppercase() }}SharedAssets"
        val syncTask = tasks.register<SyncSharedAssets>(taskName) {
            sourceDir.set(repoSharedDir)
        }
        checkNotNull(variant.sources.assets) { "variant ${variant.name} has no assets source set" }
            .addGeneratedSourceDirectory(syncTask, SyncSharedAssets::outputDir)
    }
}
