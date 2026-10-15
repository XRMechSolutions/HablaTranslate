// :audio - Kotlin JVM port of the server's audio front end (Segmenter, Silero VAD, partial/final prep).
// No Android plugin, no coroutines, no logging library. ONNX Runtime is compileOnly: :app supplies
// onnxruntime-android at runtime, the JVM tests use the desktop jar (R2, dependencies.md).
plugins {
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(project(":shared-assets"))
    compileOnly(libs.onnxruntime.jvm)

    testImplementation(libs.onnxruntime.jvm)
    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testImplementation(libs.kotlin.test.junit5)
    testRuntimeOnly(libs.junit.platform.launcher)
}

// Tests read the real shared/ directory (asr_profile.json, testdata/clips) in place.
val repoSharedDir: File = rootProject.projectDir.resolveSibling("shared")
check(repoSharedDir.isDirectory) { "shared/ not found at $repoSharedDir" }

// Windows only: ORT 1.31's onnxruntime.dll needs the MSVC 14.40+ C++ runtime, but JDK 17 builds
// ship an older msvcp140.dll in bin/, which the JVM loads at startup and ORT then binds to; creating
// the OrtEnvironment crashes the test JVM inside std::mutex. The tests therefore run on a copy of
// the toolchain JDK without that DLL, so the system's (VC++ redistributable) copy is used.
// The copy is made while the test task is configured because Gradle probes the executable then.
fun windowsTestJdk(): File {
    val home = javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(17)) }
        .get().metadata.installationPath.asFile
    val copy = layout.buildDirectory.dir("test-jdk").get().asFile
    val stamp = File(copy, "habla-source.txt")
    val source = home.absolutePath + "\n" + File(home, "release").readText()
    if (!stamp.isFile || stamp.readText() != source) {
        copy.deleteRecursively()
        home.walkTopDown()
            .onEnter { it == home || it.relativeTo(home).invariantSeparatorsPath.substringBefore('/') in setOf("bin", "conf", "lib") }
            .filter { it.isFile && it.name != "src.zip" && it.relativeTo(home).invariantSeparatorsPath != "bin/msvcp140.dll" }
            .forEach { it.copyTo(File(copy, it.relativeTo(home).path), overwrite = true) }
        File(home, "release").copyTo(File(copy, "release"), overwrite = true)
        stamp.writeText(source)
    }
    return File(copy, "bin/java.exe")
}

tasks.test {
    useJUnitPlatform()
    inputs.dir(repoSharedDir).withPathSensitivity(PathSensitivity.RELATIVE).withPropertyName("sharedDir")
    systemProperty("habla.sharedDir", repoSharedDir.absolutePath)
    // The 30 s FinalPrep and full-session parity runs allocate a few hundred MB at most.
    maxHeapSize = "1g"
    if (System.getProperty("os.name").startsWith("Windows")) {
        executable = windowsTestJdk().absolutePath
    }
}
