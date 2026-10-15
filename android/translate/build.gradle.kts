plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(project(":shared-assets"))

    // Exposed as api: public signatures use HttpClient (OpenAiHttp.create), StateFlow/SharedFlow
    // (CostMeter, ResilientProvider) and @Serializable model types read by :data.
    api(libs.ktor.client.core)
    api(libs.serialization.json)
    api(libs.coroutines.core)
    implementation(libs.ktor.client.okhttp)
    implementation(libs.ktor.client.logging)

    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.ktor.client.mock)
    testImplementation(libs.coroutines.test)
    testImplementation(libs.kotlin.test.junit5)
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

// The Gradle root is android/, so the repo's shared/ directory is its sibling. Tests load the real
// assets through FileAssetSource.fromSystemProperty().
val repoSharedDir: File = rootProject.projectDir.resolveSibling("shared")

tasks.test {
    useJUnitPlatform()
    inputs.dir(repoSharedDir)
        .withPathSensitivity(PathSensitivity.RELATIVE)
        .withPropertyName("sharedDir")
    systemProperty("habla.sharedDir", repoSharedDir.absolutePath)
    // The live OpenAI test spends real money and needs a real key; it never runs in CI.
    if (System.getenv("CI") != null) exclude("**/*LiveTest*")
}
