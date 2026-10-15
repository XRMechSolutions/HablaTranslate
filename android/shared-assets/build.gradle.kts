plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    // api: public types are @Serializable and consumers may encode them.
    api(libs.serialization.json)

    testImplementation(kotlin("test"))
    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

// The Gradle root is android/, so the repo's shared/ directory is its sibling.
val repoSharedDir: File = rootProject.projectDir.resolveSibling("shared")
check(repoSharedDir.isDirectory) {
    "shared assets directory not found: ${repoSharedDir.absolutePath}"
}

tasks.test {
    useJUnitPlatform()
    inputs.dir(repoSharedDir)
        .withPathSensitivity(PathSensitivity.RELATIVE)
        .withPropertyName("sharedDir")
    systemProperty("habla.sharedDir", repoSharedDir.absolutePath)
}
