plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.sqldelight)
}

android {
    namespace = "com.xrmech.habla.data"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    // :data never sees a key or a flavor value; a generated build-config class would only be a place for one to leak.
    buildFeatures {
        buildConfig = false
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // android.util.Log calls become no-ops in JVM unit tests, so no Robolectric is needed.
    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

sqldelight {
    databases {
        create("HablaDatabase") {
            packageName.set("com.xrmech.habla.data.db")
            // Android 10 (minSdk 29) ships SQLite 3.22: the 3.18 dialect makes UPSERT, RETURNING
            // and window functions compile errors instead of runtime failures on old devices.
            dialect(libs.sqldelight.dialect.sqlite318)
            schemaOutputDirectory.set(file("src/main/sqldelight/databases"))
            verifyMigrations.set(true)
        }
    }
}

dependencies {
    api(project(":translate"))

    implementation(libs.sqldelight.android.driver)
    implementation(libs.sqldelight.coroutines)
    implementation(libs.datastore.preferences)
    implementation(libs.coroutines.android)
    implementation(libs.serialization.json)

    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
    testImplementation(libs.coroutines.test)
    testImplementation(libs.turbine)
    testImplementation(libs.sqldelight.sqlite.driver)

    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.junit4)
    androidTestImplementation(libs.coroutines.test)

    constraints {
        // :translate's ktor-client-okhttp resolves okhttp 5.5.0, whose AAR demands compileSdk 37;
        // D034 keeps compileSdk 36. Same pin as :app. A constraint only, so :data never adds okhttp.
        implementation(libs.okhttp) { version { strictly(libs.versions.okhttp.get()) } }
    }
}

// The default unit test run skips @Tag("large"); `largeTest` runs only those, under a 64 MB heap,
// to prove the training export streams instead of buffering whole WAVs.
tasks.withType<Test>().configureEach {
    if (name != "largeTest") {
        useJUnitPlatform {
            excludeTags("large")
        }
    }
}

tasks.register<Test>("largeTest") {
    description = "Runs @Tag(\"large\") JVM tests (bounded-memory export) with a 64 MB heap."
    group = "verification"
    val unitTest = tasks.named<Test>("testDebugUnitTest").get()
    testClassesDirs = unitTest.testClassesDirs
    classpath = unitTest.classpath
    useJUnitPlatform {
        includeTags("large")
    }
    maxHeapSize = "64m"
    failOnNoDiscoveredTests.set(false)
}
