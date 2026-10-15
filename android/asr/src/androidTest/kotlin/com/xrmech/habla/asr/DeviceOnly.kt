package com.xrmech.habla.asr

/**
 * Marks suites that need the physical Samsung Galaxy A56 (D5/D6). Excluded on emulators with
 * `-Pandroid.testInstrumentationRunnerArguments.notAnnotation=com.xrmech.habla.asr.DeviceOnly`.
 */
@Retention(AnnotationRetention.RUNTIME)
@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
annotation class DeviceOnly
