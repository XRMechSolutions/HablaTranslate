package com.xrmech.habla.app.seed

import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Test

class BuildSeedTest {
    @Test
    fun `should carry no key when the flavor is public`() {
        assertNull(BuildSeed.key)
    }
}
