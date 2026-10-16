package com.xrmech.habla.app.ui.live

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.xrmech.habla.app.R

/**
 * Google's unaltered "powered by Google Translate" badge (white, for the dark theme), required
 * adjacent to every on-device translation (D050; cloud.google.com/translate/attribution).
 */
@Composable
fun GoogleAttribution(modifier: Modifier = Modifier) {
    Image(
        painter = painterResource(R.drawable.powered_by_google_translate),
        contentDescription = "powered by Google Translate",
        modifier = modifier.height(16.dp).testTag("google_attribution"),
    )
}
