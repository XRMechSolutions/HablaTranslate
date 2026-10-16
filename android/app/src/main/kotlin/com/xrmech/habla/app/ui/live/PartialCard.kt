package com.xrmech.habla.app.ui.live

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import com.xrmech.habla.app.pipeline.PartialLine
import com.xrmech.habla.app.ui.theme.HablaColors

/** The in-progress line at the end of the transcript ("Live" label, cursor, dimmed translation). */
@Composable
fun PartialCard(partial: PartialLine, translationEnabled: Boolean, modifier: Modifier = Modifier) {
    Card(
        colors = CardDefaults.cardColors(containerColor = HablaColors.Surface),
        border = BorderStroke(1.dp, HablaColors.Outline),
        modifier = modifier.fillMaxWidth().testTag("partial_card"),
    ) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                if (partial.finalizing) "Live - finishing" else "Live - ${partial.language.uppercase()}",
                style = MaterialTheme.typography.labelSmall,
                color = HablaColors.AccentText,
            )
            Text("${partial.text}▍", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.testTag("partial_text"))
            when {
                partial.translation != null -> {
                    Text(partial.translation, color = HablaColors.TextMuted, modifier = Modifier.testTag("partial_translation"))
                    if (partial.translatedOnDevice) GoogleAttribution()
                }
                translationEnabled -> Text("Translating...", color = HablaColors.TextMuted)
            }
        }
    }
}
