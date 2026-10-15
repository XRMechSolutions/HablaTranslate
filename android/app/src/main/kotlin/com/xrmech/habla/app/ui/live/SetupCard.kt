package com.xrmech.habla.app.ui.live

import android.os.Build
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import com.xrmech.habla.app.ui.theme.HablaColors
import com.xrmech.habla.asr.ModelState

data class SetupActions(
    val grantMic: () -> Unit = {},
    val openAppSettings: () -> Unit = {},
    val allowNotifications: () -> Unit = {},
    val downloadModel: () -> Unit = {},
    val addKey: () -> Unit = {},
    val skipKey: () -> Unit = {},
    val done: () -> Unit = {},
)

/** First-run checklist shown on Live while `firstRunDone` is false (design section 6, Setup card). */
@Composable
fun SetupCard(
    rows: SetupRows,
    micPermission: MicPermission,
    modelState: ModelState,
    modelSizeBytes: Long,
    actions: SetupActions,
    modifier: Modifier = Modifier,
) {
    Card(colors = CardDefaults.cardColors(containerColor = HablaColors.Raised), modifier = modifier.fillMaxWidth().testTag("setup_card")) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Get ready to listen", style = MaterialTheme.typography.titleMedium)
            SetupRow(
                "Microphone",
                if (rows.micDone) "Granted" else "Needed to hear speech",
                rows.micDone,
                when {
                    rows.micDone -> null
                    micPermission == MicPermission.PERMANENTLY_DENIED -> "Open settings" to actions.openAppSettings
                    else -> "Grant" to actions.grantMic
                },
                "setup_mic",
            )
            if (Build.VERSION.SDK_INT >= 33) {
                SetupRow(
                    "Notifications",
                    if (rows.notificationsDone) "Allowed" else "Notifications off - stop listening from the app",
                    rows.notificationsDone,
                    if (rows.notificationsDone) null else "Allow" to actions.allowNotifications,
                    "setup_notifications",
                )
            }
            val mb = (modelSizeBytes + 500_000) / 1_000_000
            SetupRow(
                "Speech model",
                when (modelState) {
                    is ModelState.Ready -> "Ready"
                    is ModelState.Downloading -> "Downloading ${(modelState.bytes + 500_000) / 1_000_000} / $mb MB"
                    ModelState.Verifying -> "Verifying..."
                    is ModelState.Failed -> "Download failed: ${ModelFailureText.of(modelState.reason)}"
                    ModelState.Missing -> "Not downloaded ($mb MB)"
                },
                rows.modelDone,
                if (modelState is ModelState.Missing || modelState is ModelState.Failed) "Download $mb MB" to actions.downloadModel else null,
                "setup_model",
            )
            SetupRow(
                "OpenAI key",
                if (rows.keyDone) "Saved or skipped" else "For translations",
                rows.keyDone,
                if (rows.keyDone) null else "Add key" to actions.addKey,
                "setup_key",
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (!rows.keyDone) {
                    TextButton(onClick = actions.skipKey, modifier = Modifier.testTag("setup_skip_key")) { Text("Skip - transcribe only") }
                }
                TextButton(onClick = actions.done, modifier = Modifier.testTag("setup_done")) { Text("Done") }
            }
        }
    }
}

@Composable
private fun SetupRow(title: String, detail: String, done: Boolean, action: Pair<String, () -> Unit>?, tag: String) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().testTag(tag)) {
        Column(Modifier.weight(1f)) {
            Text(if (done) "$title  ✓" else title, color = if (done) HablaColors.Success else HablaColors.Text)
            Text(detail, style = MaterialTheme.typography.bodySmall, color = HablaColors.TextMuted)
        }
        action?.let { (label, onClick) -> Button(onClick = onClick, modifier = Modifier.testTag("${tag}_action")) { Text(label) } }
    }
}
