package com.xrmech.habla.app.ui.live

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import android.provider.Settings as AndroidSettings
import androidx.activity.compose.LocalActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SmallFloatingActionButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.xrmech.habla.app.pipeline.ExchangeCard
import com.xrmech.habla.app.pipeline.SessionPhase
import com.xrmech.habla.app.ui.nav.LocalWindowLayout
import com.xrmech.habla.app.ui.nav.Navigator
import com.xrmech.habla.app.ui.nav.Settings
import com.xrmech.habla.app.ui.theme.HablaColors
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.FlaggedPhrase
import com.xrmech.habla.translate.model.Mode
import java.util.Locale
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.koin.compose.viewmodel.koinViewModel

/** Every user event on Live; the screen wires them to [LiveViewModel] and the system. */
data class LiveActions(
    val listen: () -> Unit = {},
    val stop: () -> Unit = {},
    val setDirection: (Direction) -> Unit = {},
    val setMode: (Mode) -> Unit = {},
    val saveVocab: (ExchangeCard, FlaggedPhrase) -> Unit = { _, _ -> },
    val bannerAction: (BannerAction) -> Unit = {},
    val setup: SetupActions = SetupActions(),
    val confirmMeteredDownload: () -> Unit = {},
    val dismissMeteredDownload: () -> Unit = {},
)

@Composable
fun LiveScreen(navigator: Navigator, vm: LiveViewModel = koinViewModel()) {
    val state by vm.ui.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val activity = LocalActivity.current
    var focusBanner by remember { mutableStateOf<BannerKind?>(null) }

    val micLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        vm.setMicPermission(micStateAfterRequest(granted, activity))
    }
    val notificationLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        vm.setNotificationsGranted(granted)
    }
    LifecycleResumeEffect(Unit) {
        // Re-read on every resume: the user may have changed permissions in system settings.
        if (isGranted(context, Manifest.permission.RECORD_AUDIO)) {
            vm.setMicPermission(MicPermission.GRANTED)
        } else if (state.local.micPermission == MicPermission.GRANTED) {
            vm.setMicPermission(MicPermission.DENIED)
        }
        vm.setNotificationsGranted(Build.VERSION.SDK_INT < 33 || isGranted(context, Manifest.permission.POST_NOTIFICATIONS))
        onPauseOrDispose {}
    }
    LaunchedEffect(vm) {
        vm.effects.collect { e ->
            when (e) {
                LiveEffect.RequestMicPermission -> micLauncher.launch(Manifest.permission.RECORD_AUDIO)
                LiveEffect.OpenAppSettings -> openAppSettings(context)
                is LiveEffect.ScrollToBanner -> focusBanner = e.kind
            }
        }
    }
    val requestMic = {
        if (state.local.micPermission == MicPermission.PERMANENTLY_DENIED) openAppSettings(context) else micLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }
    val actions = LiveActions(
        listen = vm::onListen,
        stop = vm::onStop,
        setDirection = vm::setDirection,
        setMode = vm::setMode,
        saveVocab = vm::saveVocab,
        bannerAction = { a ->
            when (a) {
                BannerAction.GRANT_MIC -> requestMic()
                BannerAction.OPEN_APP_SETTINGS -> openAppSettings(context)
                BannerAction.DOWNLOAD_MODEL, BannerAction.RETRY_DOWNLOAD -> vm.downloadModel()
                BannerAction.DELETE_AND_RETRY -> vm.deleteAndRetry()
                BannerAction.CANCEL_DOWNLOAD -> vm.cancelDownload()
                BannerAction.OPEN_SETTINGS -> navigator.navigate(Settings(Settings.KEY))
                BannerAction.ADD_KEY -> navigator.navigate(Settings(Settings.KEY))
            }
        },
        setup = SetupActions(
            grantMic = requestMic,
            openAppSettings = { openAppSettings(context) },
            allowNotifications = {
                if (Build.VERSION.SDK_INT >= 33) notificationLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            },
            downloadModel = { vm.downloadModel() },
            addKey = { navigator.navigate(Settings(Settings.KEY)) },
            skipKey = vm::skipKey,
            done = vm::dismissSetup,
        ),
        confirmMeteredDownload = { vm.downloadModel(confirmed = true) },
        dismissMeteredDownload = vm::dismissMeteredDownload,
    )
    LiveContent(state, actions, focusBanner, onBannerFocused = { focusBanner = null })
}

private fun isGranted(context: Context, permission: String) =
    ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED

/** Denied with no rationale to show after a request = the system will not ask again. */
private fun micStateAfterRequest(granted: Boolean, activity: Activity?): MicPermission = when {
    granted -> MicPermission.GRANTED
    activity != null && !activity.shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO) -> MicPermission.PERMANENTLY_DENIED
    else -> MicPermission.DENIED
}

private fun openAppSettings(context: Context) {
    context.startActivity(
        Intent(AndroidSettings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null))
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
    )
}

/** Stateless Live layout (design section 6, Live); UI tests drive it with a fake state. */
@Composable
fun LiveContent(
    state: LiveUiState,
    actions: LiveActions,
    focusBanner: BannerKind? = null,
    onBannerFocused: () -> Unit = {},
    nowMs: () -> Long = SystemClock::elapsedRealtime,
) {
    val short = LocalWindowLayout.current.shortHeight
    Column(Modifier.fillMaxSize()) {
        TopBar(state, actions, showListen = short)
        StatusRow(state, nowMs)
        BannerArea(state.banners, state.allBanners, actions.bannerAction, focusBanner, onBannerFocused)
        Box(Modifier.weight(1f).fillMaxWidth()) {
            Transcript(state, actions)
        }
        if (!short) {
            ListenButton(state, actions, Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp).testTag("listen_bottom"))
        }
    }
    if (state.local.confirmMeteredDownload) {
        AlertDialog(
            onDismissRequest = actions.dismissMeteredDownload,
            title = { Text("Download on mobile data?") },
            text = { Text("The speech model is ${(state.modelSizeBytes + 500_000) / 1_000_000} MB and this network is metered.") },
            confirmButton = { TextButton(onClick = actions.confirmMeteredDownload) { Text("Download") } },
            dismissButton = { TextButton(onClick = actions.dismissMeteredDownload) { Text("Cancel") } },
        )
    }
}

@Composable
private fun TopBar(state: LiveUiState, actions: LiveActions, showListen: Boolean) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp).testTag("live_top_bar"),
    ) {
        Text("Habla", style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.weight(1f))
        Selector(
            label = state.directionLabel,
            options = Direction.entries.map { it to directionOption(it) },
            enabled = state.settings != null,
            onSelect = actions.setDirection,
            tag = "direction_selector",
        )
        Selector(
            label = state.settings?.mode?.let(::modeLabel) ?: "",
            options = Mode.entries.map { it to modeLabel(it) },
            enabled = state.settings != null,
            onSelect = actions.setMode,
            tag = "mode_selector",
        )
        if (showListen) ListenButton(state, actions, Modifier.testTag("listen_top"))
    }
}

private fun directionOption(d: Direction) = when (d) {
    Direction.ES_TO_EN -> "ES -> EN"
    Direction.EN_TO_ES -> "EN -> ES"
    Direction.AUTO -> "Auto"
}

private fun modeLabel(m: Mode) = m.name.lowercase().replaceFirstChar { it.titlecase(Locale.ROOT) }

@Composable
private fun <T> Selector(label: String, options: List<Pair<T, String>>, enabled: Boolean, onSelect: (T) -> Unit, tag: String) {
    var open by remember { mutableStateOf(false) }
    Box {
        TextButton(onClick = { open = true }, enabled = enabled, modifier = Modifier.testTag(tag)) { Text(label) }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            options.forEach { (value, text) ->
                DropdownMenuItem(text = { Text(text) }, onClick = { open = false; onSelect(value) }, modifier = Modifier.testTag("${tag}_$text"))
            }
        }
    }
}

@Composable
private fun ListenButton(state: LiveUiState, actions: LiveActions, modifier: Modifier) {
    if (state.canStop) {
        Button(
            onClick = actions.stop,
            colors = ButtonDefaults.buttonColors(containerColor = HablaColors.ErrorFill, contentColor = Color.White),
            modifier = modifier.heightIn(min = 48.dp),
        ) { Text(if (state.live.phase is SessionPhase.Finishing) "Finishing..." else "Stop") }
    } else {
        Button(onClick = actions.listen, enabled = state.canListen, modifier = modifier.heightIn(min = 48.dp)) {
            Text(if (state.local.startRequested) "Starting..." else "Listen")
        }
    }
}

@Composable
private fun StatusRow(state: LiveUiState, nowMs: () -> Long) {
    val phase = state.live.phase
    val started = state.live.startedAtMs
    var now by remember { mutableLongStateOf(nowMs()) }
    LaunchedEffect(started, phase.active) {
        while (started != null && phase.active) {
            now = nowMs()
            delay(1_000)
        }
    }
    val (label, color) = when (phase) {
        SessionPhase.Idle -> "Idle" to HablaColors.TextMuted
        SessionPhase.Starting -> "Starting" to HablaColors.Warning
        SessionPhase.Calibrating -> "Calibrating" to HablaColors.Warning
        SessionPhase.Listening -> "Listening ${started?.let { clock(now - it) } ?: ""}".trim() to HablaColors.Success
        is SessionPhase.Finishing -> "Finishing" to HablaColors.Warning
        is SessionPhase.Failed -> "Stopped" to HablaColors.ErrorText
    }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp).testTag("status_row"),
    ) {
        Box(Modifier.size(10.dp).background(color, CircleShape))
        Text(label, modifier = Modifier.testTag("status_label"))
        if (state.live.queueDepth > 0) Text("Queue ${state.live.queueDepth}", color = HablaColors.TextMuted)
        Spacer(Modifier.weight(1f))
        Text(
            "$" + String.format(Locale.US, "%.3f", state.live.costs.sessionCostUsd),
            color = HablaColors.TextMuted,
            modifier = Modifier.testTag("status_cost"),
        )
    }
}

internal fun clock(ms: Long): String {
    val s = (ms.coerceAtLeast(0) / 1000)
    return "%d:%02d".format(s / 60, s % 60)
}

@Composable
private fun BannerArea(
    shown: BannerSet,
    all: List<Banner>,
    onAction: (BannerAction) -> Unit,
    focus: BannerKind?,
    onFocused: () -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    LaunchedEffect(focus) {
        if (focus != null) {
            if (shown.shown.none { it.kind == focus }) expanded = true
            delay(2_000)
            onFocused()
        }
    }
    val visible = if (expanded) all else shown.shown
    if (visible.isEmpty()) return
    Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        visible.forEach { b -> BannerRow(b, onAction, highlighted = b.kind == focus) }
        if (!expanded && shown.more > 0) {
            TextButton(onClick = { expanded = true }, modifier = Modifier.testTag("banner_more")) { Text("+${shown.more} more") }
        } else if (expanded && all.size > BannerReducer.MAX_SHOWN) {
            TextButton(onClick = { expanded = false }) { Text("Show fewer") }
        }
    }
}

@Composable
private fun BannerRow(b: Banner, onAction: (BannerAction) -> Unit, highlighted: Boolean) {
    val fg = when (b.severity) {
        Severity.FATAL, Severity.ERROR -> HablaColors.ErrorText
        Severity.WARNING -> HablaColors.Warning
        Severity.INFO -> HablaColors.AccentText
    }
    Surface(
        color = if (highlighted) HablaColors.Raised2 else HablaColors.Surface,
        shape = MaterialTheme.shapes.small,
        modifier = Modifier.fillMaxWidth().testTag("banner_${b.kind.name}"),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(start = 12.dp, end = 4.dp, top = 4.dp, bottom = 4.dp)) {
            Box(Modifier.width(4.dp).heightIn(min = 24.dp).background(fg))
            Text(b.text, color = fg, modifier = Modifier.weight(1f).padding(start = 8.dp, top = 6.dp, bottom = 6.dp))
            b.action?.let { a ->
                TextButton(onClick = { onAction(a) }, modifier = Modifier.testTag("banner_${b.kind.name}_action")) { Text(a.label) }
            }
        }
    }
}

@Composable
private fun Transcript(state: LiveUiState, actions: LiveActions) {
    val list = rememberLazyListState()
    val scope = rememberCoroutineScope()
    var follow by remember { mutableStateOf(true) }
    val cards = state.live.exchanges
    val partial = state.live.partial
    val setup = state.setup
    val itemCount = cards.size + (if (partial != null) 1 else 0) + (if (setup != null) 1 else 0)

    // Follow the newest card only while the user is at the bottom (web scrollIfAtBottom).
    LaunchedEffect(list) {
        snapshotFlow { list.isScrollInProgress to list.canScrollForward }.collect { (scrolling, canForward) ->
            if (scrolling) follow = !canForward
            else if (!canForward) follow = true
        }
    }
    LaunchedEffect(itemCount, partial?.text) {
        if (follow && itemCount > 0) list.scrollToItem(itemCount - 1)
    }

    LazyColumn(
        state = list,
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.fillMaxSize().testTag("transcript"),
    ) {
        if (setup != null) {
            item("setup") {
                SetupCard(setup, state.local.micPermission, state.status.modelState, state.modelSizeBytes, actions.setup)
            }
        }
        if (cards.isEmpty() && partial == null && setup == null) {
            item("empty") {
                Text(
                    if (state.live.phase.active) "Listening - speak and the transcript appears here." else "Tap Listen to start transcribing.",
                    color = HablaColors.TextMuted,
                    modifier = Modifier.testTag("transcript_empty"),
                )
            }
        }
        items(cards, key = { "card-${it.seq}" }) { card ->
            ExchangeCardView(
                card = card,
                isSaved = { p -> state.isSaved(card, p) },
                onSave = { p -> actions.saveVocab(card, p) },
                onAction = actions.bannerAction,
            )
        }
        if (partial != null) {
            item("partial") { PartialCard(partial, translationEnabled = state.settings?.partialTranslation == true) }
        }
        state.local.vocabError?.let { msg -> item("vocab_error") { Text(msg, color = HablaColors.ErrorText) } }
    }
    if (!follow && itemCount > 0) {
        Box(Modifier.fillMaxSize().padding(16.dp), contentAlignment = Alignment.BottomEnd) {
            SmallFloatingActionButton(
                onClick = {
                    follow = true
                    scope.launch { list.animateScrollToItem(itemCount - 1) }
                },
                modifier = Modifier.testTag("jump_latest"),
            ) { Text("Latest", modifier = Modifier.padding(horizontal = 12.dp)) }
        }
    }
}
