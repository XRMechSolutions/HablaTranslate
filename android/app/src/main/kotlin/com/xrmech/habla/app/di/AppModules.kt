package com.xrmech.habla.app.di

import android.content.Context
import android.media.AudioManager
import android.net.ConnectivityManager
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import com.xrmech.habla.app.BuildConfig
import com.xrmech.habla.app.capture.MicAudioSource
import com.xrmech.habla.app.pipeline.AudioSegmenterPort
import com.xrmech.habla.app.pipeline.LiveController
import com.xrmech.habla.app.pipeline.PipelineCoordinator
import com.xrmech.habla.app.pipeline.PipelineDispatchers
import com.xrmech.habla.app.pipeline.RecognizerHolder
import com.xrmech.habla.app.pipeline.SegmenterFactory
import com.xrmech.habla.app.pipeline.TrainingCapture
import com.xrmech.habla.app.seed.BuildSeed
import com.xrmech.habla.app.service.ListeningNotification
import com.xrmech.habla.app.service.ServiceAudioSourceFactory
import com.xrmech.habla.app.service.ServiceController
import com.xrmech.habla.app.service.SessionControl
import com.xrmech.habla.app.service.WakeLockHolder
import com.xrmech.habla.app.state.AppSettingsDefaults
import com.xrmech.habla.app.state.AppSettingsState
import com.xrmech.habla.app.state.ConnectivityMonitor
import com.xrmech.habla.app.state.KeyHealthTracker
import com.xrmech.habla.app.state.ModelDownloadController
import com.xrmech.habla.app.state.KeyState
import com.xrmech.habla.app.state.NoProAccess
import com.xrmech.habla.app.state.ProAccess
import com.xrmech.habla.app.state.ThermalMonitor
import com.xrmech.habla.app.translate.LanguagePack
import com.xrmech.habla.app.translate.MlKitTranslator
import com.xrmech.habla.app.translate.PackState
import com.xrmech.habla.app.ui.history.ExportController
import com.xrmech.habla.app.ui.history.ExportPort
import com.xrmech.habla.app.ui.history.HistoryViewModel
import com.xrmech.habla.app.ui.history.SessionDetailViewModel
import com.xrmech.habla.app.ui.live.LiveViewModel
import com.xrmech.habla.app.ui.settings.KeyTester
import com.xrmech.habla.app.ui.settings.SettingsViewModel
import com.xrmech.habla.app.ui.vocab.VocabViewModel
import com.xrmech.habla.asr.KtorModelDownloader
import com.xrmech.habla.asr.LanguageVoter
import com.xrmech.habla.asr.ModelDownloader
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.audio.SileroVad
import com.xrmech.habla.data.ApiKeyStore
import com.xrmech.habla.data.DataStoreSettingsRepository
import com.xrmech.habla.data.HablaDatabaseFactory
import com.xrmech.habla.data.HistoryRepository
import com.xrmech.habla.data.KeystoreApiKeyStore
import com.xrmech.habla.data.SettingsRepository
import com.xrmech.habla.data.SqlHistoryRepository
import com.xrmech.habla.data.SqlUsageRepository
import com.xrmech.habla.data.SqlVocabRepository
import com.xrmech.habla.data.TrainingAudioStore
import com.xrmech.habla.data.TrainingExporter
import com.xrmech.habla.data.UsageRepository
import com.xrmech.habla.data.VocabRepository
import com.xrmech.habla.shared.AssetSource
import com.xrmech.habla.shared.SharedAssets
import com.xrmech.habla.translate.ConversationContext
import com.xrmech.habla.translate.CostMeter
import com.xrmech.habla.translate.GatewayClient
import com.xrmech.habla.translate.GatewayTranslator
import com.xrmech.habla.translate.IdTokenSource
import com.xrmech.habla.translate.IdiomScanner
import com.xrmech.habla.translate.OnDeviceTranslator
import com.xrmech.habla.translate.OpenAiHttp
import com.xrmech.habla.translate.OpenAiProvider
import com.xrmech.habla.translate.QuickTranslator
import com.xrmech.habla.translate.ResilientProvider
import com.xrmech.habla.translate.TopicTracker
import com.xrmech.habla.translate.TranslationProvider
import com.xrmech.habla.translate.Translator
import io.ktor.client.HttpClient
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import org.koin.android.ext.koin.androidContext
import org.koin.core.module.dsl.viewModel
import org.koin.core.qualifier.named
import org.koin.dsl.bind
import org.koin.dsl.binds
import org.koin.dsl.module

/** Qualifier of the process-lifetime CoroutineScope (SupervisorJob + Default). */
val APP_SCOPE = named("app")

val sharedAssetsModule = module {
    single<AssetSource> { AndroidAssetSource(androidContext().assets) }
    single { SharedAssets(get()) }
}

val audioModule = module {
    single<SegmenterFactory> { AudioSegmenterPort.FACTORY }
}

val asrModule = module {
    single<ModelDownloader> { KtorModelDownloader() }
    // Models live under noBackupFilesDir: hundreds of MB of re-downloadable weights stay out of Auto Backup (R5).
    single { ModelManager(androidContext().noBackupFilesDir, get<SharedAssets>().asrModels, get()) }
    single { LanguageVoter(get<SharedAssets>().asrProfile) }
    single { RecognizerHolder(get(), get(), Dispatchers.IO) }
}

val translateModule = module {
    // HablaApp sets TranslateLogging.sink before Koin starts, so this client logs through HLog.
    single<HttpClient> { OpenAiHttp.create(OpenAiHttp.okHttpEngine(), BuildConfig.DEBUG) }
    single { OpenAiProvider(get(), { get<ApiKeyStore>().get() }, get<SharedAssets>().models) }
    single<TranslationProvider> { ResilientProvider(get<OpenAiProvider>()) }
    single<KeyTester> { KeyTester.of(get()) }
    single { CostMeter(get<SharedAssets>().models) }
    single { MlKitTranslator() } binds arrayOf(OnDeviceTranslator::class, LanguagePack::class)
    // Pro (D051): Firebase sign-in is not wired yet, so the token source is signed out and the
    // gateway path falls back to the phone without a network call.
    single<IdTokenSource> { IdTokenSource { null } }
    single<ProAccess> { NoProAccess }
    if (BuildConfig.GATEWAY_URL.isNotEmpty()) {
        // Its own pinned HttpClient, built inline so it never resolves the OpenAI client.
        single {
            GatewayClient(
                GatewayClient.http(GatewayClient.okHttpEngine(BuildConfig.GATEWAY_URL), BuildConfig.DEBUG),
                BuildConfig.GATEWAY_URL, BuildConfig.VERSION_NAME, get(),
            )
        }
        single { GatewayTranslator(get(), get<SharedAssets>().asrProfile.context.maxContextExchanges) }
    }
    single {
        Translator(
            get(), get(), { get<AppSettingsState>().translateSettings(get<SharedAssets>().models) }, get(), get(),
            getOrNull<GatewayTranslator>(),
        )
    }
    single { QuickTranslator(get(), get(), { get<AppSettingsState>().translateSettings(get<SharedAssets>().models) }, get(), get()) }
    single { ConversationContext(get<SharedAssets>().asrProfile) }
    single { TopicTracker(get(), get()) }
    single { IdiomScanner(get<SharedAssets>().idioms) }
}

val dataModule = module {
    single<ApiKeyStore> { KeystoreApiKeyStore(androidContext(), seedKey = BuildSeed.key) }
    single { AppSettingsDefaults(get()) }
    single<SettingsRepository> { DataStoreSettingsRepository(androidContext(), get<AppSettingsDefaults>().value) }
    single { HablaDatabaseFactory.create(androidContext()) }
    single<HistoryRepository> { SqlHistoryRepository(get(), androidContext().filesDir) }
    single<VocabRepository> { SqlVocabRepository(get()) }
    single<UsageRepository> { SqlUsageRepository(get()) }
    single { TrainingAudioStore(androidContext().filesDir, get()) }
    single { TrainingExporter(androidContext().filesDir, get(), get()) }
}

val appModule = module {
    single<CoroutineScope>(APP_SCOPE) { CoroutineScope(SupervisorJob() + Dispatchers.Default) }
    single { androidContext().getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager }
    single { androidContext().getSystemService(Context.POWER_SERVICE) as PowerManager }
    single { androidContext().getSystemService(Context.AUDIO_SERVICE) as AudioManager }

    single {
        AppSettingsState(
            get(), get(APP_SCOPE),
            hasKey = { get<KeyHealthTracker>().state.value != KeyState.ABSENT },
            packReady = { get<LanguagePack>().state.value == PackState.Ready },
            proActive = { get<ProAccess>().active.value },
        )
    }
    single { KeyHealthTracker(get<ApiKeyStore>().hasKey, get(APP_SCOPE)) }
    single { ConnectivityMonitor(get(), get(APP_SCOPE)) }
    single { ThermalMonitor(get<PowerManager>(), get(APP_SCOPE)) }
    single { ModelDownloadController(get(), get(APP_SCOPE)) { get<ConnectivityManager>().isActiveNetworkMetered } }
    single { ServiceController(androidContext()) } bind SessionControl::class
    single { ExportController(androidContext(), get(), Dispatchers.IO) } bind ExportPort::class
    single { ListeningNotification(androidContext()) }
    single { WakeLockHolder(get()) }
    single { ServiceAudioSourceFactory { service -> MicAudioSource(service, get()) } }
    single { TrainingCapture.of(get()) }

    single {
        PipelineCoordinator(
            assets = get(),
            segmenterFactory = get(),
            // Per session, never a singleton: SileroVad is single-thread and per-session state (audio docs).
            vadFactory = { runCatching { SileroVad(SileroVad.bundledModelBytes()) }.getOrNull() },
            recognizers = get(),
            voter = get(),
            translator = get(),
            quickTranslator = get(),
            topicTracker = get(),
            context = get(),
            idiomScanner = get(),
            costMeter = get(),
            settings = get(),
            keyStore = get(),
            keyHealth = get(),
            history = get(),
            usage = get(),
            training = get(),
            thermal = get(),
            dispatchers = PipelineDispatchers.production(
                captureThreadSetup = { Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO) },
                segmenterThreadSetup = { Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO) },
            ),
            clock = SystemClock::elapsedRealtime,
        )
    } binds arrayOf(LiveController::class)

    viewModel {
        LiveViewModel(
            live = get(), service = get(), settings = get(), models = get(), downloads = get(), keyHealth = get(),
            online = get<ConnectivityMonitor>().online, vocab = get(), assets = get(), pack = get(), pro = get(),
        )
    }
    viewModel { HistoryViewModel(get(), get()) }
    viewModel { p -> SessionDetailViewModel(p.get(), get(), get(), get()) }
    viewModel { VocabViewModel(get()) }
    viewModel {
        SettingsViewModel(
            settings = get(), keyStore = get(), tester = get(), costMeter = get(), usage = get(), keyHealth = get(),
            models = get(), downloads = get(), assets = get(), live = get(), pack = get(),
        )
    }
}

val appModules = listOf(sharedAssetsModule, audioModule, asrModule, translateModule, dataModule, appModule)
