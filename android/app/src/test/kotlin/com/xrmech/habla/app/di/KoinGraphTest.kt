package com.xrmech.habla.app.di

import android.app.Application
import android.content.Context
import androidx.lifecycle.SavedStateHandle
import com.xrmech.habla.app.pipeline.RecognizerHolder
import com.xrmech.habla.app.ui.history.SessionDetailViewModel
import com.xrmech.habla.asr.LanguageVoter
import com.xrmech.habla.asr.ModelManager
import com.xrmech.habla.shared.AssetSource
import com.xrmech.habla.shared.AsrProfile
import com.xrmech.habla.shared.ModelCatalog
import com.xrmech.habla.shared.ModelTable
import com.xrmech.habla.translate.ConversationContext
import com.xrmech.habla.translate.CostMeter
import com.xrmech.habla.translate.IdiomScanner
import com.xrmech.habla.translate.OpenAiProvider
import com.xrmech.habla.translate.QuickTranslator
import com.xrmech.habla.translate.Translator
import io.ktor.client.HttpClient
import io.ktor.client.engine.HttpClientEngine
import java.io.File
import kotlinx.coroutines.CoroutineDispatcher
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test
import org.koin.core.error.DefinitionOverrideException
import org.koin.dsl.koinApplication
import org.koin.dsl.module
import org.koin.test.verify.definition
import org.koin.test.verify.injectedParameters
import org.koin.test.verify.verify

/**
 * `verify()` walks every bound class's constructor (Koin quirk: it cannot see what a definition
 * lambda passes). Android framework types are [extraTypes]; values a definition passes inline
 * (a directory, a dispatcher, a SharedAssets section, a lambda) are declared per definition with
 * `injectedParameters`, so a new constructor dependency anywhere else still fails this test.
 */
class KoinGraphTest {
    @Test
    fun `should resolve every constructor dependency of the app graph`() {
        module { includes(appModules) }.verify(
            extraTypes = listOf(Context::class, Application::class, SavedStateHandle::class),
            injections = injectedParameters(
                definition<ModelManager>(File::class, ModelCatalog::class),
                definition<LanguageVoter>(AsrProfile::class),
                definition<ConversationContext>(AsrProfile::class),
                definition<RecognizerHolder>(CoroutineDispatcher::class),
                definition<HttpClient>(HttpClientEngine::class),
                definition<OpenAiProvider>(Function1::class, ModelTable::class),
                definition<CostMeter>(ModelTable::class),
                definition<Translator>(Function0::class),
                definition<QuickTranslator>(Function0::class),
                definition<IdiomScanner>(List::class),
                definition<com.xrmech.habla.data.TrainingAudioStore>(File::class),
                definition<com.xrmech.habla.data.TrainingExporter>(File::class),
                definition<com.xrmech.habla.app.state.KeyHealthTracker>(kotlinx.coroutines.flow.Flow::class),
                definition<com.xrmech.habla.app.state.ModelDownloadController>(Function0::class),
                definition<com.xrmech.habla.app.ui.history.ExportController>(CoroutineDispatcher::class),
                definition<com.xrmech.habla.app.ui.live.LiveViewModel>(kotlinx.coroutines.flow.StateFlow::class),
                definition<com.xrmech.habla.app.pipeline.PipelineCoordinator>(Function0::class, com.xrmech.habla.app.pipeline.PipelineDispatchers::class),
                // parametersOf(sessionId)
                definition<SessionDetailViewModel>(Long::class),
            ),
        )
    }

    @Test
    fun `should refuse a duplicate definition`() {
        val duplicate = module { single<AssetSource> { AssetSource { "" } } }
        assertThrows(DefinitionOverrideException::class.java) {
            koinApplication {
                allowOverride(false)
                modules(appModules + duplicate)
            }
        }
    }
}
