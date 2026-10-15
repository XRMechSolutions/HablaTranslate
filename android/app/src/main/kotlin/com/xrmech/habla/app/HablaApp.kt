package com.xrmech.habla.app

import ai.onnxruntime.OrtEnvironment
import android.app.Application
import com.xrmech.habla.app.di.APP_SCOPE
import com.xrmech.habla.app.di.AppStartup
import com.xrmech.habla.app.di.appModules
import com.xrmech.habla.app.log.AndroidLogSink
import com.xrmech.habla.app.log.HLog
import com.xrmech.habla.app.service.ListeningNotification
import com.xrmech.habla.translate.TranslateLog
import com.xrmech.habla.translate.TranslateLogging
import org.koin.android.ext.koin.androidContext
import org.koin.android.ext.koin.androidLogger
import org.koin.core.context.GlobalContext
import org.koin.core.context.startKoin
import org.koin.core.logger.Level

/** Process start (design section 10): log sinks, ORT telemetry off, Koin, channel, startup work. */
open class HablaApp : Application() {
    lateinit var startup: AppStartup
        private set

    override fun onCreate() {
        super.onCreate()
        HLog.sink = AndroidLogSink
        // Before any OpenAiHttp.create (translateModule) so the HTTP client logs through HLog.
        TranslateLogging.sink = TranslateLog { level, message ->
            when (level) {
                TranslateLog.Level.DEBUG -> HLog.d(HLog.TRANSLATE, message)
                TranslateLog.Level.INFO -> HLog.i(HLog.TRANSLATE, message)
                TranslateLog.Level.WARN -> HLog.w(HLog.TRANSLATE, message)
                TranslateLog.Level.ERROR -> HLog.e(HLog.TRANSLATE, message)
            }
        }
        // ORT telemetry (R2) off before any SileroVad session exists.
        OrtEnvironment.getEnvironment().setTelemetry(false)
        if (GlobalContext.getOrNull() == null) startDi()
        val koin = GlobalContext.get()
        koin.get<ListeningNotification>().ensureChannel()
        startup = AppStartup(
            assets = { koin.get() },
            training = { koin.get() },
            costMeter = { koin.get() },
            usage = { koin.get() },
            cacheDir = cacheDir,
            scope = koin.get(APP_SCOPE),
        )
        startup.run()
    }

    /** Instrumented tests replace this to load their own module set. */
    protected open fun startDi() {
        startKoin {
            androidContext(this@HablaApp)
            allowOverride(false)
            androidLogger(if (BuildConfig.DEBUG) Level.INFO else Level.ERROR)
            modules(appModules)
        }
    }
}
