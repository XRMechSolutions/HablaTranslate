package com.xrmech.habla.app

import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.xrmech.habla.app.di.StartupState
import com.xrmech.habla.app.ui.FatalScreen
import com.xrmech.habla.app.ui.nav.HablaNavHost
import com.xrmech.habla.app.ui.theme.HablaColors
import com.xrmech.habla.app.ui.theme.HablaTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        // Dark-first app: light system-bar icons regardless of the system theme.
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
        )
        super.onCreate(savedInstanceState)
        val startup = (application as HablaApp).startup
        setContent {
            HablaTheme {
                val state by startup.state.collectAsStateWithLifecycle()
                when (val s = state) {
                    // Asset parsing takes milliseconds; nothing that needs SharedAssets is built before Ready.
                    StartupState.Loading -> Box(Modifier.fillMaxSize().background(HablaColors.Page))
                    StartupState.Ready -> HablaNavHost()
                    is StartupState.Fatal -> FatalScreen(s.message)
                }
            }
        }
    }
}
