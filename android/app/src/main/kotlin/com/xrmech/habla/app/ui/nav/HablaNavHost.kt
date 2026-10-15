package com.xrmech.habla.app.ui.nav

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationRail
import androidx.compose.material3.NavigationRailItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.navigation.NavDestination
import androidx.navigation.NavDestination.Companion.hasRoute
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.toRoute
import com.xrmech.habla.app.ui.history.HistoryScreen
import com.xrmech.habla.app.ui.history.SessionDetailScreen
import com.xrmech.habla.app.ui.live.LiveScreen
import com.xrmech.habla.app.ui.settings.SettingsScreen
import com.xrmech.habla.app.ui.vocab.VocabScreen

/** Window size class the screens adapt to (design section 6, breakpoints; no adaptive library). */
data class WindowLayout(val widthDp: Dp, val heightDp: Dp) {
    /** < 600 dp: bottom NavigationBar; >= 600 dp: NavigationRail. */
    val useRail: Boolean get() = widthDp >= RAIL_MIN_WIDTH

    /** Landscape phone: Live's bottom control bar moves into the top bar. */
    val shortHeight: Boolean get() = heightDp < SHORT_HEIGHT

    companion object {
        val RAIL_MIN_WIDTH = 600.dp
        val SHORT_HEIGHT = 480.dp
        val CONTENT_MAX_WIDTH = 720.dp
    }
}

val LocalWindowLayout = staticCompositionLocalOf { WindowLayout(360.dp, 800.dp) }

/** Navigation callbacks the screens use; one function so every screen reaches every other. */
fun interface Navigator {
    fun navigate(route: Any)
}

/** App shell: one NavHost with the shared bar/rail on every screen, SessionDetail included. */
@Composable
fun HablaNavHost(navController: NavHostController = rememberNavController()) {
    val backStack by navController.currentBackStackEntryAsState()
    val selected = backStack?.destination?.let(::topLevelOf)
    val navigator = Navigator { route -> navController.navigateTo(route) }

    BoxWithConstraints(Modifier.fillMaxSize()) {
        val layout = WindowLayout(maxWidth, maxHeight)
        CompositionLocalProvider(LocalWindowLayout provides layout) {
            if (layout.useRail) {
                Row(Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing)) {
                    NavigationRail(Modifier.fillMaxHeight().testTag("nav_rail")) {
                        TopLevelDestination.entries.forEach { dest ->
                            NavigationRailItem(
                                selected = dest == selected,
                                onClick = { navigator.navigate(dest.route) },
                                icon = { Icon(painterResource(dest.icon), contentDescription = null) },
                                label = { Text(dest.label) },
                                modifier = Modifier.testTag("nav_${dest.name.lowercase()}"),
                            )
                        }
                    }
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
                        Box(Modifier.widthIn(max = WindowLayout.CONTENT_MAX_WIDTH).fillMaxWidth()) {
                            Destinations(navController, navigator)
                        }
                    }
                }
            } else {
                Scaffold(
                    bottomBar = {
                        NavigationBar(Modifier.testTag("nav_bar")) {
                            TopLevelDestination.entries.forEach { dest ->
                                NavigationBarItem(
                                    selected = dest == selected,
                                    onClick = { navigator.navigate(dest.route) },
                                    icon = { Icon(painterResource(dest.icon), contentDescription = null) },
                                    label = { Text(dest.label) },
                                    modifier = Modifier.testTag("nav_${dest.name.lowercase()}"),
                                )
                            }
                        }
                    },
                ) { padding ->
                    Box(Modifier.fillMaxSize().padding(padding)) {
                        Destinations(navController, navigator)
                    }
                }
            }
        }
    }
}

@Composable
private fun Destinations(navController: NavHostController, navigator: Navigator) {
    NavHost(navController = navController, startDestination = Live) {
        composable<Live> { LiveScreen(navigator) }
        composable<History> { HistoryScreen(navigator) }
        composable<SessionDetail> { entry -> SessionDetailScreen(entry.toRoute<SessionDetail>().sessionId, navigator) }
        composable<Vocab> { VocabScreen(navigator) }
        composable<Settings> { entry -> SettingsScreen(entry.toRoute<Settings>().section, navigator) }
    }
}

private fun topLevelOf(destination: NavDestination): TopLevelDestination? =
    destination.hierarchy.firstNotNullOfOrNull { d ->
        TopLevelDestination.entries.firstOrNull { dest -> dest.marks.any { d.hasRoute(it) } }
    }

/** Top-level items keep their own back stack (saveState/restoreState); other routes push. */
private fun NavHostController.navigateTo(route: Any) {
    val topLevel = TopLevelDestination.entries.any { it.route == route } || route is Settings
    navigate(route) {
        if (topLevel) {
            popUpTo(graph.findStartDestination().id) { saveState = true }
            launchSingleTop = true
            restoreState = route !is Settings || route.section == null
        }
    }
}
