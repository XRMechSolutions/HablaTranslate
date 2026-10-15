package com.xrmech.habla.app.ui.nav

import androidx.annotation.DrawableRes
import com.xrmech.habla.app.R
import kotlin.reflect.KClass

/**
 * The only list the bottom bar and the rail are built from (ui-standards section 1): adding a
 * screen is one entry here. [marks] lists the routes that select this item (SessionDetail
 * marks History).
 */
enum class TopLevelDestination(
    val route: Any,
    val label: String,
    @DrawableRes val icon: Int,
    val marks: List<KClass<*>>,
) {
    LIVE(Live, "Live", R.drawable.ic_nav_live, listOf(Live::class)),
    HISTORY(History, "History", R.drawable.ic_nav_history, listOf(History::class, SessionDetail::class)),
    VOCAB(Vocab, "Vocab", R.drawable.ic_nav_vocab, listOf(Vocab::class)),
    SETTINGS(Settings(), "Settings", R.drawable.ic_nav_settings, listOf(Settings::class)),
    ;

    companion object {
        /** The item a route belongs to, or null for a route outside the shell. */
        fun forRoute(routeClass: KClass<*>): TopLevelDestination? = entries.firstOrNull { routeClass in it.marks }
    }
}
