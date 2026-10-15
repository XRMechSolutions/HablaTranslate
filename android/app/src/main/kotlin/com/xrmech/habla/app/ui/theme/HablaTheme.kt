package com.xrmech.habla.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/**
 * Dark-first palette lifted from the web client (`habla/client/styles.css`), measured per
 * ui-standards section 3. Contrast and luminance of every readable foreground against its real
 * surface are recorded in `.dev/planning/ui-standards.md`. Dynamic colour is off.
 */
object HablaColors {
    val Page = Color(0xFF0F1419)
    val Surface = Color(0xFF1A2027)
    val Raised = Color(0xFF1E272F)
    val Raised2 = Color(0xFF253340)

    /** Primary text and control labels (lum 0.812). */
    val Text = Color(0xFFE7E9EA)

    /** Secondary text, lifted from the web's #8b98a5 so it clears 4.5:1 on Raised2 too. */
    val TextMuted = Color(0xFFA3B0BD)

    /** Fills and indicators only, never body text; dark text sits on it. */
    val Accent = Color(0xFF1D9BF0)
    val OnAccent = Page

    /** Accent-coloured readable text (links, selected labels). */
    val AccentText = Color(0xFF6CC4FF)
    val Success = Color(0xFF3FD69F)
    val Warning = Color(0xFFF59E0B)

    /** Error text; the web's #f4212e is too dark to read on a dimmed screen (lum 0.205). */
    val ErrorText = Color(0xFFFF7A7A)
    val ErrorFill = Color(0xFFF4212E)

    /** Borders: >= 3:1 against every surface. */
    val Outline = Color(0xFF6E8091)

    val IdiomBg = Color(0xFF1C2B1F)
    val IdiomBorder = Color(0xFF2D6A3E)
    val CorrectionBg = Color(0xFF2B1C1C)
    val CorrectionBorder = Color(0xFF6A2D2D)
}

private val Scheme = darkColorScheme(
    primary = HablaColors.Accent,
    onPrimary = HablaColors.OnAccent,
    primaryContainer = HablaColors.Raised2,
    onPrimaryContainer = HablaColors.Text,
    secondary = HablaColors.AccentText,
    onSecondary = HablaColors.Page,
    secondaryContainer = HablaColors.Raised2,
    onSecondaryContainer = HablaColors.Text,
    tertiary = HablaColors.Success,
    onTertiary = HablaColors.Page,
    background = HablaColors.Page,
    onBackground = HablaColors.Text,
    surface = HablaColors.Surface,
    onSurface = HablaColors.Text,
    surfaceVariant = HablaColors.Raised,
    onSurfaceVariant = HablaColors.TextMuted,
    surfaceContainerLowest = HablaColors.Page,
    surfaceContainerLow = HablaColors.Surface,
    surfaceContainer = HablaColors.Surface,
    surfaceContainerHigh = HablaColors.Raised,
    surfaceContainerHighest = HablaColors.Raised2,
    error = HablaColors.ErrorText,
    onError = HablaColors.Page,
    errorContainer = HablaColors.CorrectionBg,
    onErrorContainer = HablaColors.Text,
    outline = HablaColors.Outline,
    outlineVariant = HablaColors.Raised2,
)

@Composable
fun HablaTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Scheme, content = content)
}
