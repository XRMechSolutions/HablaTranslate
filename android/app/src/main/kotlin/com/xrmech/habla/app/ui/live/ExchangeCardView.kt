package com.xrmech.habla.app.ui.live

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import com.xrmech.habla.app.pipeline.CardTranslation
import com.xrmech.habla.app.pipeline.ExchangeCard
import com.xrmech.habla.app.ui.theme.HablaColors
import com.xrmech.habla.translate.model.FlaggedPhrase
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

private val TIME = DateTimeFormatter.ofPattern("HH:mm:ss")

/**
 * One final (design section 6, ExchangeCard): language badge, time, corrected source with flagged
 * spans, translation, confidence, correction block and savable phrase chips. Used read-only by
 * SessionDetail too ([onEdit] set there for exchanges with audio).
 */
@Composable
fun ExchangeCardView(
    card: ExchangeCard,
    isSaved: (FlaggedPhrase) -> Boolean,
    onSave: (FlaggedPhrase) -> Unit,
    onAction: (BannerAction) -> Unit,
    modifier: Modifier = Modifier,
    onEdit: (() -> Unit)? = null,
    /** Replaces the failed line (History rows do not keep the failure kind); no action is offered then. */
    failedText: String? = null,
) {
    val t = card.translation
    val done = t as? CardTranslation.Done
    val correction = done?.result?.takeIf { it.isCorrection }?.correctionDetail
    Card(
        colors = CardDefaults.cardColors(containerColor = if (correction != null) HablaColors.CorrectionBg else HablaColors.Raised),
        border = if (correction != null) BorderStroke(1.dp, HablaColors.CorrectionBorder) else null,
        modifier = modifier.fillMaxWidth().testTag("card_${card.seq}"),
    ) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                LanguageBadge((card.detectedLanguage ?: card.direction.sourceCode).uppercase())
                Text(
                    TIME.format(Instant.ofEpochMilli(card.createdAtMs).atZone(ZoneId.systemDefault())),
                    style = MaterialTheme.typography.labelSmall,
                    color = HablaColors.TextMuted,
                )
                if (done != null) {
                    Text(
                        "${(done.result.confidence * 100).toInt()}%",
                        style = MaterialTheme.typography.labelSmall,
                        color = HablaColors.TextMuted,
                        modifier = Modifier.testTag("card_${card.seq}_confidence"),
                    )
                }
                if (onEdit != null) {
                    Row(Modifier.weight(1f), horizontalArrangement = Arrangement.End) {
                        TextButton(onClick = onEdit, modifier = Modifier.testTag("card_${card.seq}_edit")) { Text("Edit transcript") }
                    }
                }
            }
            val source = done?.result?.corrected?.takeIf { it.isNotBlank() } ?: card.rawTranscript
            Text(highlightSpans(source, done?.flagged.orEmpty()), style = MaterialTheme.typography.bodyLarge)
            when (t) {
                CardTranslation.Pending -> Text(
                    "Translating...",
                    color = HablaColors.TextMuted,
                    modifier = Modifier.testTag("card_${card.seq}_pending"),
                )
                is CardTranslation.Done -> Text(
                    t.result.translated,
                    color = HablaColors.AccentText,
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.testTag("card_${card.seq}_translation"),
                )
                is CardTranslation.Failed -> Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        failedText ?: CardTexts.failedLine(t.kind, t.message),
                        color = HablaColors.ErrorText,
                        modifier = Modifier.weight(1f).testTag("card_${card.seq}_failed"),
                    )
                    (if (failedText == null) CardTexts.failedAction(t.kind) else null)?.let { a ->
                        TextButton(onClick = { onAction(a) }, modifier = Modifier.testTag("card_${card.seq}_action")) { Text(a.label) }
                    }
                }
            }
            if (correction != null) {
                Column(Modifier.testTag("card_${card.seq}_correction")) {
                    Text(correction.wrong, color = HablaColors.ErrorText, textDecoration = TextDecoration.LineThrough)
                    Text(correction.right, color = HablaColors.Success, fontWeight = FontWeight.SemiBold)
                    Text(correction.explanation, color = HablaColors.TextMuted, style = MaterialTheme.typography.bodySmall)
                }
            }
            if (done != null && done.flagged.isNotEmpty()) {
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    done.flagged.forEach { p -> PhraseChip(card.seq, p, isSaved(p), onSave) }
                }
            }
        }
    }
}

@Composable
private fun LanguageBadge(code: String) {
    Surface(color = HablaColors.Raised2, shape = RoundedCornerShape(4.dp)) {
        Text(code, style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp))
    }
}

@Composable
private fun PhraseChip(seq: Int, phrase: FlaggedPhrase, saved: Boolean, onSave: (FlaggedPhrase) -> Unit) {
    var expanded by rememberSaveable(seq, phrase.phrase) { mutableStateOf(false) }
    Column {
        AssistChip(
            onClick = { expanded = !expanded },
            label = { Text(phrase.phrase) },
            colors = AssistChipDefaults.assistChipColors(containerColor = HablaColors.IdiomBg, labelColor = HablaColors.Text),
            border = BorderStroke(1.dp, HablaColors.IdiomBorder),
            modifier = Modifier.testTag("chip_${seq}_${phrase.phrase}"),
        )
        if (expanded) {
            Column(Modifier.padding(start = 4.dp, bottom = 4.dp)) {
                phrase.literal?.takeIf { it.isNotBlank() }?.let { Text("Literally: $it", color = HablaColors.TextMuted) }
                Text(phrase.meaning)
                phrase.region?.takeIf { it.isNotBlank() }?.let { Text("Region: $it", color = HablaColors.TextMuted) }
                if (saved) {
                    Text("Saved", color = HablaColors.Success, modifier = Modifier.testTag("chip_${seq}_${phrase.phrase}_saved"))
                } else {
                    TextButton(onClick = { onSave(phrase) }, modifier = Modifier.testTag("chip_${seq}_${phrase.phrase}_save")) {
                        Text("Save to vocab")
                    }
                }
            }
        }
    }
}

/** Underlines flagged spans; offsets are code points into the transcript and are skipped when out of range. */
internal fun highlightSpans(text: String, flagged: List<FlaggedPhrase>): AnnotatedString = buildAnnotatedString {
    append(text)
    val cpCount = text.codePointCount(0, text.length)
    flagged.forEach { p ->
        val s = p.spanStart ?: return@forEach
        val e = p.spanEnd ?: return@forEach
        if (s < 0 || e <= s || e > cpCount) return@forEach
        addStyle(
            SpanStyle(background = HablaColors.IdiomBg, textDecoration = TextDecoration.Underline),
            text.offsetByCodePoints(0, s),
            text.offsetByCodePoints(0, e),
        )
    }
}
