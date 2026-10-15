package com.xrmech.habla.app.state

import com.xrmech.habla.data.AppSettings
import com.xrmech.habla.shared.SharedAssets
import com.xrmech.habla.translate.model.Direction
import com.xrmech.habla.translate.model.Mode

/**
 * First-run settings, built from the shared assets so the model ids are never duplicated in code
 * (data interfaces.md: AUTO, CONVERSATION, partials on, partial translation on, models.json
 * defaults, the asr catalog default, training capture off, setup not done).
 */
class AppSettingsDefaults(assets: SharedAssets) {
    val value: AppSettings = AppSettings(
        direction = Direction.AUTO,
        mode = Mode.CONVERSATION,
        partialsEnabled = true,
        partialTranslation = true,
        finalModel = assets.models.finalDefault,
        quickModel = assets.models.quickDefault,
        asrModelId = assets.asrModels.defaultId,
        trainingCapture = false,
        firstRunDone = false,
    )
}
