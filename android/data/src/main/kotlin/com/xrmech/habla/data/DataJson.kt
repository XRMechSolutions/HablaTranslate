package com.xrmech.habla.data

import kotlinx.serialization.json.Json

/** JSON used for the `flagged_json` and `correction_json` columns. */
internal val DataJson: Json = Json {
    ignoreUnknownKeys = true
    encodeDefaults = true
    explicitNulls = true
}
