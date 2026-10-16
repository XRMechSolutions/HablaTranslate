package com.xrmech.habla.app.state

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * Whether the signed-in user has an active Pro entitlement on llm-gateway (D051). AUTO picks the
 * gateway only while [active] is true. The Firebase sign-in + `GET /v1/me` implementation arrives
 * with Habla's Firebase project; until then [NoProAccess] keeps AUTO on the key or the phone.
 */
interface ProAccess {
    val active: StateFlow<Boolean>
}

object NoProAccess : ProAccess {
    override val active: StateFlow<Boolean> = MutableStateFlow(false)
}
