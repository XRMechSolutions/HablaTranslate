package com.xrmech.habla.app.state

import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.stateIn

/**
 * `online` = the default network has NET_CAPABILITY_VALIDATED (design section 8). Display only:
 * translation still attempts and the provider classifies the failure.
 */
class ConnectivityMonitor(private val cm: ConnectivityManager, appScope: CoroutineScope) {
    // Collected for the app's lifetime; the callback is unregistered when the app scope ends.
    val online: StateFlow<Boolean> = callbackFlow {
        val callback = object : ConnectivityManager.NetworkCallback() {
            override fun onCapabilitiesChanged(network: Network, caps: NetworkCapabilities) {
                trySend(caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED))
            }

            override fun onLost(network: Network) {
                trySend(false)
            }
        }
        trySend(isValidatedNow())
        cm.registerDefaultNetworkCallback(callback)
        awaitClose { cm.unregisterNetworkCallback(callback) }
    }.distinctUntilChanged().stateIn(appScope, SharingStarted.Eagerly, isValidatedNow())

    /** For the metered-download confirmation dialog. */
    fun isMetered(): Boolean = cm.isActiveNetworkMetered

    private fun isValidatedNow(): Boolean =
        cm.getNetworkCapabilities(cm.activeNetwork)?.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED) == true
}
