package com.visualmemory.glasses.session

import com.visualmemory.glasses.pairing.StoredPairing
import java.time.Instant

/**
 * Decide whether startup needs its first QR or a replacement-target QR.
 *
 * A valid credential is deliberately not enough to start a session
 * automatically: every new app process opens the target-selection scan window
 * so the wearer can point at either the laptop or GN100 QR. The previous
 * pairing remains stored so a failed new claim is non-destructive and can be
 * selected with the RayNeo temple touchpad (swipe focus, single-tap activate).
 */
internal fun startupPhase(
    pairing: StoredPairing?,
    now: Instant = Instant.now(),
): SessionPhase {
    if (pairing == null) return SessionPhase.UNPAIRED
    val expiresAt = runCatching { Instant.parse(pairing.credential.expiresAt) }.getOrNull()
        ?: return SessionPhase.UNPAIRED
    return if (expiresAt.isAfter(now)) {
        SessionPhase.SELECTING_TARGET
    } else {
        SessionPhase.UNPAIRED
    }
}
