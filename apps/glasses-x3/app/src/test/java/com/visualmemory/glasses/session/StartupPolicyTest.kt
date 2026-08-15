package com.visualmemory.glasses.session

import com.visualmemory.glasses.model.DeviceCredential
import com.visualmemory.glasses.pairing.StoredPairing
import java.time.Instant
import org.junit.Assert.assertEquals
import org.junit.Test

class StartupPolicyTest {
    private val now = Instant.parse("2026-08-13T18:00:00Z")

    @Test
    fun validSavedPairingRequiresTargetQrInsteadOfStarting() {
        val pairing = storedPairing(expiresAt = "2026-08-14T18:00:00Z")

        assertEquals(SessionPhase.SELECTING_TARGET, startupPhase(pairing, now))
    }

    @Test
    fun expiredSavedPairingRequiresANewQr() {
        val pairing = storedPairing(expiresAt = "2026-08-12T18:00:00Z")

        assertEquals(SessionPhase.UNPAIRED, startupPhase(pairing, now))
    }

    @Test
    fun malformedSavedPairingRequiresANewQr() {
        val pairing = storedPairing(expiresAt = "not-an-instant")

        assertEquals(SessionPhase.UNPAIRED, startupPhase(pairing, now))
    }

    @Test
    fun missingSavedPairingRequiresANewQr() {
        assertEquals(SessionPhase.UNPAIRED, startupPhase(null, now))
    }

    private fun storedPairing(expiresAt: String) = StoredPairing(
        gatewayUrl = "http://192.168.50.10:8080",
        credential = DeviceCredential(
            deviceId = "x3-test",
            credential = "device-secret",
            expiresAt = expiresAt,
        ),
    )
}
