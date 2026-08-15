package com.visualmemory.glasses.model

import java.time.Instant
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ContractsTest {
    private val now = Instant.parse("2026-08-12T18:00:00Z")

    @Test
    fun pairingPayloadAcceptsOneGatewayUrlAndFutureSingleUseCode() {
        val payload = PairingPayloadCodec.decode(
            """{"gateway_url":"http://192.168.1.42:8080/","pairing_code":"abcdefghijklmnop","expires_at":"2026-08-12T18:02:00Z"}""",
            now,
        )

        assertEquals("http://192.168.1.42:8080", payload.gatewayUrl)
        assertEquals("abcdefghijklmnop", payload.pairingCode)
    }

    @Test(expected = IllegalArgumentException::class)
    fun pairingPayloadRejectsExpiredCode() {
        PairingPayloadCodec.decode(
            """{"gateway_url":"http://192.168.1.42:8080","pairing_code":"abcdefghijklmnop","expires_at":"2026-08-12T17:59:59Z"}""",
            now,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun pairingPayloadRejectsNonHttpGateway() {
        PairingPayloadCodec.decode(
            """{"gateway_url":"file:///tmp/gateway","pairing_code":"abcdefghijklmnop","expires_at":"2026-08-12T18:02:00Z"}""",
            now,
        )
    }

    @Test
    fun tokenRefreshIsScheduledAtSixtyPercentOfRemainingLifetime() {
        val delay = refreshDelayMillis(
            expiresAt = now.plusSeconds(300),
            now = now,
        )

        assertEquals(180_000L, delay)
        assertTrue(delay < 300_000L)
    }
}
