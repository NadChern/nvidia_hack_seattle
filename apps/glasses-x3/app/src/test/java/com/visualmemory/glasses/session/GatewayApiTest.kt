package com.visualmemory.glasses.session

import com.visualmemory.glasses.model.DeviceCredential
import com.visualmemory.glasses.model.PairingPayload
import java.time.Instant
import kotlinx.coroutines.test.runTest
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Before
import org.junit.Test

class GatewayApiTest {
    private lateinit var server: MockWebServer
    private lateinit var api: GatewayApi

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()
        api = GatewayApi()
    }

    @After
    fun tearDown() {
        server.shutdown()
    }

    @Test
    fun pairingClaimDoesNotSendAnInternalBearer() = runTest {
        server.enqueue(
            MockResponse().setResponseCode(200).setBody(
                """{"device_id":"x3-01","credential":"device-secret","expires_at":"2026-08-19T18:00:00Z"}""",
            ),
        )
        val payload = PairingPayload(
            gatewayUrl = server.url("/").toString().trimEnd('/'),
            pairingCode = "abcdefghijklmnop",
            expiresAt = Instant.now().plusSeconds(120).toString(),
        )

        val claimed = api.claim(payload, "x3-01")
        val request = server.takeRequest()

        assertEquals("x3-01", claimed.deviceId)
        assertEquals("/v1/pairing/claim", request.path)
        assertFalse(request.headers.names().contains("Authorization"))
    }

    @Test
    fun sessionCreationUsesOnlyTheDeviceCredential() = runTest {
        server.enqueue(
            MockResponse().setResponseCode(201).setBody(
                """{"session_id":"sess_01","device_id":"x3-01","room":"vma-sess_01","livekit_url":"ws://livekit:7880","identity":"x3-01","token":"jwt","expires_at":"2026-08-12T18:05:00Z"}""",
            ),
        )
        val credential = DeviceCredential(
            deviceId = "x3-01",
            credential = "device-secret",
            expiresAt = "2026-08-19T18:00:00Z",
        )

        api.createSession(server.url("/").toString(), credential)
        val request = server.takeRequest()

        assertEquals("Bearer device-secret", request.getHeader("Authorization"))
        assertEquals("/v1/sessions", request.path)
    }
}
