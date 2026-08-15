package com.visualmemory.glasses.model

import java.net.URI
import java.time.Duration
import java.time.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class PairingPayload(
    @SerialName("gateway_url") val gatewayUrl: String,
    @SerialName("pairing_code") val pairingCode: String,
    @SerialName("expires_at") val expiresAt: String,
)

@Serializable
data class PairingClaimRequest(
    @SerialName("pairing_code") val pairingCode: String,
    @SerialName("device_id") val deviceId: String,
)

@Serializable
data class DeviceCredential(
    @SerialName("device_id") val deviceId: String,
    val credential: String,
    @SerialName("expires_at") val expiresAt: String,
)

@Serializable
data class CreateSessionRequest(@SerialName("device_id") val deviceId: String)

@Serializable
data class SessionToken(
    @SerialName("session_id") val sessionId: String,
    @SerialName("device_id") val deviceId: String,
    val room: String,
    @SerialName("livekit_url") val livekitUrl: String,
    val identity: String,
    val token: String,
    @SerialName("expires_at") val expiresAt: String,
)

@Serializable
data class TranscriptEvent(
    val type: String,
    val text: String,
    @SerialName("session_id") val sessionId: String,
    @SerialName("epoch_id") val epochId: String,
    @SerialName("pts_samples_start") val ptsSamplesStart: Long,
    val samples: Long,
    @SerialName("sample_rate") val sampleRate: Int,
    @SerialName("occurred_at") val occurredAt: String,
)

@Serializable
data class ReplyEvent(
    val type: String,
    val question: String,
    val reply: String,
    @SerialName("answer_status") val answerStatus: String? = null,
    @SerialName("object_id") val objectId: String? = null,
    val guard: String,
    @SerialName("latency_ms") val latencyMs: Int,
    @SerialName("occurred_at") val occurredAt: String,
)

sealed interface HudEvent {
    data class Transcript(val value: TranscriptEvent) : HudEvent
    data class Reply(val value: ReplyEvent) : HudEvent
}

object ContractJson {
    val instance = Json { ignoreUnknownKeys = true }
}

object PairingPayloadCodec {
    fun decode(raw: String, now: Instant = Instant.now()): PairingPayload {
        val payload = ContractJson.instance.decodeFromString<PairingPayload>(raw)
        val uri = URI(payload.gatewayUrl)
        require(uri.scheme == "http" || uri.scheme == "https") {
            "gateway_url must use http or https"
        }
        require(!uri.host.isNullOrBlank()) { "gateway_url must include a host" }
        require(payload.pairingCode.length >= 16) { "pairing_code is too short" }
        require(Instant.parse(payload.expiresAt).isAfter(now)) { "pairing code has expired" }
        return payload.copy(gatewayUrl = payload.gatewayUrl.trimEnd('/'))
    }
}

fun refreshDelayMillis(
    expiresAt: Instant,
    now: Instant = Instant.now(),
    fraction: Double = 0.6,
): Long {
    require(fraction in 0.1..0.9)
    val remaining = Duration.between(now, expiresAt).toMillis().coerceAtLeast(1_000)
    return (remaining * fraction).toLong().coerceAtLeast(1_000)
}
