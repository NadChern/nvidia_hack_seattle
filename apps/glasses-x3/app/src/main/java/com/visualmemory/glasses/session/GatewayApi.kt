package com.visualmemory.glasses.session

import com.visualmemory.glasses.model.ContractJson
import com.visualmemory.glasses.model.CreateSessionRequest
import com.visualmemory.glasses.model.DeviceCredential
import com.visualmemory.glasses.model.HudEvent
import com.visualmemory.glasses.model.PairingClaimRequest
import com.visualmemory.glasses.model.PairingPayload
import com.visualmemory.glasses.model.ReplyEvent
import com.visualmemory.glasses.model.SessionToken
import com.visualmemory.glasses.model.TranscriptEvent
import java.io.Closeable
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener

private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

@kotlinx.serialization.Serializable
private data class UnitResponse(val expires_at: String)

class GatewayApi(private val client: OkHttpClient = OkHttpClient()) {
    suspend fun claim(
        payload: PairingPayload,
        deviceId: String,
    ): DeviceCredential = post(
        url = "${payload.gatewayUrl}/v1/pairing/claim",
        body = ContractJson.instance.encodeToString(
            PairingClaimRequest(payload.pairingCode, deviceId),
        ),
        credential = null,
    )

    suspend fun createSession(
        gatewayUrl: String,
        credential: DeviceCredential,
    ): SessionToken = post(
        url = "${gatewayUrl.trimEnd('/')}/v1/sessions",
        body = ContractJson.instance.encodeToString(CreateSessionRequest(credential.deviceId)),
        credential = credential.credential,
    )

    suspend fun deleteSession(
        gatewayUrl: String,
        credential: DeviceCredential,
        sessionId: String,
    ) = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("${gatewayUrl.trimEnd('/')}/v1/sessions/$sessionId")
            .delete()
            .header("Authorization", "Bearer ${credential.credential}")
            .build()
        client.newCall(request).execute().use { response ->
            check(response.isSuccessful) { "gateway ${response.code}: ${response.body?.string()}" }
        }
    }

    suspend fun armManualTrigger(
        gatewayUrl: String,
        credential: DeviceCredential,
        sessionId: String,
    ) {
        post<UnitResponse>(
            url = "${gatewayUrl.trimEnd('/')}/v1/device/$sessionId/manual-trigger",
            body = "{}",
            credential = credential.credential,
        )
    }

    suspend fun refreshSession(
        gatewayUrl: String,
        credential: DeviceCredential,
        sessionId: String,
    ): SessionToken = post(
        url = "${gatewayUrl.trimEnd('/')}/v1/sessions/$sessionId/token",
        body = "{}",
        credential = credential.credential,
    )

    inline fun <reified T> decode(response: Response): T {
        val body = response.body?.string() ?: error("gateway returned an empty response")
        check(response.isSuccessful) { "gateway ${response.code}: $body" }
        return ContractJson.instance.decodeFromString(body)
    }

    private suspend inline fun <reified T> post(
        url: String,
        body: String,
        credential: String?,
    ): T = withContext(Dispatchers.IO) {
        val builder = Request.Builder()
            .url(url)
            .post(body.toRequestBody(jsonMediaType))
        if (credential != null) builder.header("Authorization", "Bearer $credential")
        client.newCall(builder.build()).execute().use { decode<T>(it) }
    }

    fun openEvents(
        gatewayUrl: String,
        credential: DeviceCredential,
        sessionId: String,
        scope: CoroutineScope,
    ): DeviceEventConnection {
        val base = gatewayUrl.trimEnd('/')
        val socketUrl = when {
            base.startsWith("https://") -> "wss://${base.removePrefix("https://")}"
            base.startsWith("http://") -> "ws://${base.removePrefix("http://")}"
            else -> error("gateway URL must use http or https")
        }
        val request = Request.Builder()
            .url("$socketUrl/v1/device/$sessionId/events")
            .header("Authorization", "Bearer ${credential.credential}")
            .build()
        return DeviceEventConnection(client, request, scope)
    }
}

class DeviceEventConnection(
    private val client: OkHttpClient,
    private val request: Request,
    private val scope: CoroutineScope,
) : Closeable {
    private val mutableEvents = MutableSharedFlow<HudEvent>(
        extraBufferCapacity = 32,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val events: Flow<HudEvent> = mutableEvents

    private val stopped = AtomicBoolean(false)
    private var socket: WebSocket? = null
    private var reconnectJob: Job? = null
    private var reconnectAttempt = 0

    init {
        connect()
    }

    @Synchronized
    private fun connect() {
        if (stopped.get()) return
        socket = client.newWebSocket(
            request,
            object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    reconnectAttempt = 0
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    parseEvent(text)?.let(mutableEvents::tryEmit)
                }

                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    scheduleReconnect()
                }

                override fun onFailure(
                    webSocket: WebSocket,
                    t: Throwable,
                    response: Response?,
                ) {
                    scheduleReconnect()
                }
            },
        )
    }

    @Synchronized
    private fun scheduleReconnect() {
        if (stopped.get() || reconnectJob?.isActive == true) return
        reconnectAttempt += 1
        val delayMs = (1_000L shl (reconnectAttempt - 1).coerceAtMost(3)).coerceAtMost(10_000)
        reconnectJob = scope.launch {
            delay(delayMs)
            connect()
        }
    }

    private fun parseEvent(text: String): HudEvent? = runCatching {
        val type = ContractJson.instance.parseToJsonElement(text)
            .jsonObject["type"]?.jsonPrimitive?.content
        when (type) {
            "transcript" -> HudEvent.Transcript(
                ContractJson.instance.decodeFromString<TranscriptEvent>(text),
            )
            "reply" -> HudEvent.Reply(
                ContractJson.instance.decodeFromString<ReplyEvent>(text),
            )
            else -> null
        }
    }.getOrNull()

    override fun close() {
        if (!stopped.compareAndSet(false, true)) return
        reconnectJob?.cancel()
        socket?.close(1000, "session stopped")
        socket = null
    }
}
