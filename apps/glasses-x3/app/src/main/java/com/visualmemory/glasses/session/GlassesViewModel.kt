package com.visualmemory.glasses.session

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.visualmemory.glasses.media.ASSISTANT_TTS_TRACK
import com.visualmemory.glasses.media.LiveKitController
import io.livekit.android.events.RoomEvent
import com.visualmemory.glasses.model.HudEvent
import com.visualmemory.glasses.model.PairingPayloadCodec
import com.visualmemory.glasses.model.SessionToken
import com.visualmemory.glasses.model.refreshDelayMillis
import com.visualmemory.glasses.pairing.PairingStore
import com.visualmemory.glasses.pairing.StoredPairing
import java.time.Instant
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class GlassesViewModel(application: Application) : AndroidViewModel(application) {
    private val store = PairingStore(application)
    private val gateway = GatewayApi()
    private val media = LiveKitController(application)

    private val mutableState = MutableStateFlow(GlassesUiState())
    val state: StateFlow<GlassesUiState> = mutableState.asStateFlow()

    private var pairing: StoredPairing? = null
    private var session: SessionToken? = null
    private var events: DeviceEventConnection? = null
    private var eventJob: Job? = null
    private var refreshJob: Job? = null
    private var manualTriggerJob: Job? = null
    private var mediaWatchJob: Job? = null

    init {
        viewModelScope.launch {
            val loaded = store.load()
            when (startupPhase(loaded)) {
                SessionPhase.SELECTING_TARGET -> {
                    pairing = loaded
                    mutableState.value = GlassesUiState(
                        phase = SessionPhase.SELECTING_TARGET,
                        gatewayUrl = loaded?.gatewayUrl,
                    )
                }
                else -> {
                    if (loaded != null) store.clear()
                    mutableState.value = GlassesUiState(phase = SessionPhase.UNPAIRED)
                }
            }
        }
    }

    fun claimPairing(rawPayload: String) {
        if (mutableState.value.phase == SessionPhase.PAIRING) return
        mutableState.value = mutableState.value.copy(phase = SessionPhase.PAIRING, error = null)
        viewModelScope.launch {
            runCatching {
                val payload = PairingPayloadCodec.decode(rawPayload)
                val credential = gateway.claim(payload, store.deviceId())
                val stored = StoredPairing(payload.gatewayUrl, credential)
                store.save(payload.gatewayUrl, credential)
                pairing = stored
                stored
            }.onSuccess { stored ->
                mutableState.value = GlassesUiState(
                    phase = SessionPhase.READY,
                    gatewayUrl = stored.gatewayUrl,
                )
            }.onFailure(::showPairingError)
        }
    }

    fun continueSavedPairing() {
        val saved = pairing ?: return
        mutableState.value = GlassesUiState(
            phase = SessionPhase.READY,
            gatewayUrl = saved.gatewayUrl,
        )
    }

    fun start() {
        val paired = pairing ?: return
        if (mutableState.value.phase == SessionPhase.CONNECTING ||
            mutableState.value.phase == SessionPhase.LIVE
        ) return
        mutableState.value = mutableState.value.copy(
            phase = SessionPhase.CONNECTING,
            error = null,
        )
        viewModelScope.launch {
            runCatching {
                val existingSessionId = session?.sessionId ?: store.activeSessionId()
                val created = if (existingSessionId == null) {
                    gateway.createSession(paired.gatewayUrl, paired.credential)
                } else {
                    runCatching {
                        gateway.refreshSession(
                            paired.gatewayUrl,
                            paired.credential,
                            existingSessionId,
                        )
                    }.getOrElse {
                        store.saveActiveSessionId(null)
                        gateway.createSession(paired.gatewayUrl, paired.credential)
                    }
                }
                session = created
                store.saveActiveSessionId(created.sessionId)
                openEvents(paired, created)
                // Foreground before capture: Android 12 revokes camera and
                // microphone from a background app, and starting the service
                // after connecting leaves a window where that can happen.
                SessionForegroundService.start(getApplication<Application>())
                // Collect room events *before* connecting. `connect` subscribes
                // to tracks the gateway has already published, so a collector
                // started afterwards misses the `assistant-tts` subscription and
                // the HUD reports no reply audio for a session that has it.
                watchForDrops(paired)
                media.connect(created.livekitUrl, created.token)
                scheduleRefresh(paired, created)
                created
            }.onFailure {
                events?.close()
                events = null
                eventJob?.cancel()
                mediaWatchJob?.cancel()
                session = null
                media.disconnect()
                SessionForegroundService.stop(getApplication<Application>())
            }.onSuccess { created ->
                mutableState.value = mutableState.value.copy(
                    phase = SessionPhase.LIVE,
                    sessionId = created.sessionId,
                        recording = true,
                )
            }.onFailure(::showError)
        }
    }

    fun reconnect() {
        val current = session ?: return
        mutableState.value = mutableState.value.copy(phase = SessionPhase.CONNECTING)
        viewModelScope.launch {
            runCatching {
                media.disconnect()
                media.connect(current.livekitUrl, current.token)
            }.onSuccess {
                mutableState.value = mutableState.value.copy(phase = SessionPhase.LIVE)
            }.onFailure(::showError)
        }
    }

    fun armManualTrigger() {
        val paired = pairing ?: return
        val current = session ?: return
        viewModelScope.launch {
            runCatching {
                gateway.armManualTrigger(
                    paired.gatewayUrl,
                    paired.credential,
                    current.sessionId,
                )
            }.onSuccess {
                mutableState.value = mutableState.value.copy(manualTriggerArmed = true)
                manualTriggerJob?.cancel()
                manualTriggerJob = viewModelScope.launch {
                    delay(15_000)
                    mutableState.value = mutableState.value.copy(manualTriggerArmed = false)
                }
            }.onFailure(::showError)
        }
    }

    fun toggleRecording() {
        val enabled = !mutableState.value.recording
        viewModelScope.launch {
            runCatching {
                media.setCameraEnabled(enabled)
                media.setMicrophoneEnabled(enabled)
            }.onSuccess {
                mutableState.value = mutableState.value.copy(recording = enabled)
            }.onFailure(::showError)
        }
    }

    fun forgetPairing() {
        val paired = pairing
        val activeSessionId = session?.sessionId
        stop()
        viewModelScope.launch {
            if (paired != null && activeSessionId != null) {
                runCatching {
                    gateway.deleteSession(
                        paired.gatewayUrl,
                        paired.credential,
                        activeSessionId,
                    )
                }
            }
            store.clear()
            pairing = null
            session = null
            mutableState.value = GlassesUiState(phase = SessionPhase.UNPAIRED)
        }
    }

    fun stop() {
        refreshJob?.cancel()
        manualTriggerJob?.cancel()
        eventJob?.cancel()
        mediaWatchJob?.cancel()
        events?.close()
        events = null
        media.disconnect()
        SessionForegroundService.stop(getApplication<Application>())
        mutableState.value = mutableState.value.copy(
            phase = if (pairing == null) SessionPhase.UNPAIRED else SessionPhase.READY,
            sessionId = null,
            recording = false,
            manualTriggerArmed = false,
            assistantAudioReady = false,
        )
    }

    private companion object {
        const val REFRESH_RETRY_MILLIS = 5_000L
        const val RECONNECT_BACKOFF_START_MILLIS = 1_000L
        const val RECONNECT_BACKOFF_MAX_MILLIS = 30_000L
    }

    private fun openEvents(paired: StoredPairing, created: SessionToken) {
        events?.close()
        eventJob?.cancel()
        events = gateway.openEvents(
            paired.gatewayUrl,
            paired.credential,
            created.sessionId,
            viewModelScope,
        )
        eventJob = viewModelScope.launch {
            events?.events?.collect { event ->
                when (event) {
                    is HudEvent.Transcript -> mutableState.value = mutableState.value.copy(
                        transcript = event.value.text,
                        manualTriggerArmed = false,
                    )
                    is HudEvent.Reply -> mutableState.value = mutableState.value.copy(
                        reply = event.value.reply,
                        answerStatus = event.value.answerStatus,
                        guard = event.value.guard,
                    )
                }
            }
        }
    }

    /**
     * Keep a usable token in hand for the whole wearing.
     *
     * A live LiveKit connection outlives its token, so this is not about the
     * current stream -- it is about having something valid to reconnect with.
     * A single throw used to kill this loop permanently and silently, which is
     * the failure it exists to prevent, so a failed refresh retries instead.
     */
    private fun scheduleRefresh(paired: StoredPairing, initial: SessionToken) {
        refreshJob?.cancel()
        refreshJob = viewModelScope.launch {
            var current = initial
            while (true) {
                delay(refreshDelayMillis(Instant.parse(current.expiresAt)))
                val refreshed = runCatching {
                    gateway.refreshSession(paired.gatewayUrl, paired.credential, current.sessionId)
                }.getOrNull()
                if (refreshed == null) {
                    // Retry rather than exit. The current token is valid for a
                    // while yet, so a transient network failure is not terminal
                    // -- and a throw here used to end refreshing for good.
                    delay(REFRESH_RETRY_MILLIS)
                    continue
                }
                current = refreshed
                session = refreshed
            }
        }
    }

    /**
     * Reconnect a dropped session with a *freshly minted* token.
     *
     * The SDK reconnects with the token it was given at `connect()`. Once that
     * token has expired -- five minutes by default, against a wearing measured
     * in hours -- its own retry can never succeed, so the reconnect has to be
     * driven from here with a token obtained now.
     */
    private fun watchForDrops(paired: StoredPairing) {
        mediaWatchJob?.cancel()
        mediaWatchJob = viewModelScope.launch {
            media.events.collect { event ->
                if (event is RoomEvent.TrackSubscribed &&
                    event.publication.name == ASSISTANT_TTS_TRACK
                ) {
                    mutableState.value = mutableState.value.copy(assistantAudioReady = true)
                    return@collect
                }
                if (event !is RoomEvent.Disconnected) return@collect
                mutableState.value = mutableState.value.copy(assistantAudioReady = false)
                if (mutableState.value.phase != SessionPhase.LIVE) return@collect
                val current = session ?: return@collect

                mutableState.value = mutableState.value.copy(phase = SessionPhase.CONNECTING)
                var backoff = RECONNECT_BACKOFF_START_MILLIS
                while (mutableState.value.phase == SessionPhase.CONNECTING) {
                    val ok = runCatching {
                        val fresh = gateway.refreshSession(
                            paired.gatewayUrl,
                            paired.credential,
                            current.sessionId,
                        )
                        session = fresh
                        media.connect(fresh.livekitUrl, fresh.token)
                    }.isSuccess
                    if (ok) {
                        mutableState.value = mutableState.value.copy(phase = SessionPhase.LIVE)
                        return@collect
                    }
                    delay(backoff)
                    backoff = (backoff * 2).coerceAtMost(RECONNECT_BACKOFF_MAX_MILLIS)
                }
            }
        }
    }

    private fun showPairingError(error: Throwable) {
        mutableState.value = GlassesUiState(
            phase = if (pairing == null) {
                SessionPhase.UNPAIRED
            } else {
                SessionPhase.SELECTING_TARGET
            },
            gatewayUrl = pairing?.gatewayUrl,
            error = error.message ?: error::class.java.simpleName,
        )
    }

    private fun showError(error: Throwable) {
        mutableState.value = mutableState.value.copy(
            phase = if (pairing == null) SessionPhase.UNPAIRED else SessionPhase.ERROR,
            error = error.message ?: error::class.java.simpleName,
        )
    }

    override fun onCleared() {
        stop()
        media.release()
        super.onCleared()
    }
}

enum class SessionPhase {
    LOADING,
    UNPAIRED,
    SELECTING_TARGET,
    PAIRING,
    READY,
    CONNECTING,
    LIVE,
    ERROR,
}

data class GlassesUiState(
    val phase: SessionPhase = SessionPhase.LOADING,
    val gatewayUrl: String? = null,
    val sessionId: String? = null,
    val recording: Boolean = false,
    val transcript: String? = null,
    val reply: String? = null,
    val answerStatus: String? = null,
    val guard: String? = null,
    val manualTriggerArmed: Boolean = false,
    /**
     * Whether the gateway's `assistant-tts` track is subscribed.
     *
     * LiveKit auto-subscribes and plays remote audio, so the reply is audible
     * without any code -- which also means a silent failure looks exactly like
     * an assistant that had nothing to say. G3 needs an observable signal, so
     * the HUD shows whether the reply channel actually exists.
     */
    val assistantAudioReady: Boolean = false,
    val error: String? = null,
)
