package com.visualmemory.glasses.media

import android.content.Context
import io.livekit.android.LiveKit
import io.livekit.android.RoomOptions
import io.livekit.android.events.RoomEvent
import io.livekit.android.room.Room
import io.livekit.android.room.participant.VideoTrackPublishDefaults
import io.livekit.android.room.track.CameraPosition
import io.livekit.android.room.track.LocalAudioTrackOptions
import io.livekit.android.room.track.LocalVideoTrackOptions
import io.livekit.android.room.track.VideoCaptureParameter
import io.livekit.android.room.track.VideoEncoding
import kotlinx.coroutines.flow.SharedFlow
import livekit.org.webrtc.RtpParameters

internal const val CAPTURE_WIDTH = 1280
internal const val CAPTURE_HEIGHT = 720
internal const val CAPTURE_FPS = 15

/** The gateway publishes synthesized speech on this track; see docs/12. */
const val ASSISTANT_TTS_TRACK = "assistant-tts"

class LiveKitController(context: Context, cameraIdOverride: String? = null) {

    /**
     * Resolved once: LiveKit reads `deviceId` when the capturer is created, and
     * SG-D showed the id cannot be assumed on this hardware.
     */
    private val cameraId = CameraSelection.worldFacingCameraId(
        context = context.applicationContext,
        width = CAPTURE_WIDTH,
        height = CAPTURE_HEIGHT,
        override = cameraIdOverride,
    )

    private val room: Room = LiveKit.create(
        appContext = context.applicationContext,
        options = RoomOptions(
            adaptiveStream = false,
            dynacast = false,
            audioTrackCaptureDefaults = LocalAudioTrackOptions(
                // SG-D found no hardware AEC on this device, so the WebRTC
                // software APM is the only thing keeping the assistant's own
                // reply out of the microphone and back into the transcript.
                echoCancellation = true,
                noiseSuppression = true,
                autoGainControl = true,
            ),
            videoTrackCaptureDefaults = LocalVideoTrackOptions(
                deviceId = cameraId,
                position = CameraPosition.BACK,
                captureParams = VideoCaptureParameter(
                    width = CAPTURE_WIDTH,
                    height = CAPTURE_HEIGHT,
                    maxFps = CAPTURE_FPS,
                ),
            ),
            videoTrackPublishDefaults = VideoTrackPublishDefaults(
                videoEncoding = VideoEncoding(maxBitrate = 1_500_000, maxFps = CAPTURE_FPS),
                // One layer, deliberately. SG-C measured that with simulcast on,
                // an admin viewer joining collapsed the *gateway's* frames to
                // 320x180 and Vision silently lost its resolution.
                simulcast = false,
                degradationPreference = RtpParameters.DegradationPreference.MAINTAIN_RESOLUTION,
            ),
        ),
    )

    /** Room lifecycle, so the session owner can react to a dropped connection. */
    val events: SharedFlow<RoomEvent> = room.events.events

    val selectedCameraId: String get() = cameraId

    suspend fun connect(livekitUrl: String, token: String) {
        room.connect(url = livekitUrl, token = token)
        room.localParticipant.setCameraEnabled(true)
        room.localParticipant.setMicrophoneEnabled(true)
    }

    suspend fun setMicrophoneEnabled(enabled: Boolean) {
        room.localParticipant.setMicrophoneEnabled(enabled)
    }

    suspend fun setCameraEnabled(enabled: Boolean) {
        room.localParticipant.setCameraEnabled(enabled)
    }

    fun disconnect() {
        room.disconnect()
    }

    fun release() {
        room.release()
    }
}
