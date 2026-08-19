package com.visualmemory.glasses.media

import android.content.Context
import android.util.Log
import io.livekit.android.LiveKit
import io.livekit.android.RoomOptions
import io.livekit.android.events.RoomEvent
import io.livekit.android.room.Room
import io.livekit.android.room.participant.VideoTrackPublishDefaults
import io.livekit.android.room.track.CameraPosition
import io.livekit.android.room.track.LocalVideoTrack
import io.livekit.android.room.track.LocalAudioTrackOptions
import io.livekit.android.room.track.LocalVideoTrackOptions
import io.livekit.android.room.track.VideoCaptureParameter
import io.livekit.android.room.track.Track
import io.livekit.android.room.track.VideoEncoding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.launch
import livekit.org.webrtc.RtpParameters
import livekit.org.webrtc.RtpSender

internal const val CAPTURE_WIDTH = 1280
internal const val CAPTURE_HEIGHT = 720
internal const val CAPTURE_FPS = 15

/**
 * Floor for the video encoder, in bits per second.
 *
 * Without one, WebRTC's bandwidth estimator settles at 40-130 kb/s on a link
 * measured at 480 Mb/s with zero packet loss and 8 ms RTT, and the encoder then
 * sends about one frame per second of the fifteen the camera produces. The
 * estimate is self-reinforcing: too little bandwidth yields ~1 fps, second-long
 * gaps between packet bursts make the receiver compute ~100 ms of jitter, and a
 * delay-based controller reads that as congestion and holds the estimate down.
 *
 * 300 kb/s was measured as the smallest floor that escapes it -- 1.5 Mb/s works
 * too but is no better, and the floor is precisely the amount we keep pushing
 * into a link that has genuinely gone bad, so smaller is safer. Adaptation above
 * this value is untouched.
 *
 * See docs/spikes/sender-bitrate/RESULTS.md.
 */
internal const val MIN_VIDEO_BITRATE_BPS = 300_000

/** Logcat tag for publish-health diagnostics. */
private const val MEDIA_TAG = "VMAMedia"

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

    /**
     * Apply [MIN_VIDEO_BITRATE_BPS] to the camera sender once it exists.
     *
     * The sender appears a moment after publishing, so this retries rather than
     * guessing a delay, and gives up loudly instead of leaving the stream
     * silently stuck at one frame per second -- the failure it exists to prevent
     * is itself invisible.
     */
    fun applyMinVideoBitrate(
        scope: CoroutineScope,
        minBps: Int = MIN_VIDEO_BITRATE_BPS,
        attempts: Int = 20,
        intervalMs: Long = 500,
    ) {
        scope.launch {
            repeat(attempts) {
                delay(intervalMs)
                val track = room.localParticipant
                    .getTrackPublication(Track.Source.CAMERA)
                    ?.track as? LocalVideoTrack
                if (track != null && setMinBitrate(track, minBps)) {
                    Log.i(MEDIA_TAG, "video min bitrate set to $minBps bps")
                    return@launch
                }
            }
            Log.w(
                MEDIA_TAG,
                "could not set video min bitrate; expect ~1 fps if the estimator settles low",
            )
        }
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

    /**
     * Set the minimum bitrate on every encoding of a track's sender.
     *
     * The SDK exposes `maxBitrate` but no floor, and its `RtpSender` accessor is
     * internal, so the sender is reached reflectively. `VideoSenderAccessTest`
     * fails if that accessor is renamed, which is the upgrade hazard worth
     * catching at build time rather than in the field.
     */
    private fun setMinBitrate(track: LocalVideoTrack, minBps: Int): Boolean =
        runCatching {
            val sender = senderOf(track) ?: return false
            val parameters = sender.parameters ?: return false
            if (parameters.encodings.isEmpty()) return false
            parameters.encodings.forEach { encoding ->
                encoding.minBitrateBps = minBps
                encoding.maxBitrateBps = maxOf(minBps, encoding.maxBitrateBps ?: minBps)
            }
            sender.parameters = parameters
            true
        }.getOrElse {
            Log.w(MEDIA_TAG, "min bitrate not applied: $it")
            false
        }
}

/**
 * The SDK's internal `RtpSender` accessor, by reflection.
 *
 * Kotlin mangles internal members, so the JVM name carries a module suffix that
 * a rename would change. Kept in one place so there is a single thing to fix,
 * and one test guarding it.
 */
internal fun senderOf(track: LocalVideoTrack): RtpSender? =
    track.javaClass.methods
        .firstOrNull { it.name.startsWith("getSender") && it.parameterCount == 0 }
        ?.invoke(track) as? RtpSender
