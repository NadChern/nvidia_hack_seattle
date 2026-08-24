package com.visualmemory.glasses.media

import io.livekit.android.room.track.LocalVideoTrack
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Guards the one reflective reach in the media layer.
 *
 * `LocalVideoTrack`'s `RtpSender` accessor is internal to the SDK, so its JVM
 * name carries a module suffix an upgrade could change. If it does, the minimum
 * bitrate silently stops being applied and the stream drops to about one frame
 * per second -- a failure with no error and no symptom short of measuring frame
 * rate. Better to fail here, at build time.
 */
class VideoSenderAccessTest {

    @Test
    fun `the SDK still exposes a zero-arg RtpSender accessor`() {
        val accessors = LocalVideoTrack::class.java.methods
            .filter { it.name.startsWith("getSender") && it.parameterCount == 0 }

        assertTrue(
            "LocalVideoTrack has no zero-arg getSender* accessor. The LiveKit SDK " +
                "likely renamed it; senderOf() in LiveKitController.kt needs updating, " +
                "or the video stream falls back to ~1 fps. Sender-ish methods found: " +
                LocalVideoTrack::class.java.methods.map { it.name }
                    .filter { it.contains("ender", ignoreCase = true) },
            accessors.isNotEmpty(),
        )
    }

    @Test
    fun `the floor escapes the estimator trap without disabling adaptation`() {
        // 300 kb/s was measured as the smallest floor reaching full frame rate.
        // Much lower and the trap reasserts; much higher and we keep pushing
        // into a link that has genuinely degraded.
        assertTrue(MIN_VIDEO_BITRATE_BPS >= 150_000)
        assertTrue(MIN_VIDEO_BITRATE_BPS <= 600_000)
    }
}
