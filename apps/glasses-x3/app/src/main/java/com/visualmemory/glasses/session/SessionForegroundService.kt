package com.visualmemory.glasses.session

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import com.visualmemory.glasses.R

/**
 * Keeps camera and microphone capture alive while a session is running.
 *
 * Without this the app is an Activity and nothing else, and Android revokes
 * camera and microphone the moment that Activity stops being the foreground
 * app -- a system dialog, a notification shade pull, or the wearer glancing at
 * another app silently ends the session. `FLAG_KEEP_SCREEN_ON` does not help:
 * it keeps the display awake, not the app foregrounded.
 *
 * The service does not own the LiveKit room. It exists so the process holds
 * foreground capture privileges for as long as the wearer is recording, and so
 * that recording is something the wearer can see and stop.
 *
 * Platform notification APIs rather than `NotificationCompat`: `androidx.core`
 * is only a transitive dependency here, and the module pins a `gradle.lockfile`.
 */
class SessionForegroundService : Service() {

    private var captureWakeLock: PowerManager.WakeLock? = null
    private var displayWakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForegroundCompat()

        // The display sleeping must not end a recording the wearer started.
        val power = getSystemService(Context.POWER_SERVICE) as PowerManager
        if (captureWakeLock == null) {
            captureWakeLock = power.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK,
                CAPTURE_WAKE_LOCK_TAG,
            ).apply {
                setReferenceCounted(false)
                acquire(MAX_SESSION_MILLIS)
            }
        }
        if (displayWakeLock == null) {
            // FLAG_KEEP_SCREEN_ON only applies while our window owns focus.
            // RayNeo's system shade/overlays can temporarily own focus while
            // this Activity remains resumed, and the X3 then enters Dozing.
            // A session is an explicit, bounded wearing, so keep its display
            // bright until stop() or the four-hour safety ceiling.
            @Suppress("DEPRECATION")
            displayWakeLock = power.newWakeLock(
                PowerManager.SCREEN_BRIGHT_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP,
                DISPLAY_WAKE_LOCK_TAG,
            ).apply {
                setReferenceCounted(false)
                acquire(MAX_SESSION_MILLIS)
            }
        }

        // Deliberately not START_STICKY: a session the system killed must not
        // silently resume capture without the wearer asking for it again.
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        captureWakeLock?.let { if (it.isHeld) it.release() }
        captureWakeLock = null
        displayWakeLock?.let { if (it.isHeld) it.release() }
        displayWakeLock = null
        super.onDestroy()
    }

    private fun startForegroundCompat() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.session_channel_name),
                // Low, not default: a persistent status, not an alert, and it
                // sits in the wearer's field of view.
                NotificationManager.IMPORTANCE_LOW,
            ),
        )

        val notification: Notification = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.session_notification_title))
            .setContentText(getString(R.string.session_notification_text))
            .setSmallIcon(android.R.drawable.presence_video_online)
            .setOngoing(true)
            .setCategory(Notification.CATEGORY_SERVICE)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA or
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    companion object {
        private const val CHANNEL_ID = "vma-session"
        private const val NOTIFICATION_ID = 1
        private const val CAPTURE_WAKE_LOCK_TAG = "visual-memory:session-capture"
        private const val DISPLAY_WAKE_LOCK_TAG = "visual-memory:session-display"

        /** A ceiling, not an expectation: an unbounded wake lock is a bug. */
        private const val MAX_SESSION_MILLIS = 4L * 60L * 60L * 1000L

        fun start(context: Context) {
            context.startForegroundService(Intent(context, SessionForegroundService::class.java))
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, SessionForegroundService::class.java))
        }
    }
}
