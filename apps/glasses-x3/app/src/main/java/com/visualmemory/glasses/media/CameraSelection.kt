package com.visualmemory.glasses.media

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.params.StreamConfigurationMap
import android.util.Log
import android.util.Size

private const val TAG = "CameraSelection"

/**
 * Resolve the world-facing camera id at runtime.
 *
 * SG-D found the X3 Pro reports **two** back-facing cameras and does not
 * support `cmd media.camera get-camera-ids`, so a hardcoded "0" was a guess
 * that happened to be untested. Enumerate instead, and prefer a back-facing
 * camera that can actually produce the capture size we ask for -- a device
 * whose second camera is a depth or low-resolution sensor would otherwise be
 * selected by ordering alone.
 *
 * Falls back to the first back-facing camera, then to "0", so a device that
 * refuses enumeration behaves exactly as before rather than failing to start.
 */
object CameraSelection {

    fun worldFacingCameraId(
        context: Context,
        width: Int,
        height: Int,
        override: String? = null,
    ): String {
        if (!override.isNullOrBlank()) return override

        val manager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
            ?: return FALLBACK_CAMERA_ID

        val backFacing = runCatching {
            manager.cameraIdList.filter { id ->
                val characteristics = manager.getCameraCharacteristics(id)
                characteristics.get(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_BACK
            }
        }.getOrElse { error ->
            Log.w(TAG, "camera enumeration failed; using $FALLBACK_CAMERA_ID", error)
            return FALLBACK_CAMERA_ID
        }

        if (backFacing.isEmpty()) {
            Log.w(TAG, "no back-facing camera reported; using $FALLBACK_CAMERA_ID")
            return FALLBACK_CAMERA_ID
        }

        val exact = backFacing.firstOrNull { id ->
            supportsSize(manager, id, width, height)
        }
        val chosen = exact ?: backFacing.first()
        Log.i(
            TAG,
            "world camera $chosen of back-facing $backFacing " +
                "(exact ${width}x$height match: ${exact != null})",
        )
        return chosen
    }

    private fun supportsSize(
        manager: CameraManager,
        id: String,
        width: Int,
        height: Int,
    ): Boolean = runCatching {
        val map = manager.getCameraCharacteristics(id)
            .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?: return false
        map.outputSizesFor(width, height)
    }.getOrDefault(false)

    private fun StreamConfigurationMap.outputSizesFor(width: Int, height: Int): Boolean =
        getOutputSizes(android.graphics.SurfaceTexture::class.java)
            ?.contains(Size(width, height)) == true

    const val FALLBACK_CAMERA_ID: String = "0"
}
