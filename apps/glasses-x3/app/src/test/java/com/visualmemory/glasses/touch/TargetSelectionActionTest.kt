package com.visualmemory.glasses.touch

import org.junit.Assert.assertEquals
import org.junit.Test

class TargetSelectionActionTest {
    @Test
    fun swipeMovesFromQrToSavedTarget() {
        assertEquals(
            TargetSelectionAction.RECONNECT_SAVED,
            movedTargetSelection(TargetSelectionAction.SCAN_QR),
        )
    }

    @Test
    fun nextSwipeWrapsBackToQr() {
        assertEquals(
            TargetSelectionAction.SCAN_QR,
            movedTargetSelection(TargetSelectionAction.RECONNECT_SAVED),
        )
    }

    @Test
    fun tapOnSavedTargetReconnects() {
        assertEquals(
            TempleTapAction.RECONNECT_SAVED,
            templeTapAction(
                targetSelectionActive = true,
                liveSessionActive = false,
                selectedStartupAction = TargetSelectionAction.RECONNECT_SAVED,
            ),
        )
    }

    @Test
    fun tapDuringLiveSessionArmsManualVoiceTrigger() {
        assertEquals(
            TempleTapAction.ARM_MANUAL_TRIGGER,
            templeTapAction(
                targetSelectionActive = false,
                liveSessionActive = true,
                selectedStartupAction = TargetSelectionAction.SCAN_QR,
            ),
        )
    }

    @Test
    fun tapOnQrSelectionLeavesScannerRunning() {
        assertEquals(
            TempleTapAction.NONE,
            templeTapAction(
                targetSelectionActive = true,
                liveSessionActive = false,
                selectedStartupAction = TargetSelectionAction.SCAN_QR,
            ),
        )
    }
}
