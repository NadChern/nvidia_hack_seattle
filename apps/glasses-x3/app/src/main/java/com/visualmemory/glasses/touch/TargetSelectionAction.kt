package com.visualmemory.glasses.touch

enum class TargetSelectionAction {
    SCAN_QR,
    RECONNECT_SAVED,
}

internal fun movedTargetSelection(
    current: TargetSelectionAction,
): TargetSelectionAction = when (current) {
    TargetSelectionAction.SCAN_QR -> TargetSelectionAction.RECONNECT_SAVED
    TargetSelectionAction.RECONNECT_SAVED -> TargetSelectionAction.SCAN_QR
}

enum class TempleTapAction {
    NONE,
    RECONNECT_SAVED,
    ARM_MANUAL_TRIGGER,
}

internal fun templeTapAction(
    targetSelectionActive: Boolean,
    liveSessionActive: Boolean,
    selectedStartupAction: TargetSelectionAction,
): TempleTapAction = when {
    targetSelectionActive && selectedStartupAction == TargetSelectionAction.RECONNECT_SAVED -> {
        TempleTapAction.RECONNECT_SAVED
    }
    liveSessionActive -> TempleTapAction.ARM_MANUAL_TRIGGER
    else -> TempleTapAction.NONE
}
