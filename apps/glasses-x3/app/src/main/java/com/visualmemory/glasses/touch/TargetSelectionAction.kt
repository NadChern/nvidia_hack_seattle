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

/**
 * The two focusable HUD buttons during a live session. Speak is the default
 * focus, so a tap with no swipe behaves exactly as the old push-to-talk did;
 * one swipe reaches Register, the grounder-free, speech-free enrollment path.
 */
enum class LiveAction {
    SPEAK,
    REGISTER,
}

internal fun movedLiveSelection(
    current: LiveAction,
): LiveAction = when (current) {
    LiveAction.SPEAK -> LiveAction.REGISTER
    LiveAction.REGISTER -> LiveAction.SPEAK
}

enum class TempleTapAction {
    NONE,
    RECONNECT_SAVED,
    ARM_MANUAL_TRIGGER,
    ARM_REGISTER,
}

internal fun templeTapAction(
    targetSelectionActive: Boolean,
    liveSessionActive: Boolean,
    selectedStartupAction: TargetSelectionAction,
    selectedLiveAction: LiveAction,
): TempleTapAction = when {
    targetSelectionActive && selectedStartupAction == TargetSelectionAction.RECONNECT_SAVED -> {
        TempleTapAction.RECONNECT_SAVED
    }
    liveSessionActive && selectedLiveAction == LiveAction.REGISTER -> TempleTapAction.ARM_REGISTER
    liveSessionActive -> TempleTapAction.ARM_MANUAL_TRIGGER
    else -> TempleTapAction.NONE
}
