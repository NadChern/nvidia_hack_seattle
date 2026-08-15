package com.visualmemory.glasses

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.ffalcon.mercury.android.sdk.touch.CommonTouchCallback
import com.ffalcon.mercury.android.sdk.touch.FlingArgs
import com.ffalcon.mercury.android.sdk.touch.TouchDispatcher
import com.visualmemory.glasses.pairing.PairingScanner
import com.visualmemory.glasses.session.GlassesUiState
import com.visualmemory.glasses.session.GlassesViewModel
import com.visualmemory.glasses.session.SessionPhase
import com.visualmemory.glasses.touch.TargetSelectionAction
import com.visualmemory.glasses.touch.TempleTapAction
import com.visualmemory.glasses.touch.movedTargetSelection
import com.visualmemory.glasses.touch.templeTapAction
import kotlinx.coroutines.delay

class MainActivity : ComponentActivity() {
    private var permissionsGranted by mutableStateOf(false)
    private var pendingPairingPayload by mutableStateOf<String?>(null)
    private var selectedStartupAction by mutableStateOf(TargetSelectionAction.SCAN_QR)
    private var reconnectSavedRequest by mutableStateOf(0L)
    private var manualTriggerRequest by mutableStateOf(0L)
    private var targetSelectionActive = false
    private var liveSessionActive = false

    private val templeTouchDispatcher = TouchDispatcher(TouchDispatcher.Source.Activity)
    private val templeTouchCallback = object : CommonTouchCallback() {
        override fun onTPClick(): Boolean {
            Log.i(TOUCH_LOG_TAG, "single tap; selected=$selectedStartupAction")
            when (
                templeTapAction(
                    targetSelectionActive = targetSelectionActive,
                    liveSessionActive = liveSessionActive,
                    selectedStartupAction = selectedStartupAction,
                )
            ) {
                TempleTapAction.RECONNECT_SAVED -> reconnectSavedRequest += 1
                TempleTapAction.ARM_MANUAL_TRIGGER -> {
                    manualTriggerRequest += 1
                    Log.i(TOUCH_LOG_TAG, "arming manual voice trigger")
                }
                TempleTapAction.NONE -> Unit
            }
            return true
        }

        override fun onTPSlideForward(args: FlingArgs): Boolean {
            moveStartupFocus("forward")
            return true
        }

        override fun onTPSlideBackward(args: FlingArgs): Boolean {
            moveStartupFocus("backward")
            return true
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { permissions ->
        permissionsGranted = permissions.values.all { it }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        pendingPairingPayload = intent.getStringExtra(EXTRA_PAIRING_PAYLOAD)
        permissionsGranted = requiredPermissions().all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
        if (!permissionsGranted) permissionLauncher.launch(requiredPermissions())

        setContent {
            MaterialTheme(colorScheme = MaterialTheme.colorScheme.copy(background = Color.Black)) {
                Surface(modifier = Modifier.fillMaxSize(), color = Color.Black) {
                    if (permissionsGranted) {
                        GlassesApp(
                            pairingPayload = pendingPairingPayload,
                            selectedStartupAction = selectedStartupAction,
                            reconnectSavedRequest = reconnectSavedRequest,
                            manualTriggerRequest = manualTriggerRequest,
                            onPairingPayloadHandled = { pendingPairingPayload = null },
                            onPhaseChange = { phase ->
                                targetSelectionActive = phase == SessionPhase.SELECTING_TARGET
                                liveSessionActive = phase == SessionPhase.LIVE
                            },
                        )
                    } else {
                        PermissionRequired()
                    }
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        pendingPairingPayload = intent.getStringExtra(EXTRA_PAIRING_PAYLOAD)
    }

    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        val deviceName = event.device?.name
        if (deviceName == LEFT_TEMPLE_DEVICE || deviceName == RIGHT_TEMPLE_DEVICE) {
            // Compose consumes touch events in its root AndroidComposeView, so
            // BaseTouchActivity.onTouchEvent never sees them. Dispatch at the
            // Activity boundary, exactly where RayNeo's documentation places
            // TouchDispatcher, before offering non-temple input to Compose.
            templeTouchDispatcher.onMotionEvent(event, templeTouchCallback)
            return true
        }
        return super.dispatchTouchEvent(event)
    }

    private fun moveStartupFocus(direction: String) {
        if (!targetSelectionActive) return
        selectedStartupAction = movedTargetSelection(selectedStartupAction)
        Log.i(TOUCH_LOG_TAG, "$direction swipe; selected=$selectedStartupAction")
    }

    private fun requiredPermissions() = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO,
    )

    private companion object {
        const val EXTRA_PAIRING_PAYLOAD = "pairing_payload"
        const val LEFT_TEMPLE_DEVICE = "cyttsp6_mt"
        const val RIGHT_TEMPLE_DEVICE = "cyttsp5_mt"
        const val TOUCH_LOG_TAG = "VmaTempleTouch"
    }
}

@Composable
private fun GlassesApp(
    pairingPayload: String?,
    selectedStartupAction: TargetSelectionAction,
    reconnectSavedRequest: Long,
    manualTriggerRequest: Long,
    onPairingPayloadHandled: () -> Unit,
    onPhaseChange: (SessionPhase) -> Unit,
    model: GlassesViewModel = viewModel(),
) {
    val state by model.state.collectAsStateWithLifecycle()
    LaunchedEffect(pairingPayload) {
        if (pairingPayload != null) {
            model.claimPairing(pairingPayload)
            onPairingPayloadHandled()
        }
    }
    LaunchedEffect(reconnectSavedRequest) {
        if (reconnectSavedRequest > 0) model.continueSavedPairing()
    }
    LaunchedEffect(manualTriggerRequest) {
        if (manualTriggerRequest > 0) model.armManualTrigger()
    }
    LaunchedEffect(state.phase) {
        onPhaseChange(state.phase)
        when (state.phase) {
            SessionPhase.READY -> model.start()
            SessionPhase.ERROR -> {
                delay(2_000)
                model.start()
            }
            else -> Unit
        }
    }
    DisposableEffect(Unit) {
        onDispose { onPhaseChange(SessionPhase.LOADING) }
    }
    StereoLayout {
        when (state.phase) {
            SessionPhase.UNPAIRED,
            SessionPhase.PAIRING,
            -> PairingPanel(state)
            SessionPhase.SELECTING_TARGET -> TargetSelectionPanel(
                state = state,
                selectedAction = selectedStartupAction,
            )
            else -> HudPanel(state = state)
        }
    }
    if (state.phase == SessionPhase.UNPAIRED ||
        (state.phase == SessionPhase.SELECTING_TARGET &&
            selectedStartupAction == TargetSelectionAction.SCAN_QR)
    ) {
        PairingScanner(onPayload = model::claimPairing)
    }
}

@Composable
private fun StereoLayout(content: @Composable () -> Unit) {
    Row(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        Box(modifier = Modifier.weight(1f).fillMaxHeight()) { content() }
        Box(modifier = Modifier.weight(1f).fillMaxHeight()) { content() }
    }
}

@Composable
private fun PairingPanel(state: GlassesUiState) {
    Box(
        modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(28.dp),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                if (state.phase == SessionPhase.PAIRING) {
                    "Claiming pairing code…"
                } else {
                    "Look at the console QR"
                },
                color = Color.White,
                fontSize = 22.sp,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                "Camera scanning is active",
                color = Color(0xFFB8C4CE),
                fontSize = 14.sp,
            )
            state.error?.let { Text(it, color = Color(0xFFFF8A80), fontSize = 14.sp) }
        }
    }
}

@Composable
private fun TargetSelectionPanel(
    state: GlassesUiState,
    selectedAction: TargetSelectionAction,
) {
    Box(
        modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(28.dp),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                "Scan laptop or GN100 QR",
                color = Color.White,
                fontSize = 22.sp,
                fontWeight = FontWeight.SemiBold,
            )
            FocusedAction(
                text = "Scan QR",
                selected = selectedAction == TargetSelectionAction.SCAN_QR,
            )
            Button(
                onClick = {},
                modifier = Modifier.border(
                    width = 2.dp,
                    color = if (selectedAction == TargetSelectionAction.RECONNECT_SAVED) {
                        Color(0xFF62E6A7)
                    } else {
                        Color.Transparent
                    },
                ),
            ) {
                Text("Reconnect saved target")
            }
            Text(
                "Swipe to select · tap to activate",
                color = Color(0xFFB8C4CE),
                fontSize = 13.sp,
                modifier = Modifier.padding(top = 8.dp),
            )
            state.error?.let { Text(it, color = Color(0xFFFF8A80), fontSize = 14.sp) }
        }
    }
}

@Composable
private fun FocusedAction(text: String, selected: Boolean) {
    Text(
        text,
        color = if (selected) Color(0xFF62E6A7) else Color.White,
        fontSize = 16.sp,
        fontWeight = FontWeight.Bold,
        modifier = Modifier
            .padding(6.dp)
            .border(
                width = 2.dp,
                color = if (selected) Color(0xFF62E6A7) else Color.Transparent,
            )
            .padding(horizontal = 14.dp, vertical = 6.dp),
    )
}

@Composable
private fun HudPanel(state: GlassesUiState) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .safeDrawingPadding()
            .padding(horizontal = 28.dp, vertical = 18.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = state.phase.name.lowercase().replaceFirstChar(Char::uppercase),
                color = if (state.phase == SessionPhase.LIVE) Color(0xFF62E6A7) else Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                // Reply audio plays through LiveKit's own routing, so a broken
                // return path is indistinguishable from an assistant with
                // nothing to say. Show whether the track is actually there.
                if (state.phase == SessionPhase.LIVE) {
                    Text(
                        text = if (state.assistantAudioReady) "🔊" else "🔇",
                        fontSize = 14.sp,
                        modifier = Modifier.padding(end = 10.dp),
                    )
                }
                Text(
                    text = if (state.recording) "● REC" else "○ PAUSED",
                    color = if (state.recording) Color(0xFFFF6B6B) else Color(0xFFFFD166),
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                )
            }
        }

        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                if (state.manualTriggerArmed) {
                    "Ask now: where did I leave…"
                } else {
                    state.transcript ?: "Say “Hey memory…” or tap, then ask where…"
                },
                color = if (state.manualTriggerArmed) Color(0xFF62E6A7) else Color(0xFFB8C4CE),
                fontSize = 17.sp,
                fontWeight = if (state.manualTriggerArmed) {
                    FontWeight.Bold
                } else {
                    FontWeight.Normal
                },
            )
            Text(
                state.reply ?: when (state.phase) {
                    SessionPhase.READY -> "Starting…"
                    SessionPhase.CONNECTING -> "Connecting camera and microphone…"
                    else -> "Listening"
                },
                color = Color.White,
                fontSize = 25.sp,
                lineHeight = 30.sp,
                fontWeight = FontWeight.Medium,
            )
            if (state.answerStatus != null || state.guard != null) {
                Text(
                    listOfNotNull(state.answerStatus, state.guard).joinToString(" · "),
                    color = if (state.guard == "passed") Color(0xFF62E6A7) else Color(0xFFFFD166),
                    fontSize = 14.sp,
                )
            }
            state.error?.let { Text(it, color = Color(0xFFFF8A80), fontSize = 14.sp) }
        }
    }
}

@Composable
private fun PermissionRequired() {
    StereoLayout {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(
                "Camera and microphone permissions are required",
                color = Color.White,
                fontSize = 20.sp,
            )
        }
    }
}
