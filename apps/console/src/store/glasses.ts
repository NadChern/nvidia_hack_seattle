import {
  Room,
  RoomEvent,
  Track,
  VideoPresets,
  VideoQuality,
  type RemoteTrack,
  type RemoteTrackPublication,
} from "livekit-client"
import { create } from "zustand"

import { del, get as apiGet, post } from "@/lib/api"
import type {
  AssistAccepted,
  AssistRequest,
  AssistRequestList,
  GatewaySessionList,
  GatewaySessionSummary,
  SessionToken,
} from "@/lib/contracts"

/**
 * Which simulcast layer the admin viewer asks for.
 *
 * HIGH is the demo default: operators need the same detail Vision sees, and
 * production glasses publish a single 720p layer, so there is no layer for the
 * SFU to demote us to. SG-C measured the other case — against a *simulcast*
 * publisher, an explicit high/low split did not hold and the gateway's own
 * frames collapsed to 320x180. If a development publisher ever needs the
 * viewer to stay out of the way, set VITE_VMA_VIEWER_VIDEO_QUALITY=low rather
 * than editing this file.
 */
const VIEWER_QUALITIES: Record<string, VideoQuality> = {
  low: VideoQuality.LOW,
  medium: VideoQuality.MEDIUM,
  high: VideoQuality.HIGH,
}

const VIEWER_QUALITY_LABEL = String(
  import.meta.env["VITE_VMA_VIEWER_VIDEO_QUALITY"] ?? "high",
).toLowerCase()

const VIEWER_QUALITY: VideoQuality = VIEWER_QUALITIES[VIEWER_QUALITY_LABEL] ?? VideoQuality.HIGH

/**
 * Where *this browser* reaches LiveKit, when that differs from where the
 * glasses reach it.
 *
 * The gateway hands every client one address (`client_livekit_url`), which has
 * to be the LAN address so the glasses can use it. Under WSL2 mirrored
 * networking that address is bound inside the VM, and measured here: Windows
 * cannot connect to it (`Test-NetConnection 10.0.0.4:7880` fails) while the
 * glasses on Wi-Fi can. Windows reaches the same services on 127.0.0.1, which
 * is how this console is being served in the first place.
 *
 * Set VITE_VMA_LIVEKIT_URL=ws://127.0.0.1:7880 in that setup. Unset, the
 * session's own URL is used, which is correct whenever browser and device
 * share a route.
 */
const LIVEKIT_URL_OVERRIDE = import.meta.env["VITE_VMA_LIVEKIT_URL"] as string | undefined

const livekitUrlFor = (session: SessionToken) => LIVEKIT_URL_OVERRIDE ?? session.livekit_url

/** See the retry loop in `watch`: one attempt is not enough over `force_tcp`. */
const VIEWER_CONNECT_ATTEMPTS = 3
const VIEWER_RETRY_DELAY_MS = 1_500

export type GlassesState =
  | "idle"
  | "connecting"
  | "publishing"
  | "viewing"
  | "helping"
  | "disconnected"
  | "failed"
export type GlassesMode = "publisher" | "viewer" | "helper" | null

export interface LogLine {
  at: string
  message: string
}

interface GlassesStore {
  state: GlassesState
  mode: GlassesMode
  session: SessionToken | null
  availableSessions: GatewaySessionSummary[]
  room: Room | null
  cameraOn: boolean
  micOn: boolean
  videoSid: string | null
  audioSid: string | null
  resolution: string | null
  log: LogLine[]
  pendingAssist: AssistRequest[]

  publish: () => Promise<void>
  watch: (sessionId: string) => Promise<void>
  refreshSessions: () => Promise<void>
  clearStaleSessions: () => Promise<void>
  rejoin: () => Promise<void>
  stop: () => Promise<void>
  toggleCamera: () => Promise<void>
  toggleMic: () => Promise<void>
  speak: () => Promise<void>
  attachPreview: (element: HTMLVideoElement | null) => void
  attachAssistantAudio: (element: HTMLAudioElement | null) => void
  /** Granny, without glasses: ask the console's own publisher session for help. */
  askForHelp: () => Promise<void>
  /** Refresh the operator's list of currently-ringing assist requests. */
  refreshAssist: () => Promise<void>
  /** Accept a pending request and join that session's room as a helper. */
  acceptAssist: (sessionId: string) => Promise<void>
}

const MAX_LOG = 200

let previewElement: HTMLVideoElement | null = null
let assistantElement: HTMLAudioElement | null = null
let viewerVideoTrack: RemoteTrack | null = null

/** Freeze the exact high-quality POV currently attached to VideoStage. */
export async function capturePreviewJpeg(): Promise<Blob> {
  const video = previewElement
  if (!video || video.videoWidth < 1 || video.videoHeight < 1) {
    throw new Error("The glasses video is not ready yet.")
  }
  const canvas = document.createElement("canvas")
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  const context = canvas.getContext("2d")
  if (!context) throw new Error("This browser cannot capture the video frame.")
  context.drawImage(video, 0, 0, canvas.width, canvas.height)
  return await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Could not encode the video frame."))),
      "image/jpeg",
      0.94,
    )
  })
}

export const useGlasses = create<GlassesStore>((set, get) => {
  const append = (message: string) =>
    set((s) => ({
      log: [...s.log, { at: new Date().toISOString().slice(11, 23), message }].slice(-MAX_LOG),
    }))

  const refreshTracks = () => {
    const room = get().room
    if (!room) return
    const publications = [...room.localParticipant.trackPublications.values()]
    const video = publications.find((p) => p.kind === "video")
    const audio = publications.find((p) => p.kind === "audio")
    const settings = video?.track?.mediaStreamTrack?.getSettings()
    set({
      videoSid: video?.trackSid ?? null,
      audioSid: audio?.trackSid ?? null,
      resolution:
        settings?.width && settings?.height ? `${settings.width}x${settings.height}` : null,
      cameraOn: room.localParticipant.isCameraEnabled,
      micOn: room.localParticipant.isMicrophoneEnabled,
    })
  }

  /** A publisher token is short-lived; keep margin so a rejoin cannot race it. */
  const tokenUsable = () => {
    const session = get().session
    return session !== null && new Date(session.expires_at).getTime() - Date.now() > 15_000
  }

  const releaseSession = async () => {
    const session = get().session
    if (!session) return
    // The gateway allows only a couple of concurrent sessions, and a leaked one
    // holds its slot until the TTL expires an hour later.
    await del("gateway", `/v1/sessions/${session.session_id}`)
    set({ session: null })
  }

  const connect = async ({ reuse = false }: { reuse?: boolean } = {}) => {
    set({ state: "connecting" })

    let session = get().session
    if (reuse && session) {
      // A real reconnect keeps the session and room; only track SIDs change.
      // Refresh in place when needed instead of silently splitting one wearing
      // into a new session after the five-minute LiveKit token expires.
      if (!tokenUsable()) {
        session = await post<SessionToken>(
          "gateway",
          `/v1/sessions/${session.session_id}/token`,
        )
        set({ session })
        append(`refreshed token for ${session.session_id}`)
      } else {
        append(`reusing session ${session.session_id}`)
      }
    } else {
      if (session) await releaseSession()
      session = await post<SessionToken>("gateway", "/v1/sessions", {
        device_id: "browser-glasses",
      })
      set({ session, mode: "publisher" })
      append(`session ${session.session_id}`)
    }

    const room = new Room({
      adaptiveStream: false,
      dynacast: false,
      // One layer only. With simulcast the SFU picks a layer per subscriber and
      // in practice keeps sending the lowest even when the gateway asks for the
      // highest, so a 720p camera reaches detection at 320x180.
      publishDefaults: { simulcast: false },
      videoCaptureDefaults: { resolution: VideoPresets.h720.resolution },
    })

    room.on(RoomEvent.Connected, () => {
      set({ state: "publishing" })
      append(`connected as ${session.identity}`)
    })
    room.on(RoomEvent.Disconnected, (reason) => {
      set({ state: "disconnected" })
      append(`disconnected (${reason ?? "no reason"})`)
    })
    room.on(RoomEvent.LocalTrackPublished, (publication) => {
      append(`published ${publication.kind} track ${publication.trackSid}`)
      refreshTracks()
    })
    room.on(
      RoomEvent.TrackSubscribed,
      (track: RemoteTrack, publication: RemoteTrackPublication) => {
        // The gateway publishes synthesized speech back as an audio track.
        if (track.kind === Track.Kind.Audio && assistantElement) {
          track.attach(assistantElement)
          append(`hearing ${publication.trackName}`)
        }
      },
    )

    set({ room, mode: "publisher" })
    await room.connect(livekitUrlFor(session), session.token)
    await room.localParticipant.setCameraEnabled(true)
    await room.localParticipant.setMicrophoneEnabled(true)

    const camera = room.localParticipant.getTrackPublication(Track.Source.Camera)
    if (camera?.track && previewElement) camera.track.attach(previewElement)
    refreshTracks()
  }

  return {
    state: "idle",
    mode: null,
    session: null,
    availableSessions: [],
    room: null,
    cameraOn: false,
    micOn: false,
    videoSid: null,
    audioSid: null,
    resolution: null,
    log: [],
    pendingAssist: [],

    publish: async () => {
      try {
        await connect()
      } catch (error) {
        set({ state: "failed" })
        append(`error: ${error instanceof Error ? error.message : String(error)}`)
        await get().stop()
      }
    },

    watch: async (sessionId) => {
      if (get().room) await get().stop()
      set({ state: "connecting", mode: "viewer" })
      let viewerRoom: Room | null = null

      /** Wire one room. Called per attempt: see the retry loop below. */
      const buildRoom = (session: SessionToken) => {
        const room = new Room({ adaptiveStream: false, dynacast: false })
        room.on(RoomEvent.Connected, () => {
          set({ state: "viewing" })
          append(`viewing ${session.device_id} as ${session.identity}`)
        })
        room.on(RoomEvent.Disconnected, (reason) => {
          set({ state: "disconnected" })
          append(`viewer disconnected (${reason ?? "no reason"})`)
        })
        room.on(
          RoomEvent.TrackSubscribed,
          (track: RemoteTrack, publication: RemoteTrackPublication) => {
            if (track.kind !== Track.Kind.Video) return
            publication.setVideoQuality(VIEWER_QUALITY)
            viewerVideoTrack = track
            if (previewElement) track.attach(previewElement)
            const settings = track.mediaStreamTrack.getSettings()
            set({
              videoSid: publication.trackSid,
              resolution:
                settings.width && settings.height
                  ? `${settings.width}x${settings.height}`
                  : null,
            })
            append(`viewing ${publication.trackName}; ${VIEWER_QUALITY_LABEL} quality requested`)
          },
        )
        room.on(RoomEvent.TrackUnsubscribed, (track) => {
          if (track !== viewerVideoTrack) return
          track.detach()
          viewerVideoTrack = null
          set({ videoSid: null, resolution: null })
        })
        return room
      }

      try {
        const session = await post<SessionToken>(
          "gateway",
          `/v1/sessions/${sessionId}/viewer`,
        )
        set({ session })

        // A fresh Room per attempt. A LiveKit Room is not reusable once a
        // connect has failed -- it has already torn itself down and emitted
        // Disconnected, so calling connect again on the same object fails
        // instantly. Reusing it made every retry look like a server problem.
        let lastError: unknown = null
        for (let attempt = 1; attempt <= VIEWER_CONNECT_ATTEMPTS; attempt += 1) {
          const room = buildRoom(session)
          viewerRoom = room
          set({ room })
          try {
            await room.connect(livekitUrlFor(session), session.token)
            lastError = null
            break
          } catch (error) {
            lastError = error
            // Carry the reason. Logging only "failed" makes the retry loop look
            // like the problem when it is only reporting one.
            const why = error instanceof Error ? error.message : String(error)
            append(`viewer connect attempt ${attempt} failed: ${why}`)
            await room.disconnect().catch(() => {})
            if (attempt < VIEWER_CONNECT_ATTEMPTS) {
              await new Promise((resolve) => setTimeout(resolve, VIEWER_RETRY_DELAY_MS))
            }
          }
        }
        if (lastError) throw lastError
      } catch (error) {
        if (viewerRoom) await viewerRoom.disconnect().catch(() => {})
        set({ state: "failed", mode: null, session: null, room: null })
        append(`viewer error: ${error instanceof Error ? error.message : String(error)}`)
      }
    },

    askForHelp: async () => {
      const session = get().session
      if (!session) return
      try {
        const result = await post<AssistRequest>(
          "gateway",
          `/v1/assist/${session.session_id}/request`,
        )
        append(`assist request ${result.request_id} · expires ${result.expires_at}`)
      } catch (error) {
        append(`ask for help failed: ${error instanceof Error ? error.message : String(error)}`)
      }
    },

    refreshAssist: async () => {
      try {
        const listing = await apiGet<AssistRequestList>("gateway", "/v1/assist/requests")
        set({ pendingAssist: listing.requests })
      } catch (error) {
        append(`assist list failed: ${error instanceof Error ? error.message : String(error)}`)
      }
    },

    acceptAssist: async (sessionId) => {
      if (get().room) await get().stop()
      set({ state: "connecting", mode: "helper" })

      const room = new Room({ adaptiveStream: false, dynacast: false })
      room.on(RoomEvent.Connected, () => {
        set({ state: "helping" })
        append(`helping ${sessionId}`)
      })
      room.on(RoomEvent.Disconnected, (reason) => {
        set({ state: "disconnected" })
        append(`helper disconnected (${reason ?? "no reason"})`)
      })
      room.on(
        RoomEvent.TrackSubscribed,
        (track: RemoteTrack, publication: RemoteTrackPublication) => {
          if (track.kind !== Track.Kind.Video) return
          viewerVideoTrack = track
          if (previewElement) track.attach(previewElement)
          const settings = track.mediaStreamTrack.getSettings()
          set({
            videoSid: publication.trackSid,
            resolution:
              settings.width && settings.height
                ? `${settings.width}x${settings.height}`
                : null,
          })
          append(`seeing ${publication.trackName}`)
        },
      )
      room.on(RoomEvent.TrackUnsubscribed, (track) => {
        if (track !== viewerVideoTrack) return
        track.detach()
        viewerVideoTrack = null
        set({ videoSid: null, resolution: null })
      })

      try {
        await post<AssistAccepted>("gateway", `/v1/assist/${sessionId}/accept`)
        const session = await post<SessionToken>(
          "gateway",
          `/v1/sessions/${sessionId}/helper`,
        )
        set({ session, room })
        await room.connect(livekitUrlFor(session), session.token)
        await room.localParticipant.setMicrophoneEnabled(true)
        // Not the shared refreshTracks(): that reads *local* publications for
        // its videoSid, and a helper has none -- it would overwrite the
        // remote videoSid the TrackSubscribed handler above just set. Camera
        // stays off deliberately; the helper grant forbids publishing it
        // (can_publish_sources=["microphone"]), and calling
        // setCameraEnabled here would fail silently rather than raise.
        const mic = room.localParticipant.getTrackPublication(Track.Source.Microphone)
        set({ micOn: true, audioSid: mic?.trackSid ?? null })
        await get().refreshAssist()
      } catch (error) {
        await room.disconnect().catch(() => {})
        set({ state: "failed", mode: null, session: null, room: null })
        append(`assist accept failed: ${error instanceof Error ? error.message : String(error)}`)
      }
    },

    refreshSessions: async () => {
      try {
        const listing = await apiGet<GatewaySessionList>("gateway", "/v1/sessions")
        set({ availableSessions: listing.sessions.filter((session) => session.publisher_present) })
      } catch (error) {
        append(`session list failed: ${error instanceof Error ? error.message : String(error)}`)
      }
    },

    clearStaleSessions: async () => {
      // A session slot is held by any session the gateway has minted, whether
      // or not anyone ever joined it. A device that fails to reach LiveKit and
      // retries mints a new one each attempt, so two failed joins exhaust the
      // budget and the glasses get `429 capacity_exhausted` with no way out
      // from the console. Only sessions with no publisher are removed, so this
      // can never cut off a wearer who is actually connected.
      try {
        const listing = await apiGet<GatewaySessionList>("gateway", "/v1/sessions")
        const stale = listing.sessions.filter((session) => !session.publisher_present)
        if (stale.length === 0) {
          append("no stale sessions to clear")
        }
        for (const session of stale) {
          await del("gateway", `/v1/sessions/${session.session_id}`)
          append(`cleared stale session ${session.session_id}`)
        }
        await get().refreshSessions()
      } catch (error) {
        append(`clear failed: ${error instanceof Error ? error.message : String(error)}`)
      }
    },

    rejoin: async () => {
      // Reproduces a dropped connection: session, room and identity all stay,
      // only the track SIDs change. The gateway treats a changed camera track
      // SID as a new media epoch, which is exactly what the vision pipeline
      // must reset tracker state on -- the reason this button exists.
      append("rejoining -- session and identity stay, track SIDs change")
      const room = get().room
      if (room) await room.disconnect()
      set({ room: null })
      try {
        await connect({ reuse: true })
      } catch (error) {
        append(`error: ${error instanceof Error ? error.message : String(error)}`)
        await get().stop()
      }
    },

    stop: async () => {
      const { mode, room } = get()
      if (viewerVideoTrack) {
        viewerVideoTrack.detach()
        viewerVideoTrack = null
      }
      if (room) await room.disconnect()
      if (mode === "publisher") await releaseSession()
      set({
        mode: null,
        session: null,
        room: null,
        state: "idle",
        videoSid: null,
        audioSid: null,
        resolution: null,
        cameraOn: false,
        micOn: false,
      })
    },

    toggleCamera: async () => {
      const room = get().room
      if (!room) return
      const on = room.localParticipant.isCameraEnabled
      await room.localParticipant.setCameraEnabled(!on)
      append(`camera ${on ? "off" : "on"}`)
      refreshTracks()
    },

    toggleMic: async () => {
      const room = get().room
      if (!room) return
      const on = room.localParticipant.isMicrophoneEnabled
      await room.localParticipant.setMicrophoneEnabled(!on)
      append(`microphone ${on ? "muted" : "live"}`)
      refreshTracks()
    },

    speak: async () => {
      const session = get().session
      if (!session) return
      try {
        // Stands in for the Speech Service: the gateway plays a tone on the
        // assistant track, which arrives back here as return audio.
        const body = await post<{ frames: number; hz: number }>(
          "gateway",
          `/v1/return-audio/${session.session_id}/tone?hz=660&seconds=2`,
        )
        append(`assistant spoke: ${body.frames} frames at ${body.hz} Hz`)
      } catch (error) {
        append(`speak failed: ${error instanceof Error ? error.message : String(error)}`)
      }
    },

    attachPreview: (element) => {
      previewElement = element
      if (!element) return
      const mode = get().mode
      if ((mode === "viewer" || mode === "helper") && viewerVideoTrack) {
        viewerVideoTrack.attach(element)
        return
      }
      const room = get().room
      const camera = room?.localParticipant.getTrackPublication(Track.Source.Camera)
      if (camera?.track) camera.track.attach(element)
    },

    attachAssistantAudio: (element) => {
      assistantElement = element
    },
  }
})

/**
 * Hand the session slot back when the tab goes away.
 *
 * `pagehide` fires on close, navigation and mobile backgrounding where
 * `beforeunload` does not, and `keepalive` lets the request outlive the page.
 * It must be `fetch` rather than `sendBeacon`, which can only POST -- a beacon
 * to this path would 405 and silently leak the slot until its TTL expired.
 */
if (typeof window !== "undefined") {
  window.addEventListener("pagehide", () => {
    const { mode, session } = useGlasses.getState()
    if (mode !== "publisher" || !session) return
    void fetch(`/api/gateway/v1/sessions/${session.session_id}`, {
      method: "DELETE",
      keepalive: true,
    })
  })
}
