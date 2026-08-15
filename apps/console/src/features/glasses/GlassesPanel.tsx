import { useEffect, useRef, useState } from "react"
import {
  Eye,
  LifeBuoy,
  Mic,
  MicOff,
  QrCode,
  Radio,
  Eraser,
  RefreshCw,
  RotateCw,
  Square,
  Video,
  VideoOff,
  Volume2,
} from "lucide-react"
import { QRCodeSVG } from "qrcode.react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { post } from "@/lib/api"
import type { PairingCode } from "@/lib/contracts"
import { useGlasses } from "@/store/glasses"

function Field({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="tnum truncate font-medium" title={value ?? undefined}>
        {value ?? "—"}
      </span>
    </div>
  )
}

export function GlassesPanel() {
  const audioRef = useRef<HTMLAudioElement>(null)
  const {
    state,
    mode,
    session,
    availableSessions,
    cameraOn,
    micOn,
    videoSid,
    audioSid,
    resolution,
    log,
    publish,
    watch,
    refreshSessions,
    clearStaleSessions,
    rejoin,
    stop,
    toggleCamera,
    toggleMic,
    speak,
    attachAssistantAudio,
    askForHelp,
  } = useGlasses()

  useEffect(() => {
    attachAssistantAudio(audioRef.current)
    return () => attachAssistantAudio(null)
  }, [attachAssistantAudio])

  const connected = state === "publishing" || state === "viewing" || state === "helping"
  const publishing = state === "publishing"
  const [selectedSession, setSelectedSession] = useState("")
  const [pairing, setPairing] = useState<PairingCode | null>(null)
  const [pairingError, setPairingError] = useState<string | null>(null)
  const logRef = useRef<HTMLDivElement>(null)

  const gatewayUrl =
    import.meta.env["VITE_VMA_GATEWAY_PUBLIC_URL"] ??
    `${window.location.protocol}//${window.location.hostname}:8080`
  const pairingPayload = pairing
    ? JSON.stringify({
        gateway_url: gatewayUrl,
        pairing_code: pairing.pairing_code,
        expires_at: pairing.expires_at,
      })
    : null

  const issuePairingCode = async () => {
    setPairing(null)
    setPairingError(null)
    try {
      setPairing(await post<PairingCode>("gateway", "/v1/pairing"))
    } catch (error) {
      setPairingError(error instanceof Error ? error.message : String(error))
    }
  }

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight })
  }, [log])

  useEffect(() => {
    void refreshSessions()
  }, [refreshSessions])

  useEffect(() => {
    if (!availableSessions.some((item) => item.session_id === selectedSession)) {
      setSelectedSession(availableSessions[0]?.session_id ?? "")
    }
  }, [availableSessions, selectedSession])

  return (
    <Card className="flex h-full min-h-0 flex-col overflow-hidden">
      <CardHeader className="flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle className="flex items-center gap-2 text-base">
          <Radio
            className={connected ? "size-4 text-primary" : "size-4 text-muted-foreground"}
          />
          Glasses
        </CardTitle>
        <Badge variant={connected ? "default" : "secondary"}>{state}</Badge>
      </CardHeader>

      <CardContent className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto">
        <div className="space-y-2 rounded-lg border p-3">
          <div className="flex items-center justify-between gap-2">
            <div>
              <p className="text-sm font-medium">Pair real glasses</p>
              <p className="text-xs text-muted-foreground">Open a large, scannable code.</p>
            </div>
            <Dialog>
              <DialogTrigger asChild>
                <Button size="sm" variant="outline" onClick={() => void issuePairingCode()}>
                  <QrCode /> Pair
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-md">
                <DialogHeader>
                  <DialogTitle>Pair RayNeo X3 Pro</DialogTitle>
                  <DialogDescription>
                    Look at this code through the glasses. It is single-use and contains no
                    internal bearer token.
                  </DialogDescription>
                </DialogHeader>
                <div className="grid min-h-96 place-items-center">
                  {pairing && pairingPayload ? (
                    <div className="rounded-xl bg-white p-4">
                      <QRCodeSVG value={pairingPayload} size={360} level="M" />
                    </div>
                  ) : pairingError ? (
                    <p className="text-sm text-destructive">{pairingError}</p>
                  ) : (
                    <p className="text-sm text-muted-foreground">Generating pairing code…</p>
                  )}
                </div>
                {pairing && (
                  <div className="space-y-1 text-center text-xs text-muted-foreground">
                    <p className="truncate" title={gatewayUrl}>{gatewayUrl}</p>
                    <p>Expires {new Date(pairing.expires_at).toLocaleTimeString()}</p>
                  </div>
                )}
              </DialogContent>
            </Dialog>
          </div>
        </div>

        <div className="space-y-2 rounded-lg border p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-medium">Real glasses viewer</span>
            <div className="flex items-center gap-1">
              {/*
                Recovery for `429 capacity_exhausted` on the glasses. A device
                that cannot reach LiveKit mints a fresh session per retry, and
                two of those fill the slot budget. Only publisher-less sessions
                are removed, so a live wearer is never cut off.
              */}
              <Button
                size="sm"
                variant="ghost"
                title="Free session slots held by sessions nobody ever joined"
                onClick={() => void clearStaleSessions()}
              >
                <Eraser /> Clear stale
              </Button>
              <Button size="sm" variant="ghost" onClick={() => void refreshSessions()}>
                <RefreshCw /> Refresh
              </Button>
            </div>
          </div>
          <div className="flex gap-2">
            <select
              aria-label="Live glasses session"
              value={selectedSession}
              onChange={(event) => setSelectedSession(event.target.value)}
              disabled={connected}
              className="min-w-0 flex-1 rounded-md border bg-background px-2 text-sm"
            >
              {availableSessions.length === 0 && <option value="">No live glasses</option>}
              {availableSessions.map((item) => (
                <option key={item.session_id} value={item.session_id}>
                  {item.device_id} — {item.session_id}
                </option>
              ))}
            </select>
            <Button
              size="sm"
              onClick={() => void watch(selectedSession)}
              disabled={connected || !selectedSession}
            >
              <Eye /> Watch
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Viewer grants are subscribe-only. The console requests high quality, and
            production glasses publish one 720p layer because SG-C proved that subscription
            pinning alone does not protect Vision.
          </p>
        </div>

        <p className="text-xs text-muted-foreground">
          Or publish this machine&apos;s camera and microphone as virtual glasses.
        </p>

        <div className="flex flex-wrap gap-2">
          <Button size="sm" onClick={() => void publish()} disabled={connected}>
            Publish
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void askForHelp()}
            disabled={!publishing}
            title="Granny, without glasses: raise a remote-assist request for this session"
          >
            <LifeBuoy /> Ask for help
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void rejoin()}
            disabled={!publishing}
          >
            <RotateCw /> Rejoin
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void toggleCamera()}
            disabled={!publishing}
          >
            {cameraOn ? <Video /> : <VideoOff />} Camera
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void toggleMic()}
            disabled={!publishing}
          >
            {micOn ? <Mic /> : <MicOff />} Mic
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void speak()}
            disabled={!publishing}
          >
            <Volume2 /> Speak
          </Button>
          <Button
            size="sm"
            variant="destructive"
            onClick={() => void stop()}
            disabled={!connected}
          >
            <Square /> Stop
          </Button>
        </div>

        <div className="space-y-1.5 rounded-lg border p-3">
          <Field label="mode" value={mode} />
          <Field label="device" value={session?.device_id ?? null} />
          <Field label="session" value={session?.session_id ?? null} />
          <Field label="room" value={session?.room ?? null} />
          <Field label="camera SID" value={videoSid} />
          <Field label="mic SID" value={audioSid} />
          <Field label="resolution" value={resolution} />
        </div>

        <p className="text-xs text-muted-foreground">
          <strong className="font-medium text-foreground">Rejoin</strong> reproduces a
          dropped virtual-glasses connection: session, room and identity stay the same and
          only the track SIDs change. The gateway treats a changed camera SID as a new media
          epoch.
        </p>

        <ScrollArea className="h-40 rounded-lg border bg-black/30" ref={logRef}>
          <pre className="p-3 font-mono text-[11px] leading-relaxed text-muted-foreground">
            {log.length === 0
              ? "idle"
              : log.map((line) => `${line.at}  ${line.message}`).join("\n")}
          </pre>
        </ScrollArea>

        <audio ref={audioRef} autoPlay className="hidden" />
      </CardContent>
    </Card>
  )
}
