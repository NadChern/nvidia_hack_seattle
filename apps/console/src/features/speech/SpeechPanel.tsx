import { useRef, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Volume2 } from "lucide-react"

import { LiveWaveform } from "@/components/ui/live-waveform"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useTranscriptStream } from "@/hooks/transcript-context"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useGlasses } from "@/store/glasses"
import { get, postForBlob } from "@/lib/api"
import type { SpeechStatus } from "@/lib/contracts"

/**
 * Bench for the speech service.
 *
 * The waveform is `live-waveform` from the ElevenLabs UI registry, which reads
 * the microphone through the native Web Audio API and talks to no cloud -- this
 * system's speech runs on-prem (Parakeet/Kokoro), so anything reaching for
 * `@elevenlabs/client` would be unusable here.
 */
export function SpeechPanel() {
  const audioRef = useRef<HTMLAudioElement>(null)
  const [text, setText] = useState(
    "I last confirmed the keys on the coffee table at 10:42, but they were picked up afterward.",
  )
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [listening, setListening] = useState(false)

  const status = useQuery({
    queryKey: ["speech", "status"],
    queryFn: ({ signal }) => get<SpeechStatus>("speech", "/v1/status", signal),
    refetchInterval: 10_000,
    retry: false,
  })
  const tts = status.data?.backends.tts
  const stt = status.data?.backends.stt

  // Agent owns Speech's STT connection and forwards completed transcripts
  // through the Gateway event channel shared by this console and the HUD.
  const sessionId = useGlasses((glasses) => glasses.session?.session_id ?? null)
  const heard = useTranscriptStream()

  const synthesize = async () => {
    setBusy(true)
    setError(null)
    setInfo(null)
    try {
      const blob = await postForBlob("speech", "/v1/synthesize", { text })
      const url = URL.createObjectURL(blob)
      if (audioRef.current) {
        audioRef.current.src = url
        await audioRef.current.play().catch(() => undefined)
      }
      setInfo(`${(blob.size / 1024).toFixed(1)} KB of ${blob.type || "audio"}`)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle className="text-base">Speech</CardTitle>
        {status.isError ? (
          <Badge variant="destructive">unreachable</Badge>
        ) : tts ? (
          <Badge variant={tts.real ? "default" : "secondary"}>
            {tts.real ? tts.name : "stub — no model"}
          </Badge>
        ) : null}
      </CardHeader>

      <CardContent className="flex flex-1 flex-col gap-3">
        {tts && !tts.real && (
          // The honest version of what was happening before: the endpoint
          // answers 200 with a valid WAV of pure silence, so pressing "Speak
          // it" and hearing nothing looked exactly like a broken feature.
          <p className="rounded-md border border-amber-500/30 bg-amber-500/10 p-2 text-xs text-amber-200">
            TTS is disabled on this runtime profile, so synthesis returns 0.1s of
            silence. {stt?.real ? `${stt.name} STT remains real.` : "STT is also stubbed."}
          </p>
        )}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
              What you said
            </div>
            <Badge variant={heard.connection === "listening" ? "default" : "secondary"}>
              {sessionId ? heard.connection : "no session"}
            </Badge>
          </div>
          <ScrollArea className="h-32 rounded-lg border">
            {!sessionId ? (
              <p className="p-3 text-xs text-muted-foreground">
                Publish from the Glasses tab first — transcription runs against that
                session's audio on the gateway relay, not this browser's microphone.
              </p>
            ) : heard.transcripts.length === 0 ? (
              <p className="p-3 text-xs text-muted-foreground">
                A line appears when you stop speaking — a transcript covers one
                contiguous stretch of speech, so there is no interim result to show
                mid-sentence.
              </p>
            ) : (
              <div className="divide-y">
                {heard.transcripts.map((t) => (
                  <div key={`${t.epoch_id}-${t.pts_samples_start}`} className="px-3 py-2">
                    <p className="text-sm leading-snug">{t.text || <em>(silence)</em>}</p>
                    <p className="tnum mt-0.5 text-[10px] text-muted-foreground">
                      {new Date(t.receivedAt).toLocaleTimeString(undefined, { hour12: false })} ·{" "}
                      {(t.samples / t.sample_rate).toFixed(1)}s
                    </p>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        </div>

        <div className="space-y-1.5">
          <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
            Microphone
          </div>
          <div className="flex items-center gap-3 rounded-lg border p-2">
            <Button
              size="sm"
              variant={listening ? "default" : "outline"}
              onClick={() => setListening((on) => !on)}
            >
              {listening ? "Stop" : "Listen"}
            </Button>
            <div className="h-10 flex-1">
              {listening ? (
                <LiveWaveform active barWidth={2} barGap={2} className="h-10 w-full" />
              ) : (
                <div className="grid h-10 place-items-center text-xs text-muted-foreground">
                  idle
                </div>
              )}
            </div>
          </div>
          <p className="text-[10px] text-muted-foreground">
            Local input level only, drawn in this browser — transcription happens in
            the speech service against the gateway's audio relay
            {stt ? ` (${stt.real ? stt.name : "stub — no model"})` : ""}, so this meter
            moving does not mean anything is being heard.
          </p>
        </div>

        <div className="space-y-1.5">
          <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
            Synthesize
          </div>
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={3}
            className="w-full resize-none rounded-md border bg-transparent px-3 py-2 text-sm outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
          />
          <Button size="sm" onClick={() => void synthesize()} disabled={busy}>
            <Volume2 /> Speak it
          </Button>
        </div>

        {error && (
          <p className="text-xs text-destructive">
            {error.includes("404") || error.includes("Failed to fetch")
              ? "No speech service on /api/speech."
              : error}
          </p>
        )}

        <div className="mt-auto space-y-1">
          <audio ref={audioRef} controls className="w-full" />
          {info && <p className="text-[10px] text-muted-foreground">{info}</p>}
        </div>
      </CardContent>
    </Card>
  )
}
