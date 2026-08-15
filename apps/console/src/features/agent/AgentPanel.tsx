import { useCallback, useEffect, useRef, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Mic, ShieldCheck, Volume2 } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useTranscriptStream } from "@/hooks/transcript-context"
import type { HeardTranscript } from "@/hooks/useTranscripts"
import { get, post, postForBlob } from "@/lib/api"
import type {
  AgentAnswer,
  AgentAnswerStatus,
  AgentStatus,
} from "@/lib/contracts"
import { useGlasses } from "@/store/glasses"

type TurnPhase = "idle" | "armed" | "waiting" | "asking" | "speaking"
type BadgeVariant = "default" | "secondary" | "destructive" | "outline"

const STATUS_VARIANT: Record<AgentAnswerStatus, BadgeVariant> = {
  confirmed: "default",
  last_confirmed_only: "outline",
  unknown: "secondary",
  ambiguous_object: "destructive",
}
const TRANSCRIPT_WAIT_TIMEOUT_MS = 12_000

function transcriptKey(transcript: HeardTranscript | undefined): string | null {
  return transcript
    ? `${transcript.epoch_id}:${transcript.pts_samples_start}:${transcript.receivedAt}`
    : null
}

function phaseLabel(phase: TurnPhase): string {
  switch (phase) {
    case "armed":
      return "Listening — release when done"
    case "waiting":
      return "Waiting for transcript…"
    case "asking":
      return "Checking memory…"
    case "speaking":
      return "Speaking…"
    default:
      return "Hold to ask"
  }
}

/**
 * Push-to-talk conversational loop.
 *
 * Audio already travels browser → LiveKit → gateway → Speech. Holding the
 * button only arms the next completed transcript; it does not open a second
 * microphone or create a MediaRecorder path. A segment arrives after speech
 * ends, so release leaves the turn armed until that transcript appears or a
 * bounded timeout disarms it. A late unrelated transcript can never reuse an
 * expired turn.
 */
export function AgentPanel() {
  const audioRef = useRef<HTMLAudioElement>(null)
  const audioUrlRef = useRef<string | null>(null)
  const armedRef = useRef(false)
  const releasedRef = useRef(false)
  const armAfterTranscriptRef = useRef<string | null>(null)
  const pendingTranscriptsRef = useRef<HeardTranscript[]>([])
  const seenTranscriptKeysRef = useRef<Set<string>>(new Set())
  const transcriptTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const sessionId = useGlasses((glasses) => glasses.session?.session_id ?? null)
  const heard = useTranscriptStream()
  const [phase, setPhase] = useState<TurnPhase>("idle")
  const [transcript, setTranscript] = useState<string | null>(null)
  const [answer, setAnswer] = useState<AgentAnswer | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [playbackNote, setPlaybackNote] = useState<string | null>(null)

  const status = useQuery({
    queryKey: ["agent", "status"],
    queryFn: ({ signal }) => get<AgentStatus>("agent", "/v1/status", signal),
    refetchInterval: 10_000,
    retry: false,
  })

  const clearTranscriptTimer = useCallback(() => {
    if (transcriptTimerRef.current !== null) {
      clearTimeout(transcriptTimerRef.current)
      transcriptTimerRef.current = null
    }
  }, [])

  const disarm = useCallback(() => {
    armedRef.current = false
    releasedRef.current = false
    armAfterTranscriptRef.current = null
    pendingTranscriptsRef.current = []
    seenTranscriptKeysRef.current.clear()
    clearTranscriptTimer()
  }, [clearTranscriptTimer])

  useEffect(() => {
    return () => {
      clearTranscriptTimer()
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current)
    }
  }, [clearTranscriptTimer])

  useEffect(() => {
    disarm()
    setPhase("idle")
  }, [disarm, sessionId])

  useEffect(() => {
    const pushed = heard.replies[heard.replies.length - 1]
    if (!pushed) return
    setTranscript(pushed.question)
    setAnswer(pushed)
    setError(null)
  }, [heard.replies])

  useEffect(() => {
    if (heard.connection !== "listening" && armedRef.current) {
      disarm()
      setPhase("idle")
      setError("Speech connection was interrupted. Hold to try again.")
    }
  }, [disarm, heard.connection])

  const runTurn = useCallback(
    async (next: HeardTranscript) => {
      setTranscript(next.text)
      setAnswer(null)
      setError(null)
      setPlaybackNote(null)
      setPhase("asking")

      try {
        const result = await post<AgentAnswer>("agent", "/v1/agent/query", {
          text: next.text,
          session_id: sessionId,
        })
        setAnswer(result)

        const blob = await postForBlob("speech", "/v1/synthesize", {
          text: result.reply,
        })
        if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current)
        const url = URL.createObjectURL(blob)
        audioUrlRef.current = url

        if (!audioRef.current) {
          setPhase("idle")
          return
        }
        audioRef.current.src = url
        setPhase("speaking")
        try {
          await audioRef.current.play()
        } catch {
          // Browsers may expire the original pointer gesture while STT + LLM
          // are running. Controls remain visible so the answer is never lost.
          setPlaybackNote("Autoplay was blocked — press play below to hear the reply.")
          setPhase("idle")
        }
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught))
        setPhase("idle")
      }
    },
    [sessionId],
  )

  const submitPendingTranscripts = useCallback(() => {
    const pending = pendingTranscriptsRef.current
    if (pending.length === 0) return false

    const latest = pending[pending.length - 1]
    const combined: HeardTranscript = {
      ...latest,
      text: pending
        .map((item) => item.text.trim())
        .filter(Boolean)
        .join(" "),
    }
    disarm()
    void runTurn(combined)
    return true
  }, [disarm, runTurn])

  useEffect(() => {
    if (!armedRef.current || heard.transcripts.length === 0) return
    const latest = heard.transcripts[heard.transcripts.length - 1]
    const key = transcriptKey(latest)
    if (
      key === null ||
      key === armAfterTranscriptRef.current ||
      seenTranscriptKeysRef.current.has(key)
    ) {
      return
    }

    seenTranscriptKeysRef.current.add(key)
    pendingTranscriptsRef.current.push(latest)
    // A silence boundary or the Speech service's maximum utterance length may
    // produce a transcript while the pointer is still down. Hold-to-ask must
    // not start Agent or TTS until a real pointer/key release.
    if (releasedRef.current) submitPendingTranscripts()
  }, [heard.transcripts, submitPendingTranscripts])

  const arm = () => {
    if (!sessionId || heard.connection !== "listening" || phase !== "idle") return
    clearTranscriptTimer()
    const latest = heard.transcripts[heard.transcripts.length - 1]
    armAfterTranscriptRef.current = transcriptKey(latest)
    pendingTranscriptsRef.current = []
    seenTranscriptKeysRef.current.clear()
    releasedRef.current = false
    armedRef.current = true
    setError(null)
    setPlaybackNote(null)
    setPhase("armed")
  }

  const release = () => {
    if (!armedRef.current || releasedRef.current) return
    releasedRef.current = true
    setPhase("waiting")
    clearTranscriptTimer()
    if (submitPendingTranscripts()) return
    transcriptTimerRef.current = setTimeout(() => {
      if (!armedRef.current) return
      disarm()
      setPhase("idle")
      setError("No transcript arrived. Hold to try again.")
    }, TRANSCRIPT_WAIT_TIMEOUT_MS)
  }

  const cancel = () => {
    if (!armedRef.current) return
    disarm()
    setPhase("idle")
    setError("Ask canceled. Hold and release to try again.")
  }

  const backend = status.data
  const readyToArm =
    sessionId !== null && heard.connection === "listening" && phase === "idle"

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle className="text-base">Assistant</CardTitle>
        {status.isError ? (
          <Badge variant="destructive">agent unreachable</Badge>
        ) : backend ? (
          <Badge variant={backend.backend === "external" ? "destructive" : "secondary"}>
            {backend.backend} · {backend.model}
          </Badge>
        ) : null}
      </CardHeader>

      <CardContent className="flex flex-1 flex-col gap-3">
        {backend?.backend === "external" && (
          <p className="rounded-md border border-destructive/30 bg-destructive/10 p-2 text-xs">
            Transcript text is leaving this workstation through {backend.endpoint_host}.
          </p>
        )}

        <div className="flex items-center justify-between gap-2">
          <Badge variant={heard.connection === "listening" ? "default" : "secondary"}>
            {sessionId ? heard.connection : "publish first"}
          </Badge>
          <span className="text-[10px] text-muted-foreground">
            A transcript appears only after you stop speaking.
          </span>
        </div>

        <Button
          className="h-14 w-full select-none touch-none"
          variant={phase === "armed" || phase === "waiting" ? "default" : "outline"}
          disabled={
            phase === "waiting" ||
            phase === "asking" ||
            phase === "speaking" ||
            (phase === "idle" && !readyToArm)
          }
          onPointerDown={(event) => {
            if (event.button !== 0) return
            event.currentTarget.setPointerCapture(event.pointerId)
            arm()
          }}
          onPointerUp={release}
          onPointerCancel={cancel}
          onKeyDown={(event) => {
            if ((event.key === " " || event.key === "Enter") && !event.repeat) arm()
          }}
          onKeyUp={(event) => {
            if (event.key === " " || event.key === "Enter") release()
          }}
        >
          <Mic /> {phaseLabel(phase)}
        </Button>

        <ScrollArea className="min-h-48 flex-1 rounded-lg border">
          <div className="space-y-4 p-3">
            <div>
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                You said
              </div>
              <p className="mt-1 text-sm">
                {transcript ?? "Hold the button, ask where an object is, then release."}
              </p>
            </div>

            <div>
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                Assistant
              </div>
              <p className="mt-1 text-sm leading-relaxed">
                {answer?.reply ?? "No answer yet."}
              </p>
            </div>

            {answer && (
              <div className="flex flex-wrap items-center gap-2">
                {answer.answer_status ? (
                  <Badge variant={STATUS_VARIANT[answer.answer_status]}>
                    {answer.answer_status.replaceAll("_", " ")}
                  </Badge>
                ) : (
                  <Badge variant="secondary">no memory tool result</Badge>
                )}
                <Badge
                  variant={answer.guard === "passed" ? "default" : "outline"}
                  className={
                    answer.guard === "passed"
                      ? undefined
                      : "border-amber-500/50 text-amber-300"
                  }
                >
                  <ShieldCheck /> {answer.guard}
                </Badge>
                <span className="tnum text-[10px] text-muted-foreground">
                  {answer.latency_ms} ms
                </span>
              </div>
            )}
          </div>
        </ScrollArea>

        {error && (
          <p className="text-xs text-destructive">
            {error.includes("agent 404") || error.includes("Failed to fetch")
              ? "No Agent service on /api/agent."
              : error}
          </p>
        )}
        {playbackNote && <p className="text-xs text-amber-300">{playbackNote}</p>}

        <div className="mt-auto space-y-1">
          <div className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground">
            <Volume2 className="size-3" /> Reply audio
          </div>
          <audio
            ref={audioRef}
            controls
            className="w-full"
            onEnded={() => setPhase("idle")}
          />
        </div>
      </CardContent>
    </Card>
  )
}
