import { useEffect, useRef, useState } from "react"

import { websocketUrl } from "@/lib/api"
import type { DeviceReplyEvent, Transcript } from "@/lib/contracts"

export type TranscriptConnection = "idle" | "connecting" | "listening" | "closed"

export interface HeardTranscript extends Transcript {
  /** When this browser received it -- the speech service sends no wall clock. */
  receivedAt: string
}

const MAX_KEPT = 50

export interface TranscriptStream {
  connection: TranscriptConnection
  transcripts: HeardTranscript[]
  replies: DeviceReplyEvent[]
  error: string | null
  clear: () => void
}

/**
 * Subscribes to the Gateway's per-session device event stream.
 *
 * The Agent owns the one Speech STT socket for hands-free operation and pushes
 * each transcript plus guarded reply here. The console must not open another
 * Parakeet consumer, and the glasses need only the Gateway URL and credential.
 *
 * A transcript arrives only when a contiguous segment ends, so silence
 * produces nothing and there is no partial/interim result to render. The
 * connection state is therefore the only feedback available while the wearer
 * is mid-sentence, which is why it is surfaced rather than kept internal.
 */
export function useTranscripts(sessionId: string | null): TranscriptStream {
  const [connection, setConnection] = useState<TranscriptConnection>("idle")
  const [transcripts, setTranscripts] = useState<HeardTranscript[]>([])
  const [replies, setReplies] = useState<DeviceReplyEvent[]>([])
  const [error, setError] = useState<string | null>(null)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    setTranscripts([])
    setReplies([])
    setError(null)
    if (!sessionId) {
      setConnection("idle")
      return
    }

    let disposed = false
    let retry: number | undefined
    let attempt = 0

    const connect = () => {
      if (disposed) return
      setConnection("connecting")
      const socket = new WebSocket(
        websocketUrl("gateway", `/v1/device/${sessionId}/events`),
      )
      socketRef.current = socket

      socket.onopen = () => {
        attempt = 0
        setError(null)
        setConnection("listening")
      }

      socket.onmessage = (event: MessageEvent<string>) => {
        try {
          const payload = JSON.parse(event.data) as { type?: string }
          if (payload.type === "transcript") {
            const transcript = payload as Transcript
            if (typeof transcript.text !== "string") return
            setTranscripts((kept) =>
              [...kept, { ...transcript, receivedAt: new Date().toISOString() }].slice(
                -MAX_KEPT,
              ),
            )
          } else if (payload.type === "reply") {
            const reply = payload as DeviceReplyEvent
            if (typeof reply.reply !== "string" || typeof reply.guard !== "string") return
            setReplies((kept) => [...kept, reply].slice(-MAX_KEPT))
          }
        } catch {
          // A frame that is not a transcript is not worth tearing the socket
          // down for; the next segment will be.
        }
      }

      socket.onerror = () => setError("device event socket error")

      socket.onclose = () => {
        if (disposed) return
        setConnection("closed")
        attempt += 1
        const delay = Math.min(1000 * 2 ** (attempt - 1), 10_000)
        retry = window.setTimeout(connect, delay)
      }
    }

    connect()

    return () => {
      disposed = true
      if (retry !== undefined) window.clearTimeout(retry)
      const socket = socketRef.current
      socketRef.current = null
      if (!socket) return

      if (socket.readyState === WebSocket.CONNECTING) {
        // Calling close() during CONNECTING makes Chromium report
        // "closed before the connection is established". Finish the handshake
        // and immediately close instead; disposed callbacks cannot update UI.
        socket.onopen = () => socket.close(1000, "subscription replaced")
        socket.onmessage = null
        socket.onerror = null
      } else if (socket.readyState === WebSocket.OPEN) {
        socket.close(1000, "subscription replaced")
      }
    }
  }, [sessionId])

  return {
    connection,
    transcripts,
    replies,
    error,
    clear: () => {
      setTranscripts([])
      setReplies([])
    },
  }
}
