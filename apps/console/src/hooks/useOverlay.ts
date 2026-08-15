import { useCallback, useEffect, useRef, useState } from "react"

import { websocketUrl } from "@/lib/api"
import { OVERLAY_SCHEMA, isOverlayHello } from "@/lib/contracts"
import type { OverlayFrame, OverlayHello } from "@/lib/contracts"

export type OverlayConnection = "idle" | "connecting" | "live" | "closed"

export interface OverlayState {
  connection: OverlayConnection
  hello: OverlayHello | null
  /** The most recent overlay. Held in a ref, not state -- see below. */
  latest: React.RefObject<OverlayFrame | null>
  /** Frames received since connecting, and the rate they arrived at. */
  received: number
  overlayFps: number | null
  latencyMs: number | null
  /** Gaps in `sequence`: frames the pipeline processed that never arrived. */
  missed: number
  schemaMismatch: string | null
  error: string | null
}

/**
 * Subscribes to `WS /v1/overlay` and keeps the newest frame available for
 * drawing.
 *
 * **The newest frame is a ref, not React state.** Overlays arrive at the
 * source frame rate; putting each one through `setState` would re-render the
 * whole panel 8+ times a second to move some rectangles, and the canvas draws
 * from an animation frame anyway. Only the summary counters below are state,
 * and they are throttled to once a second.
 *
 * Reconnects with backoff, because the interesting moment in a demo is often
 * the one right after someone restarts the vision worker.
 */
export function useOverlay(enabled: boolean, sessionId: string | null): OverlayState {
  const latest = useRef<OverlayFrame | null>(null)
  const [connection, setConnection] = useState<OverlayConnection>("idle")
  const [hello, setHello] = useState<OverlayHello | null>(null)
  const [received, setReceived] = useState(0)
  const [overlayFps, setOverlayFps] = useState<number | null>(null)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const [missed, setMissed] = useState(0)
  const [schemaMismatch, setSchemaMismatch] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Counted on the socket's own callback and sampled once a second, so a
  // 30fps stream costs one render per second rather than thirty.
  const counters = useRef({ received: 0, missed: 0, lastSequence: -1, latencySum: 0 })

  const reset = useCallback(() => {
    counters.current = { received: 0, missed: 0, lastSequence: -1, latencySum: 0 }
    latest.current = null
    setReceived(0)
    setMissed(0)
    setOverlayFps(null)
    setLatencyMs(null)
  }, [])

  useEffect(() => {
    if (!enabled || !sessionId) {
      setConnection("idle")
      return
    }

    let socket: WebSocket | null = null
    let retry: number | undefined
    let attempt = 0
    let disposed = false

    const connect = () => {
      if (disposed) return
      setConnection("connecting")
      socket = new WebSocket(
        websocketUrl("vision", `/v1/overlay?session_id=${encodeURIComponent(sessionId)}`),
      )

      socket.onopen = () => {
        attempt = 0
        setError(null)
        reset()
      }

      socket.onmessage = (event: MessageEvent<string>) => {
        let parsed: unknown
        try {
          parsed = JSON.parse(event.data)
        } catch {
          return
        }

        if (isOverlayHello(parsed)) {
          setHello(parsed)
          setConnection("live")
          if (parsed.session_id !== sessionId) {
            setError(`overlay connected to ${parsed.session_id ?? "all sessions"}, expected ${sessionId}`)
            return
          }
          // A mismatch is not fatal -- the fields this console reads may well
          // still be present -- so it is surfaced rather than thrown. Silence
          // is the failure mode worth avoiding: boxes in the wrong place look
          // like a broken detector, not a renamed field.
          setSchemaMismatch(
            parsed.schema_version === OVERLAY_SCHEMA
              ? null
              : `console expects ${OVERLAY_SCHEMA}, service sent ${parsed.schema_version}`,
          )
          return
        }

        const frame = parsed as OverlayFrame
        if (typeof frame.sequence !== "number" || frame.session_id !== sessionId) return

        const c = counters.current
        if (c.lastSequence >= 0 && frame.sequence > c.lastSequence + 1) {
          c.missed += frame.sequence - c.lastSequence - 1
        }
        c.lastSequence = frame.sequence
        c.received += 1
        c.latencySum += frame.pipeline_latency_ms
        latest.current = frame
      }

      socket.onerror = () => setError("overlay socket error")

      socket.onclose = () => {
        if (disposed) return
        setConnection("closed")
        attempt += 1
        // Capped backoff. A vision worker being restarted is the common case,
        // and hammering it while it loads a detector helps nobody.
        const delay = Math.min(1000 * 2 ** (attempt - 1), 10_000)
        retry = window.setTimeout(connect, delay)
      }
    }

    const sampler = window.setInterval(() => {
      const c = counters.current
      setReceived(c.received)
      setMissed(c.missed)
      setOverlayFps(c.received > 0 ? c.received : 0)
      setLatencyMs(c.received > 0 ? c.latencySum / c.received : null)
      c.received = 0
      c.latencySum = 0
    }, 1000)

    connect()

    return () => {
      disposed = true
      window.clearInterval(sampler)
      if (retry !== undefined) window.clearTimeout(retry)
      socket?.close()
    }
  }, [enabled, reset, sessionId])

  return {
    connection,
    hello,
    latest,
    received,
    overlayFps,
    latencyMs,
    missed,
    schemaMismatch,
    error,
  }
}
