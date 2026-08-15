// @vitest-environment jsdom

import { act, render, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it } from "vitest"

import { TranscriptProvider } from "@/hooks/TranscriptProvider"
import { useTranscriptStream } from "@/hooks/transcript-context"

class FakeWebSocket {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static instances: FakeWebSocket[] = []

  readonly url: string
  readyState = FakeWebSocket.CONNECTING
  closeCalls = 0
  onopen: (() => void) | null = null
  onmessage: ((event: MessageEvent<string>) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null

  constructor(url: string | URL) {
    this.url = String(url)
    FakeWebSocket.instances.push(this)
  }

  close(_code?: number, _reason?: string) {
    this.closeCalls += 1
  }
}

function Consumer() {
  const stream = useTranscriptStream()
  return (
    <div>
      consumer {stream.transcripts.at(-1)?.text ?? "none"} {stream.replies.at(-1)?.reply ?? "none"}
    </div>
  )
}

describe("TranscriptProvider", () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
    Object.defineProperty(globalThis, "WebSocket", {
      configurable: true,
      value: FakeWebSocket,
    })
  })

  afterEach(() => {
    Reflect.deleteProperty(globalThis, "WebSocket")
  })

  it("defers closing a connecting socket until its handshake completes", async () => {
    const view = render(
      <TranscriptProvider sessionId="sess_01">
        <Consumer />
      </TranscriptProvider>,
    )
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    const first = FakeWebSocket.instances[0]

    view.rerender(
      <TranscriptProvider sessionId="sess_02">
        <Consumer />
      </TranscriptProvider>,
    )
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(2))

    expect(first.closeCalls).toBe(0)
    act(() => first.onopen?.())
    expect(first.closeCalls).toBe(1)
  })

  it("keeps one Gateway event socket while tab consumers mount and unmount", async () => {
    const view = render(
      <TranscriptProvider sessionId="sess_01">
        <Consumer />
      </TranscriptProvider>,
    )

    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    const socket = FakeWebSocket.instances[0]
    expect(socket.url).toContain("/api/gateway/v1/device/sess_01/events")

    view.rerender(
      <TranscriptProvider sessionId="sess_01">
        <div>another tab</div>
      </TranscriptProvider>,
    )
    view.rerender(
      <TranscriptProvider sessionId="sess_01">
        <Consumer />
      </TranscriptProvider>,
    )

    expect(FakeWebSocket.instances).toHaveLength(1)
    expect(socket.closeCalls).toBe(0)

    act(() => {
      socket.readyState = FakeWebSocket.OPEN
      socket.onopen?.()
    })
    view.unmount()
    expect(socket.closeCalls).toBe(1)
  })

  it("shares transcript and guarded reply events from one socket", async () => {
    const view = render(
      <TranscriptProvider sessionId="sess_01">
        <Consumer />
      </TranscriptProvider>,
    )
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    const socket = FakeWebSocket.instances[0]

    act(() => {
      socket.onmessage?.(
        new MessageEvent("message", {
          data: JSON.stringify({
            type: "transcript",
            text: "hey memory where are my keys",
            session_id: "sess_01",
            epoch_id: "TR_AUDIO_1",
            pts_samples_start: 0,
            samples: 16000,
            sample_rate: 16000,
          }),
        }),
      )
      socket.onmessage?.(
        new MessageEvent("message", {
          data: JSON.stringify({
            schema_version: "1.0",
            type: "reply",
            session_id: "sess_01",
            question: "where are my keys",
            reply: "I cannot safely confirm a location.",
            answer_status: "unknown",
            object_id: null,
            guard: "vetoed:3",
            latency_ms: 42,
            occurred_at: "2026-08-12T18:00:00Z",
          }),
        }),
      )
    })

    await waitFor(() => expect(view.getByText(/hey memory where are my keys/)).toBeTruthy())
    expect(view.getByText(/cannot safely confirm/)).toBeTruthy()
  })
})
