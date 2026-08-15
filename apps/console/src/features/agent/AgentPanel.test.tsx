// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AgentPanel } from "@/features/agent/AgentPanel"
import { TranscriptContext } from "@/hooks/transcript-context"
import type { HeardTranscript, TranscriptStream } from "@/hooks/useTranscripts"
import { useGlasses } from "@/store/glasses"

const api = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  postForBlob: vi.fn(),
}))

vi.mock("@/lib/api", () => api)

const transcript: HeardTranscript = {
  text: "Where are my keys?",
  session_id: "sess_01",
  epoch_id: "TR_AUDIO_1",
  pts_samples_start: 0,
  samples: 16_000,
  sample_rate: 16_000,
  receivedAt: "2026-08-12T03:51:09.000Z",
}

function panel(transcripts: HeardTranscript[]) {
  const stream: TranscriptStream = {
    connection: "listening",
    transcripts,
    replies: [],
    error: null,
    clear: vi.fn(),
  }
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return (
    <QueryClientProvider client={client}>
      <TranscriptContext.Provider value={stream}>
        <AgentPanel />
      </TranscriptContext.Provider>
    </QueryClientProvider>
  )
}

describe("AgentPanel hold-to-ask", () => {
  beforeEach(() => {
    api.get.mockResolvedValue({
      backend: "external",
      model: "openrouter/test",
      endpoint_host: "openrouter.ai",
    })
    api.post.mockResolvedValue({
      reply: "I have no record of the keys.",
      answer_status: "unknown",
      object_id: null,
      guard: "vetoed:2",
      latency_ms: 10,
    })
    api.postForBlob.mockResolvedValue(new Blob(["wav"], { type: "audio/wav" }))
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:test"),
      revokeObjectURL: vi.fn(),
    })
    Object.defineProperty(HTMLElement.prototype, "setPointerCapture", {
      configurable: true,
      value: vi.fn(),
    })
    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      value: vi.fn().mockResolvedValue(undefined),
    })
    useGlasses.setState({
      session: {
        session_id: "sess_01",
        device_id: "browser-glasses",
        room: "room",
        identity: "browser-glasses",
        token: "redacted",
        livekit_url: "ws://127.0.0.1:7880",
        expires_at: "2099-01-01T00:00:00Z",
      },
    })
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.clearAllMocks()
    useGlasses.setState({ session: null })
  })

  it("does not run Agent or TTS when a transcript arrives while still held", async () => {
    const view = render(panel([]))
    const button = screen.getByRole("button", { name: /hold to ask/i })

    fireEvent.pointerDown(button, { button: 0, pointerId: 1 })
    view.rerender(panel([transcript]))

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /listening.*release when done/i })).toBeTruthy(),
    )
    expect(api.post).not.toHaveBeenCalled()
    expect(api.postForBlob).not.toHaveBeenCalled()
  })

  it("aborts rather than submitting when the browser cancels the pointer", async () => {
    const view = render(panel([]))
    const button = screen.getByRole("button", { name: /hold to ask/i })

    fireEvent.pointerDown(button, { button: 0, pointerId: 1 })
    view.rerender(panel([transcript]))
    fireEvent.pointerCancel(button, { pointerId: 1 })

    await waitFor(() => expect(screen.getByText(/ask canceled/i)).toBeTruthy())
    expect(api.post).not.toHaveBeenCalled()
    expect(api.postForBlob).not.toHaveBeenCalled()
  })

  it("submits the held transcript only after an actual pointer release", async () => {
    const view = render(panel([]))
    const button = screen.getByRole("button", { name: /hold to ask/i })

    fireEvent.pointerDown(button, { button: 0, pointerId: 1 })
    view.rerender(panel([transcript]))
    expect(api.post).not.toHaveBeenCalled()

    fireEvent.pointerUp(button, { pointerId: 1 })

    await waitFor(() => expect(api.post).toHaveBeenCalledTimes(1))
    expect(api.post).toHaveBeenCalledWith("agent", "/v1/agent/query", {
      text: transcript.text,
      session_id: "sess_01",
    })
    await waitFor(() => expect(api.postForBlob).toHaveBeenCalledTimes(1))
  })
})
