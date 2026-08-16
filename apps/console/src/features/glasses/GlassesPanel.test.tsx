// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { GlassesPanel } from "@/features/glasses/GlassesPanel"
import { useGlasses } from "@/store/glasses"

const api = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  del: vi.fn(),
}))

vi.mock("@/lib/api", () => api)

describe("GlassesPanel ask for help", () => {
  beforeEach(() => {
    // jsdom does not implement Element.scrollTo; the panel calls it to keep
    // its log pinned to the bottom.
    Object.defineProperty(Element.prototype, "scrollTo", {
      configurable: true,
      value: vi.fn(),
    })
    api.get.mockResolvedValue({ sessions: [] })
    api.post.mockResolvedValue({
      request_id: "assist_01",
      session_id: "sess_01",
      device_id: "browser-glasses",
      state: "requested",
      requested_at: "2026-08-14T00:00:00Z",
      expires_at: "2026-08-14T00:03:00Z",
    })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    useGlasses.setState({ session: null, state: "idle" })
  })

  it("is disabled with no session", async () => {
    render(<GlassesPanel />)
    const button = await screen.findByRole<HTMLButtonElement>("button", {
      name: /ask for help/i,
    })

    expect(button.disabled).toBe(true)
  })

  it("posts a request exactly once when pressed while publishing", async () => {
    useGlasses.setState({
      state: "publishing",
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

    render(<GlassesPanel />)
    const button = await screen.findByRole<HTMLButtonElement>("button", {
      name: /ask for help/i,
    })
    expect(button.disabled).toBe(false)

    fireEvent.click(button)

    await waitFor(() =>
      expect(api.post).toHaveBeenCalledWith("gateway", "/v1/assist/sess_01/request"),
    )
    expect(api.post.mock.calls.filter((call) => call[1] === "/v1/assist/sess_01/request")).toHaveLength(1)
  })
})
