// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { EnrollmentPanel } from "@/features/enrollment/EnrollmentPanel"
import * as glasses from "@/store/glasses"

const api = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  delChecked: vi.fn(),
}))

vi.mock("@/lib/api", () => api)

function panel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return (
    <QueryClientProvider client={client}>
      <EnrollmentPanel />
    </QueryClientProvider>
  )
}

const completed = {
  object_id: "object-keys-01",
  label: "keys",
  state: "succeeded",
  started_at: "2026-08-16T12:00:00Z",
  capture_ends_at: "2026-08-16T12:00:00Z",
  frames_total: 2,
  detections: 2,
  quality_passed: 2,
  selected_views: 2,
  reason_code: "manual_enrollment_complete",
  message: "operator-confirmed reference gallery stored",
}

describe("EnrollmentPanel", () => {
  beforeEach(() => {
    glasses.useGlasses.setState({
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
    vi.spyOn(glasses, "capturePreviewJpeg").mockResolvedValue(
      new Blob(["frozen-jpeg"], { type: "image/jpeg" }),
    )
    api.get.mockResolvedValue({ config: { registration_labels: ["keys", "wallet"] } })
    api.post.mockImplementation((_service: string, path: string) => {
      if (path === "/v1/objects") {
        return Promise.resolve({ object_id: "object-keys-01", label: "keys", registry_version: 1 })
      }
      if (path === "/v1/objects/object-keys-01/manual") return Promise.resolve(completed)
      return Promise.reject(new Error(`unexpected POST ${path}`))
    })
    api.delChecked.mockResolvedValue(undefined)

    let url = 0
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => `blob:reference-${++url}`),
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn(),
    })
    vi.stubGlobal(
      "createImageBitmap",
      vi.fn().mockResolvedValue({ width: 720, height: 720, close: vi.fn() }),
    )
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      drawImage: vi.fn(),
    } as unknown as CanvasRenderingContext2D)
    vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation((callback) => {
      callback(new Blob(["crop-jpeg"], { type: "image/jpeg" }))
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    glasses.useGlasses.setState({ session: null })
  })

  async function stageView() {
    fireEvent.click(screen.getByRole("button", { name: /freeze current pov/i }))
    await screen.findByAltText("Frozen glasses POV")
    fireEvent.click(screen.getByRole("button", { name: /add selected crop/i }))
    await waitFor(() => expect(screen.getAllByAltText(/Pending reference/).length).toBeGreaterThan(0))
  }

  it("keeps operator crops local until explicit confirmation", async () => {
    render(panel())
    await waitFor(() => expect(screen.getByRole("option", { name: "keys" })).toBeTruthy())

    await stageView()

    expect(api.post).not.toHaveBeenCalled()
    expect(screen.getByRole("button", { name: /confirm and register/i }).hasAttribute("disabled")).toBe(true)
  })

  it("previews and removes a pending crop", async () => {
    render(panel())
    await waitFor(() => expect(screen.getByRole("option", { name: "keys" })).toBeTruthy())
    await stageView()

    fireEvent.click(screen.getByRole("button", { name: "Enlarge pending reference 1" }))
    expect(await screen.findByRole("dialog")).toBeTruthy()
    expect(screen.getByRole("heading", { name: "Pending reference 1" })).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: "Close" }))

    fireEvent.click(screen.getByRole("button", { name: "Remove pending reference 1" }))
    await waitFor(() => expect(screen.queryByAltText("Pending reference 1")).toBeNull())
  })

  it("creates and persists the object only after two crops are confirmed", async () => {
    render(panel())
    await waitFor(() => expect(screen.getByRole("option", { name: "keys" })).toBeTruthy())
    await stageView()
    await stageView()

    fireEvent.click(screen.getByRole("button", { name: /confirm and register/i }))

    await waitFor(() => expect(api.post).toHaveBeenCalledTimes(2))
    expect(api.post).toHaveBeenNthCalledWith(1, "vision", "/v1/objects", {
      label: "keys",
      idempotency_key: expect.stringContaining("console/manual/sess_01/keys/"),
    })
    expect(api.post).toHaveBeenNthCalledWith(
      2,
      "vision",
      "/v1/objects/object-keys-01/manual",
      { views_base64: [expect.any(String), expect.any(String)] },
    )
    expect(await screen.findByText(/2 C-RADIO views stored/i)).toBeTruthy()
    expect(api.delChecked).not.toHaveBeenCalled()
  })
})
