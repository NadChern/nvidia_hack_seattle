// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { EnrollmentPanel } from "@/features/enrollment/EnrollmentPanel"
import { useGlasses } from "@/store/glasses"

const api = vi.hoisted(() => ({
  get: vi.fn(),
  getBlob: vi.fn(),
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

const progress = {
  object_id: "object-keys-01",
  label: "keys",
  state: "succeeded",
  frames_total: 30,
  detections: 12,
  quality_passed: 8,
  selected_views: 4,
  reason_code: null,
  message: "Four diverse views selected.",
}

describe("EnrollmentPanel", () => {
  beforeEach(() => {
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
    api.get.mockImplementation((_service: string, path: string) => {
      if (path === "/v1/status") {
        return Promise.resolve({ config: { registration_labels: ["keys", "wallet"] } })
      }
      if (path.endsWith("/status")) return Promise.resolve(progress)
      if (path === "/v1/objects") {
        return Promise.resolve({
          registry_version: 2,
          objects: [{ object_id: "object-keys-01", label: "keys", registry_version: 2 }],
          views: [
            {
              view_id: "view-01",
              object_id: "object-keys-01",
              view_index: 0,
              crop_reference: "/v1/objects/object-keys-01/views/view-01/crop",
            },
          ],
        })
      }
      return Promise.reject(new Error(`unexpected GET ${path}`))
    })
    api.post.mockImplementation((_service: string, path: string) => {
      if (path === "/v1/objects") {
        return Promise.resolve({ object_id: "object-keys-01", label: "keys", registry_version: 1 })
      }
      return Promise.resolve({ ...progress, state: "capturing", selected_views: 0 })
    })
    api.getBlob.mockResolvedValue(new Blob(["jpeg"], { type: "image/jpeg" }))
    api.delChecked.mockResolvedValue(undefined)
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:reference"),
      revokeObjectURL: vi.fn(),
    })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    vi.unstubAllGlobals()
    useGlasses.setState({ session: null })
  })

  async function completeRegistration() {
    render(panel())
    await waitFor(() => expect(screen.getByRole("option", { name: "keys" })).toBeTruthy())
    fireEvent.click(screen.getByRole("button", { name: /start registration/i }))
    await waitFor(() => expect(screen.getByAltText("Selected reference 1")).toBeTruthy())
  }

  it("creates, captures, polls, previews, and can discard an object", async () => {
    await completeRegistration()

    await waitFor(() => expect(api.post).toHaveBeenCalledTimes(2))
    expect(api.post).toHaveBeenNthCalledWith(1, "vision", "/v1/objects", {
      label: "keys",
      idempotency_key: expect.stringContaining("console/sess_01/keys/"),
    })
    expect(api.post).toHaveBeenNthCalledWith(
      2,
      "vision",
      "/v1/objects/object-keys-01/capture",
      {},
    )

    expect(api.getBlob).toHaveBeenCalledWith(
      "memory",
      "/v1/objects/object-keys-01/views/view-01/crop",
      expect.any(AbortSignal),
    )

    fireEvent.click(screen.getByRole("button", { name: /discard/i }))
    await waitFor(() =>
      expect(api.delChecked).toHaveBeenCalledWith("memory", "/v1/objects/object-keys-01"),
    )
  })

  it("confirms by keeping the durable object", async () => {
    await completeRegistration()

    fireEvent.click(screen.getByRole("button", { name: /^confirm$/i }))

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /start registration/i }).hasAttribute("disabled")).toBe(false),
    )
    expect(api.delChecked).not.toHaveBeenCalled()
  })
})
