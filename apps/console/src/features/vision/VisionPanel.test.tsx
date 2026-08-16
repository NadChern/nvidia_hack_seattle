// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, describe, expect, it, vi } from "vitest"

import { VisionPanel } from "@/features/vision/VisionPanel"

const api = vi.hoisted(() => ({ get: vi.fn() }))
vi.mock("@/lib/api", () => api)

function panel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return (
    <QueryClientProvider client={client}>
      <VisionPanel />
    </QueryClientProvider>
  )
}

const status = {
  ready: true,
  not_ready_reason: null,
  config: {
    reason_kind: "cosmos",
    identity_kind: "radio",
    registration_labels: ["keys", "wallet"],
    registration_capture_seconds: 6,
    registration_target_views: 4,
    registration_min_views: 2,
  },
  reasoner: {
    kind: "cosmos",
    model: "nvidia/Cosmos3-Nano",
    window_seconds: 6,
    interval_seconds: 7,
    max_frames: 4,
    event_cooldown_seconds: 20,
    promote_motion_events: false,
  },
  identity: {
    embedder: {
      identity_embedder: "c-radio-v4",
      ready: true,
      model: "nvidia/C-RADIOv4-H",
    },
    min_cosine: 0.75,
    gallery: {
      registry_version: 7,
      gallery_objects: 3,
      gallery_views: 10,
      gallery_stale: false,
      refresh_failures: 0,
    },
  },
  registration: { attempts: 4, succeeded: 3, failed: 1, active: 0 },
  analysis: { queue_depth: 4, pending: 0, dropped: 0, failed: 0 },
  ingest: { frames_dropped_stale: 0, control_dropped: 0 },
  metrics: {
    frames_processed: 120,
    windows_analyzed: 8,
    events_detected: 4,
    identity_matched: 3,
    identity_skipped: 1,
    motion_events_suppressed: 0,
    events_deduped: 0,
    observations_written: 2,
  },
}

const events = {
  events: [
    {
      at: "2026-08-16T10:30:00Z",
      label: "wallet",
      action: "placed",
      object_id: "object_01M04EXAMPLE123456789",
      outcome: "written",
      score: 0.88,
    },
  ],
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe("VisionPanel Cosmos receipts", () => {
  it("shows registration, window, identity, and Memory-write proof", async () => {
    api.get.mockImplementation((_service: string, path: string) => {
      if (path === "/v1/status") return Promise.resolve(status)
      if (path === "/v1/events") return Promise.resolve(events)
      return Promise.reject(new Error(`unexpected GET ${path}`))
    })

    render(panel())

    expect(await screen.findByText("Cosmos3-Nano")).toBeTruthy()
    expect(screen.getByText("3 objects")).toBeTruthy()
    expect(screen.getByText("10 C-RADIO reference views")).toBeTruthy()
    expect(screen.getByText("personal 88%")).toBeTruthy()
    expect(screen.getByText("wallet")).toBeTruthy()
    expect(screen.getByText("memory written")).toBeTruthy()
    expect(screen.getByText("placed-only writes")).toBeTruthy()
  })

  it("explains how to produce the first receipt", async () => {
    api.get.mockImplementation((_service: string, path: string) => {
      if (path === "/v1/status") return Promise.resolve(status)
      if (path === "/v1/events") return Promise.resolve({ events: [] })
      return Promise.reject(new Error(`unexpected GET ${path}`))
    })

    render(panel())

    expect(await screen.findByText(/No placement receipts yet/)).toBeTruthy()
  })
})
