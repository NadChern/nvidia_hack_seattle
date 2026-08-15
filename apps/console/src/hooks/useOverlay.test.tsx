// @vitest-environment jsdom

import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it } from "vitest"

import { useOverlay } from "@/hooks/useOverlay"

class FakeWebSocket {
  static instances: FakeWebSocket[] = []

  readonly url: string
  onopen: (() => void) | null = null
  onmessage: ((event: MessageEvent<string>) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null

  constructor(url: string | URL) {
    this.url = String(url)
    FakeWebSocket.instances.push(this)
  }

  close() {}
}

function message(value: unknown): MessageEvent<string> {
  return new MessageEvent("message", { data: JSON.stringify(value) })
}

describe("useOverlay", () => {
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

  it("scopes the socket and accepted frames to the selected session", async () => {
    const hook = renderHook(() => useOverlay(true, "sess_selected"))
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    const socket = FakeWebSocket.instances[0]

    expect(new URL(socket.url).searchParams.get("session_id")).toBe("sess_selected")

    act(() => {
      socket.onmessage?.(
        message({
          type: "overlay_hello",
          schema_version: "1.3",
          source_fps: 8,
          detector_kind: "yoloe",
          depth_kind: "yolo",
          session_id: "sess_selected",
        }),
      )
      socket.onmessage?.(
        message({
          schema_version: "1.3",
          session_id: "sess_other",
          media_epoch_id: "TR_other",
          sequence: 1,
          tracks: [],
          pipeline_latency_ms: 10,
        }),
      )
    })
    expect(hook.result.current.latest.current).toBeNull()

    act(() => {
      socket.onmessage?.(
        message({
          schema_version: "1.3",
          session_id: "sess_selected",
          media_epoch_id: "TR_selected",
          sequence: 2,
          tracks: [],
          pipeline_latency_ms: 12,
        }),
      )
    })
    expect(hook.result.current.latest.current?.media_epoch_id).toBe("TR_selected")
    hook.unmount()
  })
})
