// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { MemoryPanel } from "@/features/memory/MemoryPanel"

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), patch: vi.fn() }))
vi.mock("@/lib/api", () => api)

const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))
vi.mock("sonner", () => ({ toast }))

const EMPTY_GALLERY = { registry_version: 0, unchanged: false, objects: [], views: [] }

beforeEach(() => {
  api.get.mockResolvedValue(EMPTY_GALLERY)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe("MemoryPanel clear memory", () => {
  it("resets memory once the destructive action is confirmed", async () => {
    api.post.mockResolvedValue({ reset: true, registry_version: 5, purged_paths: 2 })
    render(<MemoryPanel />)

    // Opening the dialog alone must not touch memory.
    fireEvent.click(screen.getByRole("button", { name: /clear memory/i }))
    expect(api.post).not.toHaveBeenCalled()

    // The confirm button in the dialog is the second "Clear memory" control.
    const confirm = screen.getAllByRole("button", { name: /^clear memory$/i }).at(-1)!
    fireEvent.click(confirm)

    await waitFor(() =>
      expect(api.post).toHaveBeenCalledWith("memory", "/v1/maintenance/reset"),
    )
    expect(toast.success).toHaveBeenCalledTimes(1)
  })

  it("never resets when the confirmation is dismissed", () => {
    render(<MemoryPanel />)

    fireEvent.click(screen.getByRole("button", { name: /clear memory/i }))
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }))

    expect(api.post).not.toHaveBeenCalled()
  })

  it("lists registered objects and locates one by label on row click", async () => {
    const now = new Date().toISOString()
    api.get.mockResolvedValue({
      registry_version: 3,
      unchanged: false,
      objects: [
        { object_id: "obj_1", label: "a set of keys", created_at: now, updated_at: now, registry_version: 3 },
      ],
      views: [
        { view_id: "v1", object_id: "obj_1", view_index: 0 },
        { view_id: "v2", object_id: "obj_1", view_index: 1 },
      ],
    })
    api.post.mockResolvedValue({ answer_status: "confirmed", spoken_answer: "on the desk" })
    render(<MemoryPanel />)

    fireEvent.click(await screen.findByText("a set of keys"))

    await waitFor(() =>
      expect(api.post).toHaveBeenCalledWith("memory", "/v1/query", { label: "a set of keys" }),
    )
  })

  it("renames a placeholder-labelled object without triggering a locate", async () => {
    const now = new Date().toISOString()
    api.get.mockResolvedValue({
      registry_version: 4,
      unchanged: false,
      objects: [
        { object_id: "obj_2", label: "item a1b2c3", created_at: now, updated_at: now, registry_version: 4 },
      ],
      views: [],
    })
    api.patch.mockResolvedValue({ object_id: "obj_2", label: "car keys", registry_version: 5 })
    render(<MemoryPanel />)

    // Entering rename must not fire a locate query for the placeholder label.
    fireEvent.click(await screen.findByRole("button", { name: /rename "item a1b2c3"/i }))
    fireEvent.change(screen.getByRole("textbox", { name: /new object name/i }), {
      target: { value: "car keys" },
    })
    fireEvent.click(screen.getByRole("button", { name: /save name/i }))

    await waitFor(() =>
      expect(api.patch).toHaveBeenCalledWith("memory", "/v1/objects/obj_2", { label: "car keys" }),
    )
    expect(api.post).not.toHaveBeenCalledWith("memory", "/v1/query", expect.anything())
    expect(toast.success).toHaveBeenCalledTimes(1)
  })
})
