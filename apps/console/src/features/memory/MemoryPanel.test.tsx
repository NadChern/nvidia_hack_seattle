// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { MemoryPanel } from "@/features/memory/MemoryPanel"

const api = vi.hoisted(() => ({ post: vi.fn() }))
vi.mock("@/lib/api", () => api)

const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))
vi.mock("sonner", () => ({ toast }))

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
})
