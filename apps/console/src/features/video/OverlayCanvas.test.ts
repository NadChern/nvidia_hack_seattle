import { describe, expect, it } from "vitest"

import { containedMediaRect } from "@/features/video/media-geometry"

describe("containedMediaRect", () => {
  it("centers portrait media inside a wide stage", () => {
    expect(containedMediaRect(1000, 600, 720, 1280)).toEqual({
      x: 331.25,
      y: 0,
      width: 337.5,
      height: 600,
    })
  })

  it("centers landscape media inside a tall stage", () => {
    expect(containedMediaRect(600, 800, 1280, 720)).toEqual({
      x: 0,
      y: 231.25,
      width: 600,
      height: 337.5,
    })
  })
})
