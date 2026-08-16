import { useEffect, useRef } from "react"

import type { MotionState, OverlayFrame } from "@/lib/contracts"
import { containedMediaRect } from "@/features/video/media-geometry"

/**
 * Colour carries meaning here. Watching a box go amber -> yellow -> green as an
 * object is set down is the clearest evidence that a state machine is running
 * rather than a detector firing once per frame, which is the single most
 * useful thing this console can show.
 *
 * Read from the CSS variables in `index.css` so the legend beside the video and
 * the pixels drawn on it cannot drift apart.
 */
const STATE_VARIABLE: Record<MotionState, string> = {
  moving: "--state-moving",
  settling: "--state-settling",
  at_rest: "--state-at-rest",
  absent: "--state-absent",
}

function resolveStateColors(): Record<MotionState, string> {
  const styles = getComputedStyle(document.documentElement)
  const resolved = {} as Record<MotionState, string>
  for (const [state, variable] of Object.entries(STATE_VARIABLE)) {
    resolved[state as MotionState] = styles.getPropertyValue(variable).trim() || "#22d3ee"
  }
  return resolved
}

export interface OverlayCanvasProps {
  /** The newest overlay, held in a ref so new frames never re-render React. */
  frame: React.RefObject<OverlayFrame | null>
  /** The element the boxes are drawn over; supplies the display size. */
  videoRef: React.RefObject<HTMLVideoElement | null>
  showLabels: boolean
}

/**
 * Draws the newest overlay over the video, on an animation frame.
 *
 * Driven by `requestAnimationFrame` rather than by arriving messages, so the
 * draw rate follows the display instead of the network: a burst of overlays
 * costs one draw, and a gap leaves the last boxes up rather than blanking.
 *
 * Coordinates are normalised 0..1 in the contract, which is what lets this
 * scale to whatever size the video happens to be laid out at without knowing
 * the source resolution.
 */
export function OverlayCanvas({ frame, videoRef, showLabels }: OverlayCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext("2d")
    if (!context) return

    let running = true
    let colors = resolveStateColors()
    let colorsCheckedAt = 0

    const draw = (now: number) => {
      if (!running) return
      window.requestAnimationFrame(draw)

      const video = videoRef.current
      if (!video) return

      // Match the canvas to the video's *displayed* size, times the device
      // pixel ratio. Without the ratio, boxes are visibly soft on any HiDPI
      // screen -- which is every laptop a demo runs on.
      const ratio = window.devicePixelRatio || 1
      const width = video.clientWidth
      const height = video.clientHeight
      if (width === 0 || height === 0) return
      if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
        canvas.width = width * ratio
        canvas.height = height * ratio
        canvas.style.width = `${width}px`
        canvas.style.height = `${height}px`
      }

      context.setTransform(ratio, 0, 0, ratio, 0, 0)
      context.clearRect(0, 0, width, height)

      // Theme changes are rare; re-resolving every frame would mean a full
      // style recalculation 60 times a second.
      if (now - colorsCheckedAt > 2000) {
        colors = resolveStateColors()
        colorsCheckedAt = now
      }

      const current = frame.current
      if (!current) return
      const ageMs = Date.now() - Date.parse(current.emitted_at)
      if (!Number.isFinite(ageMs) || ageMs > 2_000) return

      // object-contain letterboxes the media inside the video element. Draw into
      // that same rectangle rather than stretching boxes across the element.
      const media = containedMediaRect(width, height, video.videoWidth, video.videoHeight)

      for (const track of current.tracks) {
        const color = colors[track.motion_state]
        const x = media.x + track.box.x_min * media.width
        const y = media.y + track.box.y_min * media.height
        const boxWidth = (track.box.x_max - track.box.x_min) * media.width
        const boxHeight = (track.box.y_max - track.box.y_min) * media.height

        const registered = track.identity?.object_id !== null && track.identity?.object_id !== undefined
        context.lineWidth = registered ? 4 : 2
        context.strokeStyle = registered ? "#f8fafc" : color
        context.strokeRect(x, y, boxWidth, boxHeight)

        if (!showLabels) continue

        // Depth is sampled about once a second, so it is shown with its age
        // whenever it is not fresh. A stale range quietly presented as current
        // is the kind of wrong number a demo audience would believe.
        const stale = track.depth_age_s !== null && track.depth_age_s >= 1.5
        const depth =
          track.depth_m === null
            ? ""
            : `  ${track.depth_m.toFixed(2)}m${stale ? ` (${track.depth_age_s!.toFixed(0)}s ago)` : ""}`
        const identityScore = track.identity?.best_score
        const identity = registered
          ? `  personal${identityScore === null || identityScore === undefined ? "" : ` ${(identityScore * 100).toFixed(0)}%`}`
          : ""
        const text = `${track.label}  ${(track.confidence * 100).toFixed(0)}%${identity}${depth}`
        context.font =
          "500 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
        const metrics = context.measureText(text)
        const padding = 5
        const labelHeight = 18
        const labelWidth = Math.min(metrics.width + padding * 2, media.width)
        // Keep labels within the actual image, including right-edge detections.
        const labelX = Math.min(Math.max(x, media.x), media.x + media.width - labelWidth)
        const labelY =
          y - labelHeight < media.y
            ? Math.min(y + boxHeight, media.y + media.height - labelHeight)
            : y - labelHeight

        context.fillStyle = color
        context.fillRect(labelX, labelY, labelWidth, labelHeight)
        context.save()
        context.beginPath()
        context.rect(labelX, labelY, labelWidth, labelHeight)
        context.clip()
        context.fillStyle = "#0b0f14"
        context.fillText(text, labelX + padding, labelY + 13)
        context.restore()

        // The state itself, on the box's other edge. A colour alone is not
        // readable by everyone, and a demo audience should not have to learn
        // a key to follow what is happening.
        context.font = "500 10px ui-monospace, SFMono-Regular, Menlo, monospace"
        context.fillStyle = color
        context.fillText(
          track.motion_state,
          Math.max(x, media.x),
          Math.min(labelY + labelHeight + 11, media.y + media.height - 2),
        )
      }
    }

    const handle = window.requestAnimationFrame(draw)
    return () => {
      running = false
      window.cancelAnimationFrame(handle)
    }
  }, [frame, videoRef, showLabels])

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0"
      aria-hidden="true"
    />
  )
}
