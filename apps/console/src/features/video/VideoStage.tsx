import { useCallback, useEffect, useRef, useState } from "react"

import { OverlayCanvas } from "@/features/video/OverlayCanvas"
import { useOverlay } from "@/hooks/useOverlay"
import type { MotionState } from "@/lib/contracts"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useGlasses } from "@/store/glasses"

type ViewMode = "raw" | "boxes"

const STATE_LABEL: Record<MotionState, string> = {
  moving: "moving",
  settling: "settling",
  at_rest: "at rest",
  absent: "absent",
}

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
      {(Object.keys(STATE_LABEL) as MotionState[])
        .filter((state) => state !== "absent")
        .map((state) => (
          <span key={state} className="flex items-center gap-1.5">
            <span
              className="size-2.5 rounded-[3px]"
              style={{ backgroundColor: `var(--state-${state.replace("_", "-")})` }}
            />
            {STATE_LABEL[state]}
          </span>
        ))}
    </div>
  )
}

/** A number that means something only when you can see it move. */
function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  const body = (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</span>
      <span className="tnum text-sm font-medium">{value}</span>
    </div>
  )
  if (!hint) return body
  return (
    <Tooltip>
      <TooltipTrigger asChild>{body}</TooltipTrigger>
      <TooltipContent className="max-w-xs">{hint}</TooltipContent>
    </Tooltip>
  )
}

export function VideoStage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [mode, setMode] = useState<ViewMode>("boxes")
  const attachPreview = useGlasses((s) => s.attachPreview)
  const sessionId = useGlasses((s) => s.session?.session_id ?? null)
  const mediaLive = useGlasses((s) => s.state === "publishing" || s.state === "viewing")

  // Only subscribe while boxes are actually being shown. A viewer that is not
  // drawing anything still occupies one of the vision worker's overlay slots.
  const overlay = useOverlay(mode === "boxes" && mediaLive, sessionId)

  const setVideo = useCallback(
    (element: HTMLVideoElement | null) => {
      videoRef.current = element
      attachPreview(element)
    },
    [attachPreview],
  )

  useEffect(() => () => attachPreview(null), [attachPreview])

  return (
    <Card className="flex h-full min-h-0 flex-col overflow-hidden py-4">
      <CardHeader className="flex-row items-center justify-between gap-4 space-y-0 px-4">
        <CardTitle className="text-base">What the glasses see</CardTitle>
        <ToggleGroup
          type="single"
          value={mode}
          onValueChange={(next) => next && setMode(next as ViewMode)}
          variant="outline"
          size="sm"
        >
          <ToggleGroupItem value="raw">raw</ToggleGroupItem>
          {/* No separate depth view. Depth reaches the console as a number
              per object, drawn on the box itself -- a full scene depth map
              would need MoGe per frame, which is exactly the cost the
              once-a-second cadence exists to avoid, and it would say less
              about the objects being tracked. */}
          <ToggleGroupItem value="boxes">boxes</ToggleGroupItem>
        </ToggleGroup>
      </CardHeader>

      <CardContent className="flex min-h-0 flex-1 flex-col gap-3 px-4">
        <div className="relative min-h-0 flex-1 overflow-hidden rounded-lg bg-black">
          <video
            ref={setVideo}
            autoPlay
            playsInline
            muted
            className="h-full w-full object-contain"
          />
          {mode === "boxes" && (
            <OverlayCanvas frame={overlay.latest} videoRef={videoRef} showLabels />
          )}
          {!mediaLive && (
            <div className="absolute inset-0 grid place-items-center text-sm text-muted-foreground">
              Publish the virtual glasses or select a live device to see its camera.
            </div>
          )}
        </div>

        {mode === "boxes" && (
          <div className="flex flex-wrap items-center justify-between gap-4">
            <Legend />
            <div className="flex items-center gap-5">
              <Metric
                label="latency"
                value={
                  overlay.latencyMs === null ? "—" : `${overlay.latencyMs.toFixed(0)} ms`
                }
                hint="Time from the gateway handing a frame to the pipeline until the overlay was emitted. Both stamps come from the vision worker, so this measures the pipeline rather than the gap between two machines' clocks."
              />
              <Metric
                label="overlay fps"
                value={overlay.overlayFps === null ? "—" : overlay.overlayFps.toFixed(0)}
              />
              <Metric
                label="missed"
                value={String(overlay.missed)}
                hint="Gaps in the frame sequence: overlays the pipeline produced that never arrived here, because this browser was not reading fast enough."
              />
              <Badge
                variant={overlay.connection === "live" ? "default" : "secondary"}
                className="capitalize"
              >
                {overlay.hello?.detector_kind ?? "detection"} · {overlay.connection}
              </Badge>
              <Badge variant="outline">
                depth: {overlay.hello?.depth_kind ?? "—"}
              </Badge>
            </div>
          </div>
        )}

        {overlay.error && (
          <p className="text-xs text-destructive">{overlay.error}</p>
        )}
        {overlay.schemaMismatch && (
          <p className="text-xs text-destructive">
            Overlay schema mismatch — {overlay.schemaMismatch}. Boxes may be drawn
            in the wrong place.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
