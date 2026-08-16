import { useCallback, useEffect, useRef } from "react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useGlasses } from "@/store/glasses"

export function VideoStage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const attachPreview = useGlasses((state) => state.attachPreview)
  const mediaLive = useGlasses(
    (state) =>
      state.state === "publishing" || state.state === "viewing" || state.state === "helping",
  )

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
        <div>
          <CardTitle className="text-base">What the glasses see</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">Raw first-person video</p>
        </div>
        <Badge variant={mediaLive ? "default" : "secondary"}>
          {mediaLive ? "live" : "waiting"}
        </Badge>
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
          {!mediaLive && (
            <div className="absolute inset-0 grid place-items-center px-6 text-center text-sm text-muted-foreground">
              Publish the virtual glasses or watch a live device to see its camera.
            </div>
          )}
        </div>

        <p className="text-xs text-muted-foreground">
          Cosmos analyzes short frame windows rather than producing per-frame boxes. Open
          <span className="font-medium text-foreground"> Vision </span>
          to watch placement, identity, and Memory-write receipts.
        </p>
      </CardContent>
    </Card>
  )
}
