import { useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Camera, Check, RotateCcw, Trash2 } from "lucide-react"
import { toast } from "sonner"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { delChecked, get, getBlob, post } from "@/lib/api"
import type {
  EnrolledObject,
  EnrollmentProgress,
  ObjectGallery,
  VisionStatus,
} from "@/lib/contracts"
import { useGlasses } from "@/store/glasses"

function terminal(state: EnrollmentProgress["state"] | undefined): boolean {
  return state === "succeeded" || state === "failed"
}

export function EnrollmentPanel() {
  const sessionId = useGlasses((glasses) => glasses.session?.session_id ?? null)
  const [label, setLabel] = useState("")
  const [objectId, setObjectId] = useState<string | null>(null)
  const [initialProgress, setInitialProgress] = useState<EnrollmentProgress | null>(null)
  const [thumbnailUrls, setThumbnailUrls] = useState<string[]>([])
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const vision = useQuery({
    queryKey: ["vision", "status"],
    queryFn: ({ signal }) => get<VisionStatus>("vision", "/v1/status", signal),
    refetchInterval: 10_000,
    retry: false,
  })
  const labels = useMemo(() => vision.data?.config.detection_labels ?? [], [vision.data])

  useEffect(() => {
    if (!label && labels.length > 0) setLabel(labels[0] ?? "")
  }, [label, labels])

  const progressQuery = useQuery({
    queryKey: ["vision", "enrollment", objectId],
    enabled: objectId !== null,
    queryFn: ({ signal }) =>
      get<EnrollmentProgress>("vision", `/v1/objects/${objectId}/status`, signal),
    refetchInterval: (query) => (terminal(query.state.data?.state) ? false : 500),
    retry: false,
  })
  const progress = progressQuery.data ?? initialProgress

  useEffect(() => {
    if (progress?.state !== "succeeded" || !objectId || thumbnailUrls.length > 0) return
    const controller = new AbortController()
    const urls: string[] = []
    void (async () => {
      try {
        const gallery = await get<ObjectGallery>("vision", "/v1/objects", controller.signal)
        const views = gallery.views.filter((view) => view.object_id === objectId)
        for (const view of views) {
          const blob = await getBlob("memory", view.crop_reference, controller.signal)
          urls.push(URL.createObjectURL(blob))
        }
        setThumbnailUrls(urls)
        toast.success("Reference views are ready to review.")
      } catch (caught) {
        urls.forEach((url) => URL.revokeObjectURL(url))
        if (!controller.signal.aborted) {
          setError(caught instanceof Error ? caught.message : String(caught))
        }
      }
    })()
    return () => controller.abort()
  }, [objectId, progress?.state, thumbnailUrls.length])

  useEffect(() => {
    return () => thumbnailUrls.forEach((url) => URL.revokeObjectURL(url))
  }, [thumbnailUrls])

  const reset = () => {
    thumbnailUrls.forEach((url) => URL.revokeObjectURL(url))
    setThumbnailUrls([])
    setObjectId(null)
    setInitialProgress(null)
    setError(null)
  }

  const start = async () => {
    if (!label || !sessionId) return
    setStarting(true)
    setError(null)
    reset()
    try {
      const created = await post<EnrolledObject>("vision", "/v1/objects", {
        label,
        idempotency_key: `console/${sessionId}/${label}/${Date.now()}`,
      })
      const capture = await post<EnrollmentProgress>(
        "vision",
        `/v1/objects/${created.object_id}/capture`,
        {},
      )
      setObjectId(created.object_id)
      setInitialProgress(capture)
      toast.info(`Rotate the ${label} slowly while Vision captures it.`)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setStarting(false)
    }
  }

  const discard = async () => {
    if (!objectId) return
    try {
      await delChecked("memory", `/v1/objects/${objectId}`)
      toast.success("Registration discarded.")
      reset()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    }
  }

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle className="text-base">Register an object</CardTitle>
        <Badge variant={sessionId ? "secondary" : "outline"}>
          {sessionId ? "video session active" : "publish first"}
        </Badge>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-4 overflow-auto">
        <div className="space-y-2">
          <label htmlFor="enrollment-label" className="text-xs font-medium">
            Trackable label
          </label>
          <select
            id="enrollment-label"
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            disabled={starting || Boolean(progress)}
            className="h-9 w-full rounded-md border bg-background px-3 text-sm"
          >
            {labels.length === 0 ? <option value="">No configured labels</option> : null}
            {labels.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>

        <Button
          onClick={() => void start()}
          disabled={!sessionId || !label || starting || Boolean(progress)}
        >
          <Camera /> {starting ? "Starting…" : "Start registration"}
        </Button>

        {progress ? (
          <div className="space-y-3 rounded-lg border p-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium capitalize">{progress.state}</span>
              <Badge variant={progress.state === "failed" ? "destructive" : "secondary"}>
                {progress.selected_views} views
              </Badge>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center text-xs text-muted-foreground">
              <div><strong className="block text-foreground">{progress.frames_total}</strong>frames</div>
              <div><strong className="block text-foreground">{progress.quality_passed}</strong>quality</div>
              <div><strong className="block text-foreground">{progress.selected_views}</strong>selected</div>
            </div>
            {progress.state === "capturing" ? (
              <p className="text-xs">Rotate it slowly and keep the whole object in view.</p>
            ) : null}
            {progress.message ? <p className="text-xs text-muted-foreground">{progress.message}</p> : null}
          </div>
        ) : null}

        {thumbnailUrls.length > 0 ? (
          <div className="space-y-2">
            <div className="text-xs font-medium">Selected reference views</div>
            <div className="flex gap-2 overflow-x-auto">
              {thumbnailUrls.map((url, index) => (
                <img
                  key={url}
                  src={url}
                  alt={`Selected reference ${index + 1}`}
                  className="size-24 rounded-md border object-cover"
                />
              ))}
            </div>
          </div>
        ) : null}

        {progress?.state === "succeeded" ? (
          <div className="flex gap-2">
            <Button
              onClick={() => {
                toast.success(`${label} registration confirmed.`)
                reset()
              }}
            >
              <Check /> Confirm
            </Button>
            <Button variant="destructive" onClick={() => void discard()}>
              <Trash2 /> Discard
            </Button>
          </div>
        ) : null}

        {progress?.state === "failed" ? (
          <Button variant="outline" onClick={reset}>
            <RotateCcw /> Retry
          </Button>
        ) : null}

        {error || progressQuery.error ? (
          <p className="text-xs text-destructive">
            {error ?? (progressQuery.error instanceof Error ? progressQuery.error.message : "Status unavailable")}
          </p>
        ) : null}
      </CardContent>
    </Card>
  )
}
