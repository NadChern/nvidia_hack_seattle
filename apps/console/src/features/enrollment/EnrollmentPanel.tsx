import { useEffect, useMemo, useState, type PointerEvent as ReactPointerEvent } from "react"
import { useQuery } from "@tanstack/react-query"
import { Camera, Check, Crop, RotateCcw, Trash2, ZoomIn } from "lucide-react"
import { toast } from "sonner"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { delChecked, get, post } from "@/lib/api"
import type { EnrolledObject, EnrollmentProgress, VisionStatus } from "@/lib/contracts"
import { capturePreviewJpeg, useGlasses } from "@/store/glasses"

interface Selection {
  x0: number
  y0: number
  x1: number
  y1: number
}

interface LocalView {
  blob: Blob
  url: string
}

const DEFAULT_SELECTION: Selection = { x0: 0.2, y0: 0.15, x1: 0.8, y1: 0.85 }

function revoke(view: LocalView | null): void {
  if (view) URL.revokeObjectURL(view.url)
}

async function cropJpeg(source: Blob, selection: Selection): Promise<Blob> {
  const image = await createImageBitmap(source)
  try {
    const x = Math.round(selection.x0 * image.width)
    const y = Math.round(selection.y0 * image.height)
    const width = Math.max(1, Math.round((selection.x1 - selection.x0) * image.width))
    const height = Math.max(1, Math.round((selection.y1 - selection.y0) * image.height))
    if (width < 64 || height < 64) throw new Error("Draw a larger crop around the object.")
    const canvas = document.createElement("canvas")
    canvas.width = width
    canvas.height = height
    const context = canvas.getContext("2d")
    if (!context) throw new Error("This browser cannot crop the captured frame.")
    context.drawImage(image, x, y, width, height, 0, 0, width, height)
    return await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error("Could not encode the crop."))),
        "image/jpeg",
        0.95,
      )
    })
  } finally {
    image.close()
  }
}

async function blobBase64(blob: Blob): Promise<string> {
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the crop."))
    reader.onload = () => {
      const value = String(reader.result)
      const comma = value.indexOf(",")
      if (comma < 0) reject(new Error("Could not encode the crop."))
      else resolve(value.slice(comma + 1))
    }
    reader.readAsDataURL(blob)
  })
}

export function EnrollmentPanel() {
  const sessionId = useGlasses((glasses) => glasses.session?.session_id ?? null)
  const [label, setLabel] = useState("")
  const [frozen, setFrozen] = useState<LocalView | null>(null)
  const [selection, setSelection] = useState<Selection>(DEFAULT_SELECTION)
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null)
  const [views, setViews] = useState<LocalView[]>([])
  const [previewIndex, setPreviewIndex] = useState<number | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<EnrollmentProgress | null>(null)
  const [error, setError] = useState<string | null>(null)

  const vision = useQuery({
    queryKey: ["vision", "status"],
    queryFn: ({ signal }) => get<VisionStatus>("vision", "/v1/status", signal),
    refetchInterval: 10_000,
    retry: false,
  })
  const labels = useMemo(() => vision.data?.config.registration_labels ?? [], [vision.data])

  useEffect(() => {
    if (!label && labels.length > 0) setLabel(labels[0] ?? "")
  }, [label, labels])

  const reset = () => {
    revoke(frozen)
    views.forEach((view) => revoke(view))
    setFrozen(null)
    setViews([])
    setSelection(DEFAULT_SELECTION)
    setPreviewIndex(null)
    setResult(null)
    setError(null)
  }

  const freezePov = async () => {
    setError(null)
    try {
      const blob = await capturePreviewJpeg()
      revoke(frozen)
      setFrozen({ blob, url: URL.createObjectURL(blob) })
      setSelection(DEFAULT_SELECTION)
      setResult(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    }
  }

  const point = (event: ReactPointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    return {
      x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
    }
  }

  const beginSelection = (event: ReactPointerEvent<HTMLDivElement>) => {
    const start = point(event)
    event.currentTarget.setPointerCapture(event.pointerId)
    setDragStart(start)
    setSelection({ x0: start.x, y0: start.y, x1: start.x, y1: start.y })
  }

  const moveSelection = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragStart) return
    const current = point(event)
    setSelection({
      x0: Math.min(dragStart.x, current.x),
      y0: Math.min(dragStart.y, current.y),
      x1: Math.max(dragStart.x, current.x),
      y1: Math.max(dragStart.y, current.y),
    })
  }

  const addView = async () => {
    if (!frozen || views.length >= 8) return
    setError(null)
    try {
      const blob = await cropJpeg(frozen.blob, selection)
      setViews((current) => [...current, { blob, url: URL.createObjectURL(blob) }])
      toast.success(`View ${views.length + 1} staged. Rotate the object and freeze another angle.`)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    }
  }

  const removeView = (index: number) => {
    setViews((current) => {
      revoke(current[index] ?? null)
      return current.filter((_view, candidate) => candidate !== index)
    })
    if (previewIndex === index) setPreviewIndex(null)
  }

  const confirm = async () => {
    if (!label || views.length < 2 || submitting) return
    setSubmitting(true)
    setError(null)
    setResult(null)
    let objectId: string | null = null
    try {
      const created = await post<EnrolledObject>("vision", "/v1/objects", {
        label,
        idempotency_key: `console/manual/${sessionId ?? "detached"}/${label}/${Date.now()}`,
      })
      objectId = created.object_id
      const encoded = await Promise.all(views.map((view) => blobBase64(view.blob)))
      const completed = await post<EnrollmentProgress>(
        "vision",
        `/v1/objects/${created.object_id}/manual`,
        { views_base64: encoded },
      )
      setResult(completed)
      if (completed.state !== "succeeded") {
        throw new Error(completed.message ?? "The selected crops did not pass image quality.")
      }
      toast.success(`${label} registration stored from ${completed.selected_views} approved views.`)
    } catch (caught) {
      if (objectId) await delChecked("vision", `/v1/objects/${objectId}`).catch(() => undefined)
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setSubmitting(false)
    }
  }

  const previewUrl = previewIndex === null ? null : (views[previewIndex]?.url ?? null)
  const selectionStyle = {
    left: `${selection.x0 * 100}%`,
    top: `${selection.y0 * 100}%`,
    width: `${(selection.x1 - selection.x0) * 100}%`,
    height: `${(selection.y1 - selection.y0) * 100}%`,
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
            disabled={submitting || result?.state === "succeeded"}
            className="h-9 w-full rounded-md border bg-background px-3 text-sm"
          >
            {labels.length === 0 ? <option value="">No configured labels</option> : null}
            {labels.map((item) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
        </div>

        <div className="flex gap-2">
          <Button onClick={() => void freezePov()} disabled={!sessionId || !label || submitting}>
            <Camera /> Freeze current POV
          </Button>
          <Button variant="outline" onClick={reset} disabled={submitting}>
            <RotateCcw /> Clear
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          Freeze a frame, drag a box tightly around the physical object, and add 2–8 distinct angles.
          Nothing enters the identity gallery until you press Confirm and register.
        </p>

        {frozen ? (
          <div className="space-y-2">
            <div
              role="application"
              aria-label="Draw registration crop"
              className="relative touch-none select-none overflow-hidden rounded-lg border bg-black"
              onPointerDown={beginSelection}
              onPointerMove={moveSelection}
              onPointerUp={() => setDragStart(null)}
              onPointerCancel={() => setDragStart(null)}
            >
              <img src={frozen.url} alt="Frozen glasses POV" className="block h-auto w-full" draggable={false} />
              <div className="pointer-events-none absolute border-2 border-cyan-400 bg-cyan-300/10 shadow-[0_0_0_9999px_rgba(0,0,0,0.4)]" style={selectionStyle} />
            </div>
            <Button onClick={() => void addView()} disabled={views.length >= 8}>
              <Crop /> Add selected crop
            </Button>
          </div>
        ) : null}

        {views.length > 0 ? (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs font-medium">
              <span>Pending operator-approved views</span>
              <span>{views.length}/8</span>
            </div>
            <div className="flex gap-2 overflow-x-auto pb-1">
              {views.map((view, index) => (
                <div key={view.url} className="relative shrink-0">
                  <button
                    type="button"
                    aria-label={`Enlarge pending reference ${index + 1}`}
                    onClick={() => setPreviewIndex(index)}
                    className="group overflow-hidden rounded-md border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <img src={view.url} alt={`Pending reference ${index + 1}`} className="size-24 object-cover" />
                    <span className="absolute inset-x-0 bottom-0 flex items-center justify-center gap-1 bg-black/65 py-1 text-[10px] text-white opacity-0 group-hover:opacity-100">
                      <ZoomIn className="size-3" /> Preview
                    </span>
                  </button>
                  <button
                    type="button"
                    aria-label={`Remove pending reference ${index + 1}`}
                    onClick={() => removeView(index)}
                    className="absolute right-1 top-1 rounded-full bg-black/75 p-1 text-white"
                  >
                    <Trash2 className="size-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <Button onClick={() => void confirm()} disabled={views.length < 2 || submitting || result?.state === "succeeded"}>
          <Check /> {submitting ? "Validating…" : "Confirm and register"}
        </Button>

        {result ? (
          <div className="rounded-lg border p-3 text-xs">
            <strong className="capitalize">{result.state}</strong>
            <span className="ml-2 text-muted-foreground">
              {result.selected_views} C-RADIO views stored
            </span>
          </div>
        ) : null}

        {error ? <p className="text-xs text-destructive">{error}</p> : null}

        <Dialog open={previewUrl !== null} onOpenChange={(open) => { if (!open) setPreviewIndex(null) }}>
          <DialogContent className="sm:max-w-4xl">
            <DialogHeader>
              <DialogTitle>Pending reference {previewIndex === null ? "" : previewIndex + 1}</DialogTitle>
              <DialogDescription>
                Keep it only when the target itself fills the crop and is sharp and recognizable.
              </DialogDescription>
            </DialogHeader>
            {previewUrl ? (
              <div className="flex max-h-[75vh] min-h-64 items-center justify-center rounded-lg bg-muted/40 p-3">
                <img src={previewUrl} alt="Enlarged pending reference" className="max-h-[70vh] max-w-full object-contain" />
              </div>
            ) : null}
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  )
}
