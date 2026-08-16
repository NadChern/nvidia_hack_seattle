import { useCallback, useEffect, useState } from "react"
import { RefreshCw, Search, Trash2 } from "lucide-react"
import { toast } from "sonner"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { get, post } from "@/lib/api"
import type { AnswerStatus, ObjectGallery, QueryAnswer } from "@/lib/contracts"

/** Compact "3m ago" / "2h ago" without pulling in a date library. */
function ago(iso: string): string {
  const seconds = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000))
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`
  return `${Math.round(seconds / 86400)}d ago`
}

/**
 * The status is the point, not the sentence.
 *
 * `visual_memory_memory_contract.protocol.QueryResponse` is explicit that a
 * conversational layer may shorten `spoken_answer` but **must** preserve
 * `answer_status`, the uncertainty and any invalidation. So the status is
 * rendered as loudly as the answer: a console showing only the sentence would
 * demonstrate the exact failure this system was built to prevent.
 */
const STATUS_STYLE: Record<AnswerStatus, { variant: "default" | "secondary" | "destructive" | "outline"; blurb: string }> =
  {
    confirmed: {
      variant: "default",
      blurb: "Seen there, and nothing since has invalidated it.",
    },
    stale: {
      variant: "outline",
      blurb: "Last confirmed a while ago. Still the best known location.",
    },
    in_transit: {
      variant: "secondary",
      blurb: "Picked up after it was last confirmed. No new location is known.",
    },
    unknown: {
      variant: "secondary",
      blurb: "Never confirmed anywhere. The honest answer is nothing.",
    },
    unavailable: {
      variant: "destructive",
      blurb: "A location exists but its evidence could not be loaded, so it is not claimed.",
    },
  }

export function MemoryPanel() {
  const [label, setLabel] = useState("keys")
  const [answer, setAnswer] = useState<QueryAnswer | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [asking, setAsking] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [gallery, setGallery] = useState<ObjectGallery | null>(null)

  const loadGallery = useCallback(async (signal?: AbortSignal) => {
    try {
      setGallery(await get<ObjectGallery>("memory", "/v1/objects", signal))
    } catch {
      // A transient gallery fetch failure must not blank the panel; the next
      // poll recovers, and the query path reports its own errors.
    }
  }, [])

  // Poll so objects registered elsewhere (Enroll tab, voice) appear here, and
  // a clear empties the table without a manual refresh.
  useEffect(() => {
    const controller = new AbortController()
    void loadGallery(controller.signal)
    const timer = setInterval(() => void loadGallery(), 5000)
    return () => {
      controller.abort()
      clearInterval(timer)
    }
  }, [loadGallery])

  const clearMemory = async () => {
    setClearing(true)
    try {
      await post("memory", "/v1/maintenance/reset")
      setAnswer(null)
      setError(null)
      setConfirmOpen(false)
      await loadGallery()
      toast.success("Memory cleared. Registry, galleries, and placements are empty.")
    } catch (caught) {
      toast.error(caught instanceof Error ? caught.message : "Could not clear memory.")
    } finally {
      setClearing(false)
    }
  }

  // Memory resolves by label, not by free text (the agent does the NL parsing).
  const locate = async (target: string) => {
    const wanted = target.trim()
    if (wanted === "") return
    setLabel(wanted)
    setAsking(true)
    setError(null)
    try {
      setAnswer(await post<QueryAnswer>("memory", "/v1/query", { label: wanted }))
    } catch (caught) {
      setAnswer(null)
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setAsking(false)
    }
  }

  const viewCount = (objectId: string) =>
    gallery?.views.filter((view) => view.object_id === objectId).length ?? 0

  const style = answer ? STATUS_STYLE[answer.answer_status] : null

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base">Ask memory</CardTitle>
        <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
          <DialogTrigger asChild>
            <Button
              size="sm"
              variant="ghost"
              className="text-destructive hover:text-destructive"
            >
              <Trash2 /> Clear memory
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Clear all memory?</DialogTitle>
              <DialogDescription>
                Permanently deletes every registered object, reference gallery,
                observation, and placement, and purges stored evidence. This cannot
                be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <DialogClose asChild>
                <Button variant="outline" size="sm">
                  Cancel
                </Button>
              </DialogClose>
              <Button
                variant="destructive"
                size="sm"
                disabled={clearing}
                onClick={() => void clearMemory()}
              >
                {clearing ? "Clearing…" : "Clear memory"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col gap-3">
        <div className="rounded-lg border">
          <div className="flex items-center justify-between border-b px-3 py-2">
            <span className="text-xs font-medium">
              Registered objects ({gallery?.objects.length ?? 0})
            </span>
            <Button
              size="sm"
              variant="ghost"
              className="h-6 px-2"
              onClick={() => void loadGallery()}
              aria-label="Refresh registered objects"
            >
              <RefreshCw />
            </Button>
          </div>
          <div className="max-h-44 overflow-y-auto">
            {gallery && gallery.objects.length > 0 ? (
              <table className="w-full text-sm">
                <thead className="text-xs text-muted-foreground">
                  <tr className="border-b">
                    <th className="px-3 py-1.5 text-left font-medium">Label</th>
                    <th className="px-3 py-1.5 text-right font-medium">Views</th>
                    <th className="px-3 py-1.5 text-right font-medium">Registered</th>
                  </tr>
                </thead>
                <tbody>
                  {gallery.objects.map((object) => (
                    <tr
                      key={object.object_id}
                      className="cursor-pointer border-b last:border-0 hover:bg-muted/50"
                      onClick={() => void locate(object.label)}
                      title={`Locate "${object.label}"`}
                    >
                      <td className="px-3 py-1.5">{object.label}</td>
                      <td className="px-3 py-1.5 text-right tabular-nums">
                        {viewCount(object.object_id)}
                      </td>
                      <td className="px-3 py-1.5 text-right text-muted-foreground">
                        {object.created_at ? ago(object.created_at) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="px-3 py-4 text-xs text-muted-foreground">
                Nothing registered yet. Enroll an object, then it appears here.
              </p>
            )}
          </div>
        </div>

        <form
          className="flex gap-2"
          onSubmit={(event) => {
            event.preventDefault()
            void locate(label)
          }}
        >
          <input
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            placeholder="label, e.g. keys"
            className="flex-1 rounded-md border bg-transparent px-3 py-1.5 text-sm outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
          />
          <Button size="sm" type="submit" disabled={asking || label.trim() === ""}>
            <Search /> Locate
          </Button>
        </form>

        {error && (
          <p className="text-xs text-destructive">
            {error.startsWith("memory 404") ? "No memory service on /api/memory." : error}
          </p>
        )}

        {answer && style && (
          <div className="space-y-3 rounded-lg border p-3">
            <div className="flex items-center gap-2">
              <Badge variant={style.variant}>{answer.answer_status.replace("_", " ")}</Badge>
              {answer.object_label && (
                <span className="text-xs text-muted-foreground">{answer.object_label}</span>
              )}
            </div>

            <p className="text-sm leading-relaxed">{answer.spoken_answer}</p>

            <p className="text-xs text-muted-foreground">{style.blurb}</p>

            {answer.evidence && (
              <a
                href={answer.evidence.url}
                target="_blank"
                rel="noreferrer"
                className="inline-block text-xs text-primary underline underline-offset-2"
              >
                evidence ({answer.evidence.media_type})
              </a>
            )}
          </div>
        )}

        <p className="mt-auto text-xs text-muted-foreground">
          The status travels with the sentence on purpose. Shortening the answer is
          allowed; dropping <span className="text-foreground">why it is uncertain</span>{" "}
          turns a true answer into a false one.
        </p>
      </CardContent>
    </Card>
  )
}
