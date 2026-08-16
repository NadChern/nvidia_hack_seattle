import { useQuery } from "@tanstack/react-query"
import type { ReactNode } from "react"
import { BrainCircuit, Database, Fingerprint, Images } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { get } from "@/lib/api"
import type { PipelineOutcome, VisionEvent, VisionStatus } from "@/lib/contracts"

type BadgeVariant = "default" | "secondary" | "destructive" | "outline"

const OUTCOME: Record<
  PipelineOutcome,
  { variant: BadgeVariant; label: string; explanation: string }
> = {
  written: {
    variant: "default",
    label: "memory written",
    explanation: "Cosmos placement and personal identity passed every write gate.",
  },
  skipped_no_identity: {
    variant: "destructive",
    label: "identity skipped",
    explanation: "The crop did not match a registered personal object strongly enough.",
  },
  suppressed_by_policy: {
    variant: "outline",
    label: "motion suppressed",
    explanation: "Motion events are diagnostic-only in the placed-only demo policy.",
  },
  deduped: {
    variant: "secondary",
    label: "duplicate",
    explanation: "A recent identical object/action was already recorded.",
  },
}

function clockTime(iso: string): string {
  const at = new Date(iso)
  return Number.isNaN(at.getTime())
    ? "--:--:--"
    : at.toLocaleTimeString(undefined, { hour12: false })
}

function compactModel(model: string): string {
  return model.split("/").at(-1) ?? model
}

function shortId(value: string | null): string | null {
  if (!value) return null
  return value.length > 18 ? `${value.slice(0, 10)}…${value.slice(-5)}` : value
}

function Receipt({
  icon,
  stage,
  value,
  detail,
}: {
  icon: ReactNode
  stage: string
  value: string
  detail: string
}) {
  return (
    <div className="rounded-lg border bg-muted/15 p-3">
      <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {icon}
        {stage}
      </div>
      <div className="tnum text-xl font-semibold">{value}</div>
      <div className="mt-1 text-[11px] leading-tight text-muted-foreground">{detail}</div>
    </div>
  )
}

function Stat({ label, value, warn = false }: { label: string; value: number; warn?: boolean }) {
  return (
    <div className="rounded-md border px-2.5 py-2">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className={`tnum text-base font-semibold ${warn ? "text-destructive" : ""}`}>
        {value}
      </div>
    </div>
  )
}

export function VisionPanel() {
  const status = useQuery({
    queryKey: ["vision", "status"],
    queryFn: ({ signal }) => get<VisionStatus>("vision", "/v1/status", signal),
    refetchInterval: 1000,
    retry: false,
  })

  const events = useQuery({
    queryKey: ["vision", "events"],
    queryFn: ({ signal }) => get<{ events: VisionEvent[] }>("vision", "/v1/events", signal),
    refetchInterval: 1000,
    retry: false,
  })

  const data = status.data
  const recent = [...(events.data?.events ?? [])].reverse()
  const embedder = data?.identity.embedder

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex-row items-center justify-between gap-3 space-y-0">
        <div>
          <CardTitle className="text-base">Vision pipeline receipts</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            Window reasoning → personal identity → structured Memory
          </p>
        </div>
        {status.isError ? (
          <Badge variant="destructive">unreachable</Badge>
        ) : data ? (
          <Badge variant={data.ready ? "default" : "secondary"}>
            {data.ready ? "ready" : (data.not_ready_reason ?? "starting")}
          </Badge>
        ) : (
          <Skeleton className="h-5 w-16" />
        )}
      </CardHeader>

      <CardContent className="min-h-0 flex-1">
        <ScrollArea className="h-full pr-3">
          {status.isError && (
            <p className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-xs">
              Vision status is unavailable. The raw glasses preview can remain live even when
              reasoning is not ready.
            </p>
          )}

          {data && (
            <div className="space-y-4">
              <div className="flex flex-wrap gap-1.5 text-xs">
                <Badge variant="outline">{compactModel(data.reasoner.model)}</Badge>
                <Badge variant="outline">
                  {String(embedder?.identity_embedder ?? data.config.identity_kind)}
                </Badge>
                <Badge variant="outline">
                  {data.reasoner.window_seconds}s window · every {data.reasoner.interval_seconds}s
                </Badge>
                <Badge variant={data.reasoner.promote_motion_events ? "destructive" : "secondary"}>
                  {data.reasoner.promote_motion_events ? "motion promotion on" : "placed-only writes"}
                </Badge>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <Receipt
                  icon={<Images className="size-3.5" />}
                  stage="Registered gallery"
                  value={`${data.identity.gallery.gallery_objects} objects`}
                  detail={`${data.identity.gallery.gallery_views} C-RADIO reference views`}
                />
                <Receipt
                  icon={<BrainCircuit className="size-3.5" />}
                  stage="Cosmos windows"
                  value={String(data.metrics.windows_analyzed)}
                  detail={`${data.metrics.events_detected} memory-worthy events reported`}
                />
                <Receipt
                  icon={<Fingerprint className="size-3.5" />}
                  stage="Personal identity"
                  value={String(data.metrics.identity_matched)}
                  detail={`${data.metrics.identity_skipped} events rejected by the identity gate`}
                />
                <Receipt
                  icon={<Database className="size-3.5" />}
                  stage="Durable Memory"
                  value={String(data.metrics.observations_written)}
                  detail="confirmed observations written with evidence"
                />
              </div>

              <div className="grid grid-cols-4 gap-2">
                <Stat label="frames" value={data.metrics.frames_processed} />
                <Stat label="pending" value={data.analysis.pending} warn={data.analysis.pending > 1} />
                <Stat label="dropped" value={data.analysis.dropped} warn={data.analysis.dropped > 0} />
                <Stat label="failed" value={data.analysis.failed} warn={data.analysis.failed > 0} />
              </div>

              <div className="rounded-lg border p-3 text-xs text-muted-foreground">
                <span className="font-medium text-foreground">Registration:</span>{" "}
                {data.registration.succeeded} succeeded, {data.registration.failed} failed,
                {" "}{data.registration.active} active. Identity writes require at least{" "}
                {(data.identity.min_cosine * 100).toFixed(0)}% cosine similarity.
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between">
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    Cosmos and identity activity
                  </div>
                  <div className="text-[10px] text-muted-foreground">newest first</div>
                </div>
                <div className="overflow-hidden rounded-lg border">
                  {events.isError ? (
                    <p className="p-3 text-xs text-destructive">Activity feed unavailable.</p>
                  ) : recent.length === 0 ? (
                    <p className="p-4 text-xs leading-relaxed text-muted-foreground">
                      No placement receipts yet. Register an object, place it in view, and leave
                      it at rest through a complete Cosmos window.
                    </p>
                  ) : (
                    <div className="divide-y">
                      {recent.map((event, index) => {
                        const outcome = OUTCOME[event.outcome]
                        const objectId = shortId(event.object_id)
                        return (
                          <div key={`${event.at}-${event.label}-${index}`} className="p-3">
                            <div className="flex flex-wrap items-center gap-2 text-xs">
                              <span className="tnum text-muted-foreground" title={event.at}>
                                {clockTime(event.at)}
                              </span>
                              <Badge variant="outline" className="capitalize">
                                {event.action.replaceAll("_", " ")}
                              </Badge>
                              <span className="font-semibold">{event.label}</span>
                              {event.score !== null && (
                                <Badge variant="secondary">
                                  personal {(event.score * 100).toFixed(0)}%
                                </Badge>
                              )}
                              <Badge variant={outcome.variant} className="ml-auto">
                                {outcome.label}
                              </Badge>
                            </div>
                            <div className="mt-1.5 flex items-center justify-between gap-3 text-[11px] text-muted-foreground">
                              <span>{outcome.explanation}</span>
                              {objectId && <code className="shrink-0">{objectId}</code>}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
