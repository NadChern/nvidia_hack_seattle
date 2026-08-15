import { useQuery } from "@tanstack/react-query"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { get } from "@/lib/api"
import type { PipelineOutcome, VisionEvent, VisionStatus } from "@/lib/contracts"

const OUTCOME_VARIANT: Record<PipelineOutcome, "default" | "secondary" | "destructive" | "outline"> =
  {
    confirmed: "default",
    rejected: "destructive",
    unverified: "outline",
    not_promoted: "secondary",
  }

/** Local wall-clock, to the second -- what you compare against a stopwatch. */
function clockTime(iso: string): string {
  const at = new Date(iso)
  return Number.isNaN(at.getTime()) ? "--:--:--" : at.toLocaleTimeString(undefined, { hour12: false })
}


function Stat({
  label,
  value,
  tone,
  hint,
}: {
  label: string
  value: string | number
  tone?: "normal" | "warn" | "bad"
  hint?: string
}) {
  const color =
    tone === "bad" ? "text-destructive" : tone === "warn" ? "text-amber-400" : "text-foreground"
  return (
    <div className="rounded-lg border p-2.5">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className={`tnum text-lg font-semibold ${color}`}>{value}</div>
      {hint && <div className="mt-0.5 text-[10px] leading-tight text-muted-foreground">{hint}</div>}
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

  // The rate the thresholds were built from, against the rate frames are
  // really arriving. Every stability threshold is a frame count derived from
  // the configured value, so a disagreement silently rescales all of them --
  // this is the one place a person can see it.
  const configured = data?.frame_rate.configured_fps
  const observed = data?.frame_rate.observed_fps
  const rateDisagrees =
    configured !== undefined &&
    observed !== null &&
    observed !== undefined &&
    Math.abs(observed - configured) / configured > 0.25

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle className="text-base">Vision</CardTitle>
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

      <CardContent className="flex flex-1 flex-col gap-3">
        {status.isError && (
          <p className="text-xs text-muted-foreground">
            No vision worker on <code>/api/vision</code>. Everything else on this page
            still works.
          </p>
        )}

        {data && (
          <>
            <div className="flex flex-wrap gap-1.5 text-xs">
              <Badge variant="outline">{data.config.detector_kind}</Badge>
              <Badge variant="outline">{data.verifier}</Badge>
              <Badge variant="outline">depth: {data.config.depth_kind}</Badge>
            </div>

            {data.config.detection_labels.length > 0 && (
              <p className="text-xs text-muted-foreground">
                looking for{" "}
                <span className="text-foreground">
                  {data.config.detection_labels.join(", ")}
                </span>
              </p>
            )}

            <div className="grid grid-cols-3 gap-2">
              <Stat label="frames" value={data.metrics.frames_processed} />
              <Stat
                label="fps"
                value={observed === null || observed === undefined ? "—" : observed.toFixed(1)}
                tone={rateDisagrees ? "bad" : "normal"}
                hint={rateDisagrees ? `configured ${configured}` : undefined}
              />
              <Stat label="confirmed" value={data.metrics.candidates_confirmed} />
              <Stat
                label="questions"
                value={data.metrics.vanishings_questioned}
                hint="objects that vanished at rest"
              />
              <Stat
                label="pending"
                value={data.verification.pending}
                tone={data.verification.pending > 0 ? "warn" : "normal"}
                hint="awaiting a verifier"
              />
              <Stat
                label="dropped"
                value={data.verification.dropped}
                tone={data.verification.dropped > 0 ? "bad" : "normal"}
                hint="must be zero"
              />
            </div>

            {rateDisagrees && (
              <p className="text-xs text-destructive">
                The relay is delivering {observed?.toFixed(1)} fps but thresholds were
                built for {configured}. Every dwell and settle window means a different
                duration than intended.
              </p>
            )}

            {data.verification.dropped > 0 && (
              <p className="text-xs text-destructive">
                {data.verification.dropped} candidate
                {data.verification.dropped === 1 ? "" : "s"} discarded because
                verification could not keep up. Each one is a real event that was seen
                and never recorded.
              </p>
            )}
          </>
        )}

        <div className="flex-1">
          <div className="mb-1.5 text-[10px] uppercase tracking-wide text-muted-foreground">
            Activity
          </div>
          <ScrollArea className="h-48 rounded-lg border">
            <div className="divide-y">
              {(events.data?.events ?? []).length === 0 && (
                <p className="p-3 text-xs text-muted-foreground">
                  Nothing yet. Candidates appear here as the pipeline decides them.
                </p>
              )}
              {[...(events.data?.events ?? [])].reverse().map((event, index) => (
                <div
                  key={`${event.at}-${event.track_id}-${index}`}
                  className="flex items-center gap-2 px-3 py-2 text-xs"
                >
                  {/* When it happened, first. An activity log without times
                      cannot answer "did that fire when I put the keys down?",
                      which is the only question anyone asks of it. */}
                  <span className="tnum shrink-0 text-muted-foreground" title={event.at}>
                    {clockTime(event.at)}
                  </span>
                  <Badge variant={OUTCOME_VARIANT[event.outcome]} className="shrink-0">
                    {event.action}
                  </Badge>
                  <span className="truncate font-medium">{event.label}</span>
                  <span className="tnum ml-auto shrink-0 text-muted-foreground">
                    {event.reason_code}
                  </span>
                </div>
              ))}
            </div>
          </ScrollArea>
        </div>
      </CardContent>
    </Card>
  )
}
