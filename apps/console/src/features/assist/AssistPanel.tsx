import { useEffect, useState } from "react"
import { Headset, PhoneIncoming, RefreshCw } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useGlasses } from "@/store/glasses"

const POLL_INTERVAL_MS = 2_000

function secondsUntil(iso: string): number {
  return Math.max(0, Math.round((new Date(iso).getTime() - Date.now()) / 1000))
}

/**
 * The tab B of the two-tab remote-assist demo: lists ringing requests and,
 * on Accept, joins that session's room as a microphone-only helper.
 *
 * Reuses `useGlasses`'s existing connect path (`acceptAssist`) rather than a
 * second `Room` -- the shared `VideoStage` already renders whatever this
 * store attaches, so this panel only needs the list and the button.
 */
export function AssistPanel() {
  const { pendingAssist, refreshAssist, acceptAssist, state, mode, session } = useGlasses()
  // Re-render once a second purely so "expires in Ns" counts down; the list
  // itself is refreshed on the slower POLL_INTERVAL_MS timer below.
  const [, setTick] = useState(0)

  useEffect(() => {
    void refreshAssist()
    const poll = setInterval(() => void refreshAssist(), POLL_INTERVAL_MS)
    const clock = setInterval(() => setTick((n) => n + 1), 1_000)
    return () => {
      clearInterval(poll)
      clearInterval(clock)
    }
  }, [refreshAssist])

  const helping = mode === "helper"

  return (
    <Card className="flex h-full min-h-0 flex-col overflow-hidden">
      <CardHeader className="flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle className="flex items-center gap-2 text-base">
          <Headset className={helping ? "size-4 text-primary" : "size-4 text-muted-foreground"} />
          Remote Assist
        </CardTitle>
        <Button size="sm" variant="ghost" onClick={() => void refreshAssist()}>
          <RefreshCw /> Refresh
        </Button>
      </CardHeader>

      <CardContent className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto">
        {helping && (
          <div className="space-y-1.5 rounded-lg border p-3">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium">Connected</span>
              <Badge variant={state === "helping" ? "default" : "secondary"}>{state}</Badge>
            </div>
            <p className="text-xs text-muted-foreground">
              Watching {session?.device_id ?? "—"} ({session?.session_id ?? "—"}). Microphone
              is live; camera is never published — the helper grant forbids it.
            </p>
          </div>
        )}

        <div className="space-y-2">
          <p className="text-sm font-medium">Ringing</p>
          {pendingAssist.length === 0 ? (
            <p className="text-xs text-muted-foreground">No pending requests.</p>
          ) : (
            <ul className="space-y-2">
              {pendingAssist.map((request) => (
                <li
                  key={request.request_id}
                  className="flex items-center justify-between gap-2 rounded-lg border p-2.5"
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{request.device_id}</p>
                    <p className="truncate text-xs text-muted-foreground">
                      {request.session_id} · expires in {secondsUntil(request.expires_at)}s
                    </p>
                  </div>
                  <Button
                    size="sm"
                    onClick={() => void acceptAssist(request.session_id)}
                    disabled={helping}
                  >
                    <PhoneIncoming /> Accept
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <p className="text-xs text-muted-foreground">
          A request expires unanswered after its TTL. Accepting joins the room as a
          microphone-only helper: video is subscribe-only, exactly like the Glasses tab's
          viewer mode.
        </p>
      </CardContent>
    </Card>
  )
}
