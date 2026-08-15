import type { ReactNode } from "react"

import { TranscriptContext } from "@/hooks/transcript-context"
import { useTranscripts } from "@/hooks/useTranscripts"

/**
 * Own the one Gateway HUD-event subscription for the selected session.
 *
 * The Agent owns Speech STT and pushes transcripts plus guarded replies to the
 * Gateway. This provider sits above the tabs so changing tabs never tears down
 * the event socket. Panels consume the same transcript/reply history.
 */
export function TranscriptProvider({
  sessionId,
  children,
}: {
  sessionId: string | null
  children: ReactNode
}) {
  const stream = useTranscripts(sessionId)
  return <TranscriptContext.Provider value={stream}>{children}</TranscriptContext.Provider>
}
