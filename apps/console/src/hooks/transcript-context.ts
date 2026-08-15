import { createContext, useContext } from "react"

import type { TranscriptStream } from "@/hooks/useTranscripts"

export const TranscriptContext = createContext<TranscriptStream | null>(null)

export function useTranscriptStream(): TranscriptStream {
  const stream = useContext(TranscriptContext)
  if (stream === null) {
    throw new Error("useTranscriptStream must be used inside TranscriptProvider")
  }
  return stream
}
