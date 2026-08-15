import { useState } from "react"
import { Search } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { post } from "@/lib/api"
import type { AnswerStatus, QueryAnswer } from "@/lib/contracts"

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
  const [question, setQuestion] = useState("where are my keys?")
  const [answer, setAnswer] = useState<QueryAnswer | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [asking, setAsking] = useState(false)

  const ask = async () => {
    setAsking(true)
    setError(null)
    try {
      setAnswer(await post<QueryAnswer>("memory", "/v1/query", { text: question }))
    } catch (caught) {
      setAnswer(null)
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setAsking(false)
    }
  }

  const style = answer ? STATUS_STYLE[answer.answer_status] : null

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="space-y-0">
        <CardTitle className="text-base">Ask memory</CardTitle>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col gap-3">
        <form
          className="flex gap-2"
          onSubmit={(event) => {
            event.preventDefault()
            void ask()
          }}
        >
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="where are my keys?"
            className="flex-1 rounded-md border bg-transparent px-3 py-1.5 text-sm outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
          />
          <Button size="sm" type="submit" disabled={asking || question.trim() === ""}>
            <Search /> Ask
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
