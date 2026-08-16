import { QueryClient, QueryClientProvider } from "@tanstack/react-query"

import { AgentPanel } from "@/features/agent/AgentPanel"
import { AssistPanel } from "@/features/assist/AssistPanel"
import { EnrollmentPanel } from "@/features/enrollment/EnrollmentPanel"
import { GlassesPanel } from "@/features/glasses/GlassesPanel"
import { MemoryPanel } from "@/features/memory/MemoryPanel"
import { SpeechPanel } from "@/features/speech/SpeechPanel"
import { VideoStage } from "@/features/video/VideoStage"
import { VisionPanel } from "@/features/vision/VisionPanel"
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TooltipProvider } from "@/components/ui/tooltip"
import { Toaster } from "@/components/ui/sonner"
import { TranscriptProvider } from "@/hooks/TranscriptProvider"
import { useGlasses } from "@/store/glasses"

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Every service here can be down without the console being broken --
      // panels report their own reachability rather than retrying into a wall.
      retry: false,
      refetchOnWindowFocus: false,
    },
  },
})

export default function App() {
  const sessionId = useGlasses((glasses) => glasses.session?.session_id ?? null)

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delayDuration={200}>
        <div className="flex h-dvh flex-col">
          <header className="flex items-center gap-3 border-b px-4 py-2.5">
            <h1 className="text-sm font-semibold">Visual Memory Assistant</h1>
            <span className="text-xs text-muted-foreground">
              console — glasses POV, Cosmos, personal memory, speech, assistant
            </span>
          </header>

          {/* `orientation`, not `direction` -- react-resizable-panels v4
              renamed it, and shadcn's wrapper passes props straight through. */}
          <ResizablePanelGroup orientation="horizontal" className="min-h-0 flex-1 overflow-hidden">
            <ResizablePanel defaultSize={62} minSize={35}>
              <div className="h-full min-h-0 overflow-hidden p-3">
                <VideoStage />
              </div>
            </ResizablePanel>

            <ResizableHandle withHandle />

            <ResizablePanel defaultSize={38} minSize={25} className="min-h-0 overflow-hidden">
              <TranscriptProvider sessionId={sessionId}>
                <Tabs defaultValue="glasses" className="flex h-full flex-col p-3">
                  <TabsList>
                    <TabsTrigger value="glasses">Glasses</TabsTrigger>
                    <TabsTrigger value="assist">Assist</TabsTrigger>
                    <TabsTrigger value="vision">Vision</TabsTrigger>
                    <TabsTrigger value="memory">Memory</TabsTrigger>
                    <TabsTrigger value="speech">Speech</TabsTrigger>
                    <TabsTrigger value="assistant">Assistant</TabsTrigger>
                    <TabsTrigger value="enroll">Enroll</TabsTrigger>
                  </TabsList>
                  <TabsContent
                    value="glasses"
                    forceMount
                    className="min-h-0 flex-1 data-[state=inactive]:hidden"
                  >
                    <GlassesPanel />
                  </TabsContent>
                  <TabsContent
                    value="assist"
                    forceMount
                    className="min-h-0 flex-1 data-[state=inactive]:hidden"
                  >
                    <AssistPanel />
                  </TabsContent>
                  <TabsContent value="vision" className="min-h-0 flex-1">
                    <VisionPanel />
                  </TabsContent>
                  <TabsContent value="memory" className="min-h-0 flex-1">
                    <MemoryPanel />
                  </TabsContent>
                  <TabsContent
                    value="speech"
                    forceMount
                    className="min-h-0 flex-1 data-[state=inactive]:hidden"
                  >
                    <SpeechPanel />
                  </TabsContent>
                  <TabsContent
                    value="assistant"
                    forceMount
                    className="min-h-0 flex-1 data-[state=inactive]:hidden"
                  >
                    <AgentPanel />
                  </TabsContent>
                  <TabsContent value="enroll" className="min-h-0 flex-1">
                    <EnrollmentPanel />
                  </TabsContent>
                </Tabs>
              </TranscriptProvider>
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  )
}
