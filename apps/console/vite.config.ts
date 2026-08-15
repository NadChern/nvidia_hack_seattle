import path from "node:path"

import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

/**
 * The console talks to five services on five ports. Rather than hard-code
 * hosts, everything is proxied under this dev server's own origin, so the app
 * only ever fetches same-origin paths and there is no CORS story in
 * development at all. A built console served from anywhere points at the same
 * paths, resolved by whatever serves it.
 *
 * `ws: true` on the vision and speech proxies is load-bearing: without it the
 * WebSocket upgrade is answered with a 404 that reads, in the browser, as a
 * mysterious immediate disconnect.
 */
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // What the shadcn/ElevenLabs registry generates its imports against.
    alias: { "@": path.resolve(import.meta.dirname, "./src") },
  },
  server: {
    proxy: {
      "/api/gateway": {
        target: process.env.VMA_GATEWAY_URL ?? "http://127.0.0.1:8080",
        changeOrigin: true,
        ws: true,
        rewrite: (p) => p.replace(/^\/api\/gateway/, ""),
      },
      "/api/vision": {
        target: process.env.VMA_VISION_URL ?? "http://127.0.0.1:8082",
        changeOrigin: true,
        ws: true,
        rewrite: (p) => p.replace(/^\/api\/vision/, ""),
      },
      "/api/memory": {
        target: process.env.VMA_MEMORY_URL ?? "http://127.0.0.1:8081",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/memory/, ""),
      },
      "/api/speech": {
        target: process.env.VMA_SPEECH_URL ?? "http://127.0.0.1:8085",
        changeOrigin: true,
        ws: true,
        rewrite: (p) => p.replace(/^\/api\/speech/, ""),
      },
      "/api/agent": {
        target: process.env.VMA_AGENT_URL ?? "http://127.0.0.1:8086",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/agent/, ""),
      },
    },
  },
})
