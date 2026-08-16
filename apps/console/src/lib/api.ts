/**
 * Reaching the five services.
 *
 * Everything goes through same-origin `/api/<service>` paths that the Vite dev
 * server proxies (see `vite.config.ts`). Nothing here knows a host or a port,
 * so there is no CORS story in development and a built console points at
 * whatever serves it.
 */

const BASE = {
  gateway: "/api/gateway",
  vision: "/api/vision",
  memory: "/api/memory",
  speech: "/api/speech",
  agent: "/api/agent",
} as const

export type ServiceName = keyof typeof BASE

/** The internal bearer token, when one is configured. Unset in development. */
const TOKEN: string | undefined = import.meta.env["VITE_VMA_INTERNAL_API_TOKEN"]

function authHeaders(): HeadersInit {
  return TOKEN ? { authorization: `Bearer ${TOKEN}` } : {}
}

export class ApiError extends Error {
  // Declared and assigned rather than using parameter properties:
  // `erasableSyntaxOnly` forbids syntax that emits code, and a parameter
  // property does.
  readonly service: ServiceName
  readonly status: number
  readonly detail: string

  constructor(service: ServiceName, status: number, detail: string) {
    super(`${service} ${status}: ${detail}`)
    this.name = "ApiError"
    this.service = service
    this.status = status
    this.detail = detail
  }
}

export async function get<T>(service: ServiceName, path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE[service]}${path}`, { headers: authHeaders(), signal })
  if (!response.ok) {
    throw new ApiError(service, response.status, await response.text())
  }
  return (await response.json()) as T
}

export async function post<T>(service: ServiceName, path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${BASE[service]}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeaders() },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })
  if (!response.ok) {
    throw new ApiError(service, response.status, await response.text())
  }
  return (await response.json()) as T
}

export async function del(service: ServiceName, path: string): Promise<void> {
  await fetch(`${BASE[service]}${path}`, { method: "DELETE", headers: authHeaders() }).catch(
    () => undefined,
  )
}

/** DELETE where failure must be visible rather than best-effort cleanup. */
export async function delChecked(service: ServiceName, path: string): Promise<void> {
  const response = await fetch(`${BASE[service]}${path}`, {
    method: "DELETE",
    headers: authHeaders(),
  })
  if (!response.ok) {
    throw new ApiError(service, response.status, await response.text())
  }
}

/** GET returning authenticated raw bytes, used for durable reference crops. */
export async function getBlob(
  service: ServiceName,
  path: string,
  signal?: AbortSignal,
): Promise<Blob> {
  const response = await fetch(`${BASE[service]}${path}`, { headers: authHeaders(), signal })
  if (!response.ok) {
    throw new ApiError(service, response.status, await response.text())
  }
  return await response.blob()
}

/** POST returning raw bytes -- speech synthesis hands back audio/wav. */
export async function postForBlob(
  service: ServiceName,
  path: string,
  body: unknown,
): Promise<Blob> {
  const response = await fetch(`${BASE[service]}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    throw new ApiError(service, response.status, await response.text())
  }
  return await response.blob()
}

/**
 * A WebSocket URL on this origin, so the dev proxy handles it.
 *
 * The token goes in the query string because **a browser cannot set headers on
 * a WebSocket handshake** -- `vision_worker.deps.authorize_websocket` accepts
 * it there for exactly this reason, and only there.
 */
export function websocketUrl(service: ServiceName, path: string): string {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws"
  const url = new URL(`${scheme}://${window.location.host}${BASE[service]}${path}`)
  if (TOKEN) {
    url.searchParams.set("token", TOKEN)
  }
  return url.toString()
}
