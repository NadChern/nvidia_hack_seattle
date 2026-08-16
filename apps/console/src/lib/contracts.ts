/**
 * The wire shapes this console reads.
 *
 * Hand-written for now, and deliberately narrow: each type covers only the
 * fields the console actually uses, so a service adding a field never breaks
 * the build. The intended end state is generation from each service's
 * `/openapi.json` (they are all FastAPI, publishing the same Pydantic models
 * the contracts define) plus the vision contract's JSON Schema for the
 * WebSocket shapes, which OpenAPI does not describe. Until that lands, the
 * `OVERLAY_SCHEMA` check below is what stops this file drifting silently.
 */

/** The `packages/vision-contract` version this file was written against. */
export const OVERLAY_SCHEMA = "1.4"

export type MotionState = "absent" | "moving" | "settling" | "at_rest"

export interface BoundingBox {
  x_min: number
  y_min: number
  x_max: number
  y_max: number
}

export interface IdentityMatch {
  object_id: string | null
  best_score: number | null
  margin: number | null
  runner_up_object_id: string | null
  reason_code: string
  escalated: boolean
}

export interface OverlayTrack {
  track_id: string
  label: string
  confidence: number
  box: BoundingBox
  motion_state: MotionState
  /** Metric range along the view ray -- how far away the object is. */
  depth_m: number | null
  /**
   * How old that reading is. Depth is sampled at a cadence rather than per
   * frame, so this must be shown alongside the value: a number presented as
   * live when it is seconds stale is worse than showing none.
   */
  depth_age_s: number | null
  identity?: IdentityMatch | null
}

export interface OverlayFrame {
  schema_version: string
  session_id: string
  media_epoch_id: string
  sequence: number
  captured_at: string
  relayed_at: string
  emitted_at: string
  width: number
  height: number
  tracks: OverlayTrack[]
  pipeline_latency_ms: number
}

export interface OverlayHello {
  type: "overlay_hello"
  schema_version: string
  source_fps: number
  detector_kind: string
  depth_kind: string
  session_id: string | null
}

export function isOverlayHello(value: unknown): value is OverlayHello {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { type?: unknown }).type === "overlay_hello"
  )
}

export interface VisionStatus {
  ready: boolean
  not_ready_reason: string | null
  config: {
    detector_kind: string
    depth_kind: string
    verifier_kind: string
    detection_labels: string[]
    max_detections_per_frame?: number
    source_fps: number
  }
  frame_rate: { configured_fps: number; observed_fps: number | null }
  models?: {
    detector: { ready?: boolean; device?: string; [key: string]: unknown }
    depth: { ready?: boolean; device?: string; [key: string]: unknown }
  }
  identity?: {
    gallery_objects: number
    gallery_views: number
    resolved: number
    ambiguous: number
    unmatched: number
    escalated: number
  }
  registration?: { attempts: number; succeeded: number; failed: number; active: number }
  verifier: string
  verification: {
    queue_depth: number
    concurrency: number
    pending: number
    dropped: number
    failed: number
  }
  overlay: { enabled: boolean; viewers: number; max_viewers: number; dropped: number }
  metrics: {
    frames_processed: number
    candidates_proposed: number
    candidates_confirmed: number
    candidates_rejected: number
    candidates_unverified: number
    sightings_not_promoted: number
    vanishings_questioned: number
  }
}

export type PipelineOutcome = "confirmed" | "rejected" | "unverified" | "not_promoted"

export interface VisionEvent {
  at: string
  track_id: string
  label: string
  action: string
  outcome: PipelineOutcome
  reason_code: string
  confidence: number
}

export interface GatewayStatus {
  ready: boolean
  config: { sample_fps: number; dimension_guard_mode: string }
  relay: { subscribers: number }
  metrics: {
    video: {
      received: number
      admitted: number
      rejected_dimensions: number
      relayed: number
    }
  }
}

export interface SessionToken {
  session_id: string
  device_id: string
  room: string
  identity: string
  token: string
  livekit_url: string
  expires_at: string
}

export interface GatewaySessionSummary {
  session_id: string
  device_id: string
  room: string
  created_at: string
  last_seen_at: string
  publisher_present: boolean
}

export interface GatewaySessionList {
  sessions: GatewaySessionSummary[]
}

export interface PairingCode {
  pairing_code: string
  expires_at: string
}

export interface DeviceReplyEvent extends AgentAnswer {
  schema_version: "1.0"
  type: "reply"
  session_id: string
  question: string
  occurred_at: string
}

/**
 * `answer_status` is not decoration and must never be dropped when rendering.
 * The memory contract is explicit that a conversational layer may shorten
 * `spoken_answer` but must preserve the status, the uncertainty, and any
 * invalidation -- a console that showed the sentence alone would demonstrate
 * the exact failure this system exists to prevent.
 */
export type AnswerStatus = "confirmed" | "stale" | "in_transit" | "unknown" | "unavailable"

export interface QueryAnswer {
  answer_status: AnswerStatus
  spoken_answer: string
  object_label?: string | null
  as_of?: string | null
  evidence?: { url: string; media_type: string } | null
}


export interface SpeechBackend {
  name: string
  /**
   * Whether a model is actually behind it. A stub returns a valid WAV
   * containing pure silence with a 200, so without this a caller cannot tell
   * "not installed here" from "broken" -- both are a button press and nothing
   * audible.
   */
  real: boolean
}

export interface SpeechStatus {
  service: string
  backends: { tts: SpeechBackend; stt: SpeechBackend }
}

/**
 * One contiguous stretch of the wearer's speech, as the speech service
 * transcribed it. `session_id`/`epoch_id`/`pts_samples_start` locate it back
 * in the original relay stream regardless of any resampling on the way.
 */
export interface Transcript {
  text: string
  session_id: string
  epoch_id: string
  pts_samples_start: number
  samples: number
  sample_rate: number
}

export type AgentAnswerStatus =
  | "confirmed"
  | "last_confirmed_only"
  | "unknown"
  | "ambiguous_object"

export interface AgentAnswer {
  reply: string
  answer_status: AgentAnswerStatus | null
  object_id: string | null
  guard: "passed" | `vetoed:${number}` | `registration:${"prompt" | "succeeded" | "failed"}`
  latency_ms: number
}

export interface EnrolledObject {
  object_id: string
  label: string
  created_at?: string
  registry_version: number
}

export type EnrollmentState = "capturing" | "extracting" | "succeeded" | "failed"

export interface EnrollmentProgress {
  object_id: string
  label: string
  state: EnrollmentState
  frames_total: number
  detections: number
  quality_passed: number
  selected_views: number
  reason_code: string | null
  message: string | null
}

export interface ObjectGalleryView {
  view_id: string
  object_id: string
  view_index: number
  crop_reference: string
}

export interface ObjectGallery {
  registry_version: number
  objects: EnrolledObject[]
  views: ObjectGalleryView[]
}

export interface AgentStatus {
  backend: "stub" | "local" | "external"
  model: string
  endpoint_host: string
}
