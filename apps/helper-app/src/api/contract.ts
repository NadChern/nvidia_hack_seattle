// Wire types for the assist feature. Field names mirror the gateway's JSON
// exactly, so this file should need no translation layer against the real
// backend in E3.

// Matches the gateway's `AssistState` one-for-one (see
// services/media-gateway/src/media_gateway/domain/assist.py) -- the request
// registry, the HUD device event, and this app must agree on the same three
// words per docs/12-Media-Relay-Contract.md.
export type AssistRequestState = "requested" | "accepted" | "ended";

export interface AssistRequest {
  request_id: string;
  session_id: string;
  device_id: string;
  state: AssistRequestState;
  requested_at: string;
  expires_at: string;
}

export interface AssistRequestListResponse {
  requests: AssistRequest[];
}

export interface AssistAcceptResponse {
  state: "accepted";
  helper_identity: string;
}

// Matches services/media-gateway/src/media_gateway/domain/assist.py's
// AssistNotification union exactly (hello/keepalive are written directly by
// the WS pump in api/assist.py, not part of that union, and carry no
// session_id/device_id -- unlike the device-events channel's hello).
export type AssistEvent =
  | { schema_version: "1.0"; type: "hello"; occurred_at: string }
  | { schema_version: "1.0"; type: "assist_requested"; request_id: string; session_id: string; occurred_at: string }
  | { schema_version: "1.0"; type: "assist_accepted"; request_id: string; session_id: string; occurred_at: string }
  | { schema_version: "1.0"; type: "assist_ended"; request_id: string; session_id: string; occurred_at: string }
  | { schema_version: "1.0"; type: "keepalive"; occurred_at: string };

// POST /v1/sessions/{session_id}/helper. Same response shape as /viewer -
// see services/media-gateway/src/media_gateway/api/sessions.py::SessionTokenResponse.
export interface SessionTokenResponse {
  session_id: string;
  device_id: string;
  room: string;
  livekit_url: string;
  identity: string;
  token: string;
  expires_at: string;
}

// POST /v1/pairing/claim - see services/media-gateway/src/media_gateway/api/pairing.py.
export interface PairingClaimRequest {
  pairing_code: string;
  device_id: string;
}

export interface DeviceCredentialResponse {
  device_id: string;
  credential: string;
  expires_at: string;
}

// What the console encodes into the pairing QR. gateway_url is not part of
// the /v1/pairing response - the console adds it so the phone knows which
// server to talk to.
export interface PairingQrPayload {
  gateway_url: string;
  pairing_code: string;
  expires_at: string;
}
