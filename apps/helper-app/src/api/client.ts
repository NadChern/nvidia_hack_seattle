// Real HelperApi implementation against the gateway. See src/api/index.ts
// for the swap point that replaced mock.ts with this file.
import type {
  AssistAcceptResponse,
  AssistRequest,
  AssistRequestListResponse,
  DeviceCredentialResponse,
  SessionTokenResponse,
} from "./contract";
import type { HelperApi } from "./types";
import { getPairingCredential } from "../storage/credentials";

// The gateway's error body is always { code, message, context? } -- see
// services/media-gateway/src/media_gateway/errors.py's GatewayError.to_payload.
// `status`/`code` let a caller distinguish e.g. a 409 "someone else already
// accepted this" from a network failure, instead of pattern-matching a
// message string.
export class GatewayRequestError extends Error {
  readonly status: number | null;
  readonly code: string | null;

  constructor(message: string, status: number | null = null, code: string | null = null) {
    super(message);
    this.name = "GatewayRequestError";
    this.status = status;
    this.code = code;
  }
}

function parseErrorCode(body: string): string | null {
  try {
    const parsed = JSON.parse(body) as { code?: unknown };
    return typeof parsed.code === "string" ? parsed.code : null;
  } catch {
    return null;
  }
}

async function request<T>(
  baseUrl: string,
  path: string,
  init: RequestInit & { credential?: string } = {},
): Promise<T> {
  const { credential, headers, ...rest } = init;
  const res = await fetch(`${baseUrl}${path}`, {
    ...rest,
    headers: {
      "content-type": "application/json",
      ...(credential ? { authorization: `Bearer ${credential}` } : {}),
      ...headers,
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new GatewayRequestError(
      `${init.method ?? "GET"} ${path} failed: ${res.status} ${body}`,
      res.status,
      parseErrorCode(body),
    );
  }
  return (await res.json()) as T;
}

async function authedRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const credential = await getPairingCredential();
  if (!credential) {
    throw new GatewayRequestError("not paired with a gateway yet");
  }
  return request<T>(credential.gateway_url, path, { ...init, credential: credential.credential });
}

export const gatewayApi: HelperApi = {
  async claimPairing(gatewayUrl, pairingCode, deviceId): Promise<DeviceCredentialResponse> {
    return request<DeviceCredentialResponse>(gatewayUrl, "/v1/pairing/claim", {
      method: "POST",
      body: JSON.stringify({ pairing_code: pairingCode, device_id: deviceId }),
    });
  },

  async listRequests(): Promise<AssistRequest[]> {
    const res = await authedRequest<AssistRequestListResponse>("/v1/assist/requests");
    return res.requests;
  },

  async acceptRequest(sessionId): Promise<AssistAcceptResponse> {
    return authedRequest<AssistAcceptResponse>(`/v1/assist/${sessionId}/accept`, {
      method: "POST",
    });
  },

  async getHelperToken(sessionId): Promise<SessionTokenResponse> {
    return authedRequest<SessionTokenResponse>(`/v1/sessions/${sessionId}/helper`, {
      method: "POST",
    });
  },
};
