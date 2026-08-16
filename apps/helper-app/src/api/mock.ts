// Fake HelperApi so the pairing and request-list screens run with no
// gateway present. Swapped for the real client in E3 - see src/api/index.ts.
import type {
  AssistAcceptResponse,
  AssistRequest,
  DeviceCredentialResponse,
  SessionTokenResponse,
} from "./contract";
import type { HelperApi } from "./types";

const startedAt = Date.now();
const minutesAgo = (m: number) => new Date(startedAt - m * 60_000).toISOString();
const minutesFromNow = (m: number) => new Date(startedAt + m * 60_000).toISOString();

const fakeRequests: AssistRequest[] = [
  {
    request_id: "req-mock-1",
    session_id: "session-mock-1",
    device_id: "glasses-mock-1",
    state: "requested",
    requested_at: minutesAgo(2),
    expires_at: minutesFromNow(1),
  },
  {
    request_id: "req-mock-2",
    session_id: "session-mock-2",
    device_id: "glasses-mock-2",
    state: "requested",
    requested_at: minutesAgo(6),
    expires_at: minutesFromNow(0.5),
  },
];

export const mockApi: HelperApi = {
  async claimPairing(_gatewayUrl, _pairingCode, deviceId): Promise<DeviceCredentialResponse> {
    return {
      device_id: deviceId,
      credential: "mock-credential-do-not-use",
      expires_at: minutesFromNow(60 * 24),
    };
  },

  async listRequests(): Promise<AssistRequest[]> {
    return fakeRequests;
  },

  async acceptRequest(sessionId): Promise<AssistAcceptResponse> {
    return { state: "accepted", helper_identity: `helper-${sessionId}` };
  },

  async getHelperToken(_sessionId): Promise<SessionTokenResponse> {
    throw new Error("getHelperToken has no mock yet - the real client lands in E3");
  },
};
