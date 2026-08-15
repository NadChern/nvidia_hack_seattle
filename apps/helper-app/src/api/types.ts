import type {
  AssistAcceptResponse,
  AssistRequest,
  DeviceCredentialResponse,
  SessionTokenResponse,
} from "./contract";

export interface HelperApi {
  claimPairing(gatewayUrl: string, pairingCode: string, deviceId: string): Promise<DeviceCredentialResponse>;
  listRequests(): Promise<AssistRequest[]>;
  acceptRequest(sessionId: string): Promise<AssistAcceptResponse>;
  getHelperToken(sessionId: string): Promise<SessionTokenResponse>;
}
