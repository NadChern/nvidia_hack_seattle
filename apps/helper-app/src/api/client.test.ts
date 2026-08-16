jest.mock("../storage/credentials", () => ({
  getPairingCredential: jest.fn(),
}));

import { getPairingCredential } from "../storage/credentials";
import { gatewayApi, GatewayRequestError } from "./client";

const mockGetPairingCredential = getPairingCredential as jest.Mock;

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: () => Promise.resolve(JSON.stringify(body)),
    json: () => Promise.resolve(body),
  } as unknown as Response;
}

function textResponse(status: number, body: string): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: () => Promise.resolve(body),
    json: () => Promise.reject(new Error("not json")),
  } as unknown as Response;
}

describe("gatewayApi.claimPairing", () => {
  beforeEach(() => {
    globalThis.fetch = jest.fn();
    mockGetPairingCredential.mockReset();
  });

  test("posts to /v1/pairing/claim with no Authorization header", async () => {
    (globalThis.fetch as jest.Mock).mockResolvedValue(
      jsonResponse(200, { device_id: "d1", credential: "v1.abc", expires_at: "2026-08-16T00:00:00Z" }),
    );

    const result = await gatewayApi.claimPairing("https://gateway.example", "the-code", "d1");

    expect(globalThis.fetch).toHaveBeenCalledWith(
      "https://gateway.example/v1/pairing/claim",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ pairing_code: "the-code", device_id: "d1" }),
      }),
    );
    const [, init] = (globalThis.fetch as jest.Mock).mock.calls[0];
    expect(init.headers.authorization).toBeUndefined();
    expect(result).toEqual({ device_id: "d1", credential: "v1.abc", expires_at: "2026-08-16T00:00:00Z" });
  });
});

describe("gatewayApi authed calls", () => {
  beforeEach(() => {
    globalThis.fetch = jest.fn();
    mockGetPairingCredential.mockReset();
  });

  test("listRequests attaches the stored credential as a Bearer token", async () => {
    mockGetPairingCredential.mockResolvedValue({
      gateway_url: "https://gateway.example",
      device_id: "helper-01",
      credential: "v1.stored-credential",
      expires_at: "2026-08-16T00:00:00Z",
    });
    (globalThis.fetch as jest.Mock).mockResolvedValue(jsonResponse(200, { requests: [] }));

    await gatewayApi.listRequests();

    const [url, init] = (globalThis.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("https://gateway.example/v1/assist/requests");
    expect(init.headers.authorization).toBe("Bearer v1.stored-credential");
  });

  test("acceptRequest posts to /v1/assist/{id}/accept with the stored credential", async () => {
    mockGetPairingCredential.mockResolvedValue({
      gateway_url: "https://gateway.example",
      device_id: "helper-01",
      credential: "v1.stored-credential",
      expires_at: "2026-08-16T00:00:00Z",
    });
    (globalThis.fetch as jest.Mock).mockResolvedValue(
      jsonResponse(200, { state: "accepted", helper_identity: "helper-sess-1" }),
    );

    const result = await gatewayApi.acceptRequest("sess-1");

    expect(globalThis.fetch).toHaveBeenCalledWith(
      "https://gateway.example/v1/assist/sess-1/accept",
      expect.objectContaining({ method: "POST" }),
    );
    expect(result).toEqual({ state: "accepted", helper_identity: "helper-sess-1" });
  });

  test("getHelperToken posts to /v1/sessions/{id}/helper with the stored credential", async () => {
    mockGetPairingCredential.mockResolvedValue({
      gateway_url: "https://gateway.example",
      device_id: "helper-01",
      credential: "v1.stored-credential",
      expires_at: "2026-08-16T00:00:00Z",
    });
    const tokenResponse = {
      session_id: "sess-1",
      device_id: "glasses-01",
      room: "room-1",
      livekit_url: "wss://lk.example",
      identity: "helper-sess-1",
      token: "lk-token",
      expires_at: "2026-08-16T00:05:00Z",
    };
    (globalThis.fetch as jest.Mock).mockResolvedValue(jsonResponse(200, tokenResponse));

    const result = await gatewayApi.getHelperToken("sess-1");

    expect(globalThis.fetch).toHaveBeenCalledWith(
      "https://gateway.example/v1/sessions/sess-1/helper",
      expect.objectContaining({ method: "POST" }),
    );
    expect(result).toEqual(tokenResponse);
  });

  test("throws without calling fetch when there is no stored credential", async () => {
    mockGetPairingCredential.mockResolvedValue(null);

    await expect(gatewayApi.listRequests()).rejects.toThrow(GatewayRequestError);
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });
});

describe("gatewayApi error handling", () => {
  beforeEach(() => {
    globalThis.fetch = jest.fn();
    mockGetPairingCredential.mockResolvedValue({
      gateway_url: "https://gateway.example",
      device_id: "helper-01",
      credential: "v1.stored-credential",
      expires_at: "2026-08-16T00:00:00Z",
    });
  });

  test("a 409 with a JSON error body surfaces status and code", async () => {
    (globalThis.fetch as jest.Mock).mockResolvedValue(
      jsonResponse(409, { code: "conflict", message: "already accepted" }),
    );

    let caught: unknown;
    try {
      await gatewayApi.acceptRequest("sess-1");
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(GatewayRequestError);
    const error = caught as GatewayRequestError;
    expect(error.status).toBe(409);
    expect(error.code).toBe("conflict");
  });

  test("a 404 with a JSON error body surfaces status and code", async () => {
    (globalThis.fetch as jest.Mock).mockResolvedValue(
      jsonResponse(404, { code: "not_found", message: "unknown session" }),
    );

    let caught: unknown;
    try {
      await gatewayApi.getHelperToken("sess-1");
    } catch (e) {
      caught = e;
    }

    expect((caught as GatewayRequestError).status).toBe(404);
    expect((caught as GatewayRequestError).code).toBe("not_found");
  });

  test("a non-JSON error body still surfaces status, with a null code", async () => {
    (globalThis.fetch as jest.Mock).mockResolvedValue(textResponse(500, "internal server error"));

    let caught: unknown;
    try {
      await gatewayApi.listRequests();
    } catch (e) {
      caught = e;
    }

    expect((caught as GatewayRequestError).status).toBe(500);
    expect((caught as GatewayRequestError).code).toBeNull();
  });

  test("a successful response does not throw", async () => {
    (globalThis.fetch as jest.Mock).mockResolvedValue(jsonResponse(200, { requests: [] }));

    await expect(gatewayApi.listRequests()).resolves.toEqual([]);
  });
});
