import { mockApi } from "./mock";
import type { AssistRequestState } from "./contract";

const VALID_STATES: AssistRequestState[] = ["requested", "accepted", "ended"];

describe("mockApi", () => {
  test("listRequests returns fixtures with a valid AssistRequestState", async () => {
    const requests = await mockApi.listRequests();

    expect(requests.length).toBeGreaterThan(0);
    for (const request of requests) {
      expect(VALID_STATES).toContain(request.state);
    }
  });

  test("claimPairing returns a credential that has not expired yet", async () => {
    const result = await mockApi.claimPairing("https://gateway.example", "code", "device-01");

    expect(result.device_id).toBe("device-01");
    expect(new Date(result.expires_at).getTime()).toBeGreaterThan(Date.now());
  });

  test("acceptRequest resolves to an accepted state naming the session's helper", async () => {
    const result = await mockApi.acceptRequest("session-mock-1");

    expect(result).toEqual({ state: "accepted", helper_identity: "helper-session-mock-1" });
  });

  test("getHelperToken has no mock yet and rejects rather than returning a fake token", async () => {
    await expect(mockApi.getHelperToken("session-mock-1")).rejects.toThrow();
  });
});
