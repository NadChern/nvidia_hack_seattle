jest.mock("../api", () => ({
  api: {
    claimPairing: jest.fn(),
    listRequests: jest.fn(),
    acceptRequest: jest.fn(),
    getHelperToken: jest.fn(),
  },
}));

jest.mock("../api/events", () => ({
  connectAssistEvents: jest.fn(),
}));

import { act, render, fireEvent, screen, waitFor } from "@testing-library/react-native";
import { api } from "../api";
import { connectAssistEvents } from "../api/events";
import { GatewayRequestError } from "../api/client";
import { RequestListScreen } from "./RequestListScreen";

const mockListRequests = api.listRequests as jest.Mock;
const mockAcceptRequest = api.acceptRequest as jest.Mock;
const mockGetHelperToken = api.getHelperToken as jest.Mock;
const mockConnectAssistEvents = connectAssistEvents as jest.Mock;

function request(overrides: Partial<Parameters<typeof mockListRequests>[0]> = {}) {
  return {
    request_id: "req-1",
    session_id: "sess-1",
    device_id: "glasses-01",
    state: "requested" as const,
    requested_at: "2026-08-16T00:00:00Z",
    expires_at: "2026-08-16T00:01:00Z",
    ...overrides,
  };
}

function renderScreen(overrides: Partial<Parameters<typeof RequestListScreen>[0]> = {}) {
  return render(
    <RequestListScreen onAccepted={jest.fn()} onUnpaired={jest.fn()} {...overrides} />,
  );
}

beforeEach(() => {
  jest.useFakeTimers();
  jest.clearAllMocks();
  mockListRequests.mockResolvedValue([]);
  mockConnectAssistEvents.mockReturnValue({ close: jest.fn() });
});

afterEach(() => {
  jest.useRealTimers();
});

describe("RequestListScreen", () => {
  test("shows the newest pending request as the single active call", async () => {
    mockListRequests.mockResolvedValue([
      request({ request_id: "req-old", session_id: "sess-old", requested_at: "2026-08-16T00:00:00Z" }),
      request({ request_id: "req-new", session_id: "sess-new", requested_at: "2026-08-16T00:05:00Z" }),
    ]);

    await renderScreen();

    // Only the newest request is shown -- this is a one-at-a-time "incoming
    // call" screen, not a list of everything pending.
    expect(screen.getByText("sess-new")).toBeTruthy();
    expect(screen.queryByText("sess-old")).toBeNull();
  });

  test("shows the empty state when there are no pending requests", async () => {
    await renderScreen();

    expect(screen.getByText("No one needs help right now.")).toBeTruthy();
  });

  test("polls every 3 seconds", async () => {
    await renderScreen();
    expect(mockListRequests).toHaveBeenCalledTimes(1);

    await act(() => jest.advanceTimersByTimeAsync(3000));
    expect(mockListRequests).toHaveBeenCalledTimes(2);

    await act(() => jest.advanceTimersByTimeAsync(3000));
    expect(mockListRequests).toHaveBeenCalledTimes(3);
  });

  test("stops polling after unmount", async () => {
    const { unmount } = await renderScreen();
    expect(mockListRequests).toHaveBeenCalledTimes(1);

    await unmount();
    await act(() => jest.advanceTimersByTimeAsync(9000));

    expect(mockListRequests).toHaveBeenCalledTimes(1);
  });

  test("pull-to-refresh triggers an immediate reload", async () => {
    await renderScreen();
    expect(mockListRequests).toHaveBeenCalledTimes(1);

    await act(async () => {
      await fireEvent(screen.getByTestId("request-list"), "refresh");
    });

    expect(mockListRequests).toHaveBeenCalledTimes(2);
  });

  test("an assist event from the WebSocket triggers an immediate reload", async () => {
    await renderScreen();
    expect(mockListRequests).toHaveBeenCalledTimes(1);
    const { onEvent } = mockConnectAssistEvents.mock.calls[0][0];

    await act(async () => {
      await onEvent({
        schema_version: "1.0",
        type: "assist_requested",
        request_id: "req-1",
        session_id: "sess-1",
        occurred_at: "2026-08-16T00:00:00Z",
      });
    });

    expect(mockListRequests).toHaveBeenCalledTimes(2);
  });

  test("closes the events connection on unmount", async () => {
    const close = jest.fn();
    mockConnectAssistEvents.mockReturnValue({ close });
    const { unmount } = await renderScreen();

    await unmount();

    expect(close).toHaveBeenCalledTimes(1);
  });

  test("declining hides the request and lets the next one take over", async () => {
    mockListRequests.mockResolvedValue([
      request({ request_id: "req-old", session_id: "sess-old", requested_at: "2026-08-16T00:00:00Z" }),
      request({ request_id: "req-new", session_id: "sess-new", requested_at: "2026-08-16T00:05:00Z" }),
    ]);

    await renderScreen();
    expect(screen.getByText("sess-new")).toBeTruthy();

    await act(async () => {
      fireEvent.press(screen.getByTestId("decline-sess-new"));
    });

    expect(screen.getByText("sess-old")).toBeTruthy();
    expect(screen.queryByText("sess-new")).toBeNull();
  });

  test("accept -> helper-token -> onAccepted happens in order on success", async () => {
    mockListRequests.mockResolvedValue([request()]);
    mockAcceptRequest.mockResolvedValue({ state: "accepted", helper_identity: "helper-sess-1" });
    mockGetHelperToken.mockResolvedValue({
      session_id: "sess-1",
      device_id: "glasses-01",
      room: "room-1",
      livekit_url: "wss://lk.example",
      identity: "helper-sess-1",
      token: "lk-token",
      expires_at: "2026-08-16T00:05:00Z",
    });
    const onAccepted = jest.fn();
    await renderScreen({ onAccepted });

    await fireEvent.press(screen.getByText("Accept"));

    expect(mockAcceptRequest).toHaveBeenCalledWith("sess-1");
    expect(mockGetHelperToken).toHaveBeenCalledWith("sess-1");
    expect(onAccepted).toHaveBeenCalledWith("sess-1", "wss://lk.example", "lk-token");
  });

  test("while accepting, pressing Accept again does nothing", async () => {
    mockListRequests.mockResolvedValue([request()]);
    let resolveAccept!: (value: unknown) => void;
    mockAcceptRequest.mockReturnValue(new Promise((resolve) => (resolveAccept = resolve)));
    await renderScreen();

    await act(async () => {
      fireEvent.press(screen.getByTestId("accept-sess-1"));
    });
    // The Pressable's disabled prop should make this a no-op; either way,
    // what actually matters is that a second accept never fires.
    await act(async () => {
      fireEvent.press(screen.getByTestId("accept-sess-1"));
    });

    expect(mockAcceptRequest).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveAccept({ state: "accepted", helper_identity: "helper-sess-1" });
    });
  });

  test("while accepting, Decline is disabled too", async () => {
    mockListRequests.mockResolvedValue([request()]);
    let resolveAccept!: (value: unknown) => void;
    mockAcceptRequest.mockReturnValue(new Promise((resolve) => (resolveAccept = resolve)));
    await renderScreen();

    await act(async () => {
      fireEvent.press(screen.getByTestId("accept-sess-1"));
    });
    await act(async () => {
      fireEvent.press(screen.getByTestId("decline-sess-1"));
    });

    // Still showing the same request -- decline did not go through mid-accept.
    expect(screen.getByText("sess-1")).toBeTruthy();

    await act(async () => {
      resolveAccept({ state: "accepted", helper_identity: "helper-sess-1" });
    });
  });

  test("a 409 (conflict) from accept shows a specific message and reloads the list", async () => {
    mockListRequests.mockResolvedValue([request()]);
    mockAcceptRequest.mockRejectedValue(new GatewayRequestError("failed", 409, "conflict"));
    await renderScreen();
    expect(mockListRequests).toHaveBeenCalledTimes(1);

    await fireEvent.press(screen.getByText("Accept"));

    expect(screen.getByText("Someone else already accepted this request.")).toBeTruthy();
    expect(mockGetHelperToken).not.toHaveBeenCalled();
    await waitFor(() => expect(mockListRequests).toHaveBeenCalledTimes(2));
  });

  test("a failure fetching the helper token after a successful accept surfaces an error, not a stuck spinner", async () => {
    mockListRequests.mockResolvedValue([request()]);
    mockAcceptRequest.mockResolvedValue({ state: "accepted", helper_identity: "helper-sess-1" });
    mockGetHelperToken.mockRejectedValue(new GatewayRequestError("failed", 404, "not_found"));
    const onAccepted = jest.fn();
    await renderScreen({ onAccepted });

    await fireEvent.press(screen.getByText("Accept"));

    expect(screen.getByText("This request is no longer available.")).toBeTruthy();
    expect(onAccepted).not.toHaveBeenCalled();
    // Not stuck: the button is interactive again, not permanently spinning.
    expect(screen.getByText("Accept")).toBeTruthy();
  });

  test("shows the pairing screen again on a 401 (stale credential after a gateway restart)", async () => {
    mockListRequests.mockRejectedValue(
      new GatewayRequestError("failed", 401, "unauthorized"),
    );
    const onUnpaired = jest.fn();
    await renderScreen({ onUnpaired });

    await waitFor(() => expect(onUnpaired).toHaveBeenCalledTimes(1));
  });
});
