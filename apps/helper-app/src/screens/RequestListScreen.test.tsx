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
  test("loads and renders requests sorted newest-requested-first", async () => {
    mockListRequests.mockResolvedValue([
      request({ request_id: "req-old", session_id: "sess-old", requested_at: "2026-08-16T00:00:00Z" }),
      request({ request_id: "req-new", session_id: "sess-new", requested_at: "2026-08-16T00:05:00Z" }),
    ]);

    await render(<RequestListScreen onAccepted={jest.fn()} />);

    const rendered = screen.getAllByText(/sess-(old|new)/).map((node) => node.props.children);
    expect(rendered).toEqual(["sess-new", "sess-old"]);
  });

  test("shows the empty state when there are no pending requests", async () => {
    await render(<RequestListScreen onAccepted={jest.fn()} />);

    expect(screen.getByText("No one needs help right now.")).toBeTruthy();
  });

  test("polls every 3 seconds", async () => {
    await render(<RequestListScreen onAccepted={jest.fn()} />);
    expect(mockListRequests).toHaveBeenCalledTimes(1);

    await act(() => jest.advanceTimersByTimeAsync(3000));
    expect(mockListRequests).toHaveBeenCalledTimes(2);

    await act(() => jest.advanceTimersByTimeAsync(3000));
    expect(mockListRequests).toHaveBeenCalledTimes(3);
  });

  test("stops polling after unmount", async () => {
    const { unmount } = await render(<RequestListScreen onAccepted={jest.fn()} />);
    expect(mockListRequests).toHaveBeenCalledTimes(1);

    await unmount();
    await act(() => jest.advanceTimersByTimeAsync(9000));

    expect(mockListRequests).toHaveBeenCalledTimes(1);
  });

  test("pull-to-refresh triggers an immediate reload", async () => {
    await render(<RequestListScreen onAccepted={jest.fn()} />);
    expect(mockListRequests).toHaveBeenCalledTimes(1);

    await act(async () => {
      await fireEvent(screen.getByTestId("request-list"), "refresh");
    });

    expect(mockListRequests).toHaveBeenCalledTimes(2);
  });

  test("an assist event from the WebSocket triggers an immediate reload", async () => {
    await render(<RequestListScreen onAccepted={jest.fn()} />);
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
    const { unmount } = await render(<RequestListScreen onAccepted={jest.fn()} />);

    await unmount();

    expect(close).toHaveBeenCalledTimes(1);
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
    await render(<RequestListScreen onAccepted={onAccepted} />);

    await fireEvent.press(screen.getByText("Accept"));

    expect(mockAcceptRequest).toHaveBeenCalledWith("sess-1");
    expect(mockGetHelperToken).toHaveBeenCalledWith("sess-1");
    expect(onAccepted).toHaveBeenCalledWith("sess-1", "wss://lk.example", "lk-token");
  });

  test("while accepting, pressing another row's Accept button does nothing", async () => {
    mockListRequests.mockResolvedValue([
      request({ request_id: "req-1", session_id: "sess-1" }),
      request({ request_id: "req-2", session_id: "sess-2" }),
    ]);
    let resolveAccept!: (value: unknown) => void;
    mockAcceptRequest.mockReturnValue(new Promise((resolve) => (resolveAccept = resolve)));
    await render(<RequestListScreen onAccepted={jest.fn()} />);

    await act(async () => {
      fireEvent.press(screen.getByTestId("accept-sess-1"));
    });
    // The Pressable's disabled prop should make this a no-op; either way,
    // what actually matters is that a second accept never fires.
    await act(async () => {
      fireEvent.press(screen.getByTestId("accept-sess-2"));
    });

    expect(mockAcceptRequest).toHaveBeenCalledTimes(1);
    expect(mockAcceptRequest).toHaveBeenCalledWith("sess-1");

    await act(async () => {
      resolveAccept({ state: "accepted", helper_identity: "helper-sess-1" });
    });
  });

  test("a 409 (conflict) from accept shows a specific message and reloads the list", async () => {
    mockListRequests.mockResolvedValue([request()]);
    mockAcceptRequest.mockRejectedValue(new GatewayRequestError("failed", 409, "conflict"));
    await render(<RequestListScreen onAccepted={jest.fn()} />);
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
    await render(<RequestListScreen onAccepted={onAccepted} />);

    await fireEvent.press(screen.getByText("Accept"));

    expect(screen.getByText("This request is no longer available.")).toBeTruthy();
    expect(onAccepted).not.toHaveBeenCalled();
    // Not stuck: the button is interactive again, not permanently spinning.
    expect(screen.getByText("Accept")).toBeTruthy();
  });
});
