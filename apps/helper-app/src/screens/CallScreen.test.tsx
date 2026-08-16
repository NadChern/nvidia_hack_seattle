jest.mock("@livekit/react-native", () => {
  const React = require("react");
  const { Text, View } = require("react-native");
  return {
    AudioSession: {
      startAudioSession: jest.fn(),
      stopAudioSession: jest.fn(),
    },
    LiveKitRoom: jest.fn((props: any) => React.createElement(React.Fragment, null, props.children)),
    useConnectionState: jest.fn(),
    useRoomContext: jest.fn(),
    useTracks: jest.fn(),
    isTrackReference: jest.fn((t: any) => Boolean(t?.__isTrackRef)),
    VideoTrack: jest.fn((props: any) =>
      React.createElement(View, { testID: "video-track" }, React.createElement(Text, null, props.trackRef.sid)),
    ),
  };
});

jest.mock("livekit-client", () => ({
  ConnectionState: {
    Connecting: "connecting",
    Connected: "connected",
    Reconnecting: "reconnecting",
    Disconnected: "disconnected",
  },
  Track: { Source: { Camera: "camera" } },
}));

import { act, render, fireEvent, screen } from "@testing-library/react-native";
import { AudioSession, LiveKitRoom, useConnectionState, useRoomContext, useTracks } from "@livekit/react-native";
import { ConnectionState } from "livekit-client";
import { CallScreen, describeConnectionState } from "./CallScreen";

const mockUseConnectionState = useConnectionState as jest.Mock;
const mockUseRoomContext = useRoomContext as jest.Mock;
const mockUseTracks = useTracks as jest.Mock;
const mockLiveKitRoom = LiveKitRoom as jest.Mock;

describe("describeConnectionState", () => {
  test("Connecting", () => {
    expect(describeConnectionState(ConnectionState.Connecting as any, null)).toBe("Connecting...");
  });

  test("Connected", () => {
    expect(describeConnectionState(ConnectionState.Connected as any, null)).toBe("Connected");
  });

  test("Reconnecting", () => {
    expect(describeConnectionState(ConnectionState.Reconnecting as any, null)).toBe("Reconnecting...");
  });

  test("Disconnected with no reason", () => {
    expect(describeConnectionState(ConnectionState.Disconnected as any, null)).toBe("Disconnected");
  });

  test("Disconnected with a reason includes it", () => {
    expect(describeConnectionState(ConnectionState.Disconnected as any, "server shutting down")).toBe(
      "Disconnected: server shutting down",
    );
  });
});

describe("CallScreen", () => {
  let room: { on: jest.Mock; off: jest.Mock };

  beforeEach(() => {
    jest.clearAllMocks();
    room = { on: jest.fn(), off: jest.fn() };
    mockUseRoomContext.mockReturnValue(room);
    mockUseConnectionState.mockReturnValue(ConnectionState.Connecting);
    mockUseTracks.mockReturnValue([]);
  });

  test("renders LiveKitRoom with video disabled and audio enabled -- the camera must never be requested", async () => {
    await render(<CallScreen serverUrl="wss://lk.example" token="tok" onHangUp={jest.fn()} />);

    expect(mockLiveKitRoom).toHaveBeenCalledWith(
      expect.objectContaining({
        serverUrl: "wss://lk.example",
        token: "tok",
        connect: true,
        audio: true,
        video: false,
      }),
      undefined,
    );
  });

  test("starts the audio session on mount and stops it on unmount", async () => {
    const { unmount } = await render(<CallScreen serverUrl="wss://lk.example" token="tok" onHangUp={jest.fn()} />);

    expect(AudioSession.startAudioSession).toHaveBeenCalledTimes(1);
    expect(AudioSession.stopAudioSession).not.toHaveBeenCalled();

    await unmount();

    expect(AudioSession.stopAudioSession).toHaveBeenCalledTimes(1);
  });

  test("shows a spinner while connecting and no remote track has arrived", async () => {
    mockUseConnectionState.mockReturnValue(ConnectionState.Connecting);
    mockUseTracks.mockReturnValue([]);

    await render(<CallScreen serverUrl="wss://lk.example" token="tok" onHangUp={jest.fn()} />);

    expect(screen.queryByTestId("video-track")).toBeNull();
    expect(screen.queryByText("Waiting for her camera...")).toBeNull();
  });

  test("shows a waiting message once connected with no remote track yet", async () => {
    mockUseConnectionState.mockReturnValue(ConnectionState.Connected);
    mockUseTracks.mockReturnValue([]);

    await render(<CallScreen serverUrl="wss://lk.example" token="tok" onHangUp={jest.fn()} />);

    expect(screen.getByText("Waiting for her camera...")).toBeTruthy();
    expect(screen.queryByTestId("video-track")).toBeNull();
  });

  test("renders the remote video track once one is available from another participant", async () => {
    mockUseConnectionState.mockReturnValue(ConnectionState.Connected);
    mockUseTracks.mockReturnValue([
      { __isTrackRef: true, sid: "track-1", participant: { isLocal: false } },
    ]);

    await render(<CallScreen serverUrl="wss://lk.example" token="tok" onHangUp={jest.fn()} />);

    expect(screen.getByTestId("video-track")).toBeTruthy();
    expect(screen.queryByText("Waiting for her camera...")).toBeNull();
  });

  test("ignores a local participant's own track for the remote video slot", async () => {
    mockUseConnectionState.mockReturnValue(ConnectionState.Connected);
    mockUseTracks.mockReturnValue([{ __isTrackRef: true, sid: "own-track", participant: { isLocal: true } }]);

    await render(<CallScreen serverUrl="wss://lk.example" token="tok" onHangUp={jest.fn()} />);

    expect(screen.queryByTestId("video-track")).toBeNull();
    expect(screen.getByText("Waiting for her camera...")).toBeTruthy();
  });

  test("shows the connection status label", async () => {
    mockUseConnectionState.mockReturnValue(ConnectionState.Reconnecting);

    await render(<CallScreen serverUrl="wss://lk.example" token="tok" onHangUp={jest.fn()} />);

    expect(screen.getByText("Reconnecting...")).toBeTruthy();
  });

  test("pressing Hang up calls onHangUp", async () => {
    const onHangUp = jest.fn();
    await render(<CallScreen serverUrl="wss://lk.example" token="tok" onHangUp={onHangUp} />);

    await fireEvent.press(screen.getByText("Hang up"));

    expect(onHangUp).toHaveBeenCalledTimes(1);
  });

  test("a room 'disconnected' event surfaces its reason in the status label", async () => {
    mockUseConnectionState.mockReturnValue(ConnectionState.Disconnected);

    await render(<CallScreen serverUrl="wss://lk.example" token="tok" onHangUp={jest.fn()} />);

    expect(room.on).toHaveBeenCalledWith("disconnected", expect.any(Function));
    expect(screen.getByText("Disconnected")).toBeTruthy(); // starts as plain "Disconnected"
    const onDisconnect = room.on.mock.calls.find(([event]) => event === "disconnected")![1];

    await act(async () => {
      onDisconnect("server shutting down");
    });

    expect(screen.getByText("Disconnected: server shutting down")).toBeTruthy();
  });
});
