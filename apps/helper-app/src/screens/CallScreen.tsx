import {
  AudioSession,
  isTrackReference,
  LiveKitRoom,
  useConnectionState,
  useRoomContext,
  useTracks,
  VideoTrack,
} from "@livekit/react-native";
import { ConnectionState, Track } from "livekit-client";
import { useEffect, useState } from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";

interface Props {
  serverUrl: string;
  token: string;
  onHangUp: () => void;
}

export function CallScreen({ serverUrl, token, onHangUp }: Props) {
  useEffect(() => {
    AudioSession.startAudioSession();
    return () => {
      AudioSession.stopAudioSession();
    };
  }, []);

  return (
    <LiveKitRoom
      serverUrl={serverUrl}
      token={token}
      connect
      audio
      video={false}
      options={{ adaptiveStream: { pixelDensity: "screen" } }}
      onDisconnected={onHangUp}
    >
      <RoomView onHangUp={onHangUp} />
    </LiveKitRoom>
  );
}

function RoomView({ onHangUp }: { onHangUp: () => void }) {
  const connectionState = useConnectionState();
  const room = useRoomContext();
  const [disconnectReason, setDisconnectReason] = useState<string | null>(null);
  const tracks = useTracks([Track.Source.Camera]);
  const remoteTrack = tracks.find((t) => !t.participant.isLocal && isTrackReference(t));

  useEffect(() => {
    const onDisconnect = (reason?: unknown) => {
      setDisconnectReason(reason ? String(reason) : "connection closed");
    };
    room.on("disconnected", onDisconnect);
    return () => {
      room.off("disconnected", onDisconnect);
    };
  }, [room]);

  const statusLabel = describeConnectionState(connectionState, disconnectReason);

  return (
    <View style={styles.container}>
      {remoteTrack && isTrackReference(remoteTrack) ? (
        <VideoTrack trackRef={remoteTrack} style={StyleSheet.absoluteFill} />
      ) : (
        <View style={[StyleSheet.absoluteFill, styles.waiting]}>
          {connectionState === ConnectionState.Connecting ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.waitingText}>Waiting for her camera...</Text>
          )}
        </View>
      )}
      <View style={styles.statusBar}>
        <Text style={styles.statusText}>{statusLabel}</Text>
      </View>
      <Pressable style={styles.hangUp} onPress={onHangUp}>
        <Text style={styles.hangUpText}>Hang up</Text>
      </Pressable>
    </View>
  );
}

export function describeConnectionState(state: ConnectionState, disconnectReason: string | null): string {
  switch (state) {
    case ConnectionState.Connecting:
      return "Connecting...";
    case ConnectionState.Connected:
      return "Connected";
    case ConnectionState.Reconnecting:
      return "Reconnecting...";
    case ConnectionState.Disconnected:
      return disconnectReason ? `Disconnected: ${disconnectReason}` : "Disconnected";
    default:
      return state;
  }
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "black" },
  waiting: { alignItems: "center", justifyContent: "center" },
  waitingText: { color: "white", fontSize: 16 },
  statusBar: {
    position: "absolute",
    top: 56,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  statusText: {
    color: "white",
    backgroundColor: "rgba(0,0,0,0.5)",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    overflow: "hidden",
  },
  hangUp: {
    position: "absolute",
    bottom: 48,
    alignSelf: "center",
    backgroundColor: "#d93025",
    paddingHorizontal: 28,
    paddingVertical: 14,
    borderRadius: 28,
  },
  hangUpText: { color: "white", fontWeight: "600", fontSize: 16 },
});
