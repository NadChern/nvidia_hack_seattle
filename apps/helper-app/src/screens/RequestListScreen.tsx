import { useCallback, useEffect, useState } from "react";
import { ActivityIndicator, FlatList, Pressable, StyleSheet, Text, View } from "react-native";

import { api } from "../api";
import type { AssistRequest } from "../api/contract";
import { connectAssistEvents } from "../api/events";
import { GatewayRequestError } from "../api/client";

// Kept as a fallback even with the events WebSocket wired up below: a missed
// or delayed event (a reconnect still in backoff, a dropped frame) would
// otherwise leave the list stale with no other signal telling us so.
const POLL_INTERVAL_MS = 3000;

interface Props {
  onAccepted: (sessionId: string, serverUrl: string, token: string) => void;
}

// A 409 from /accept ("someone else got there first") or a 404/409 from
// /helper ("not accepted for this session anymore") are expected races, not
// failures -- surface them as a specific, calm message rather than the raw
// "POST ... failed: 409 ..." string a generic catch would show.
function describeAcceptError(e: unknown): string {
  if (e instanceof GatewayRequestError) {
    if (e.code === "conflict") {
      return "Someone else already accepted this request.";
    }
    if (e.code === "not_found") {
      return "This request is no longer available.";
    }
  }
  return e instanceof Error ? e.message : "Could not accept that request";
}

function relativeTime(iso: string): string {
  const diffS = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
  if (diffS < 60) {
    return "just now";
  }
  const diffM = Math.round(diffS / 60);
  if (diffM < 60) {
    return `${diffM}m ago`;
  }
  const diffH = Math.round(diffM / 60);
  return `${diffH}h ago`;
}

export function RequestListScreen({ onAccepted }: Props) {
  const [requests, setRequests] = useState<AssistRequest[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [acceptingId, setAcceptingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const list = await api.listRequests();
    const sorted = [...list].sort(
      (a, b) => new Date(b.requested_at).getTime() - new Date(a.requested_at).getTime(),
    );
    setRequests(sorted);
  }, []);

  useEffect(() => {
    load();
    const timer = setInterval(load, POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [load]);

  useEffect(() => {
    // assist_requested/accepted/ended all mean "the list changed" -- a full
    // reload is simpler and no less correct than patching one row in place,
    // and this fires at most a few times a minute.
    const connection = connectAssistEvents({ onEvent: load });
    return () => connection.close();
  }, [load]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  }, [load]);

  const onAccept = useCallback(
    async (sessionId: string) => {
      setAcceptingId(sessionId);
      setError(null);
      try {
        await api.acceptRequest(sessionId);
      } catch (e) {
        setError(describeAcceptError(e));
        setAcceptingId(null);
        await load(); // A 409 here means someone else took it -- refresh so it drops off the list.
        return;
      }
      try {
        const helperToken = await api.getHelperToken(sessionId);
        onAccepted(sessionId, helperToken.livekit_url, helperToken.token);
      } catch (e) {
        setError(describeAcceptError(e));
        setAcceptingId(null);
      }
    },
    [load, onAccepted],
  );

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Pending requests</Text>
      {error && <Text style={styles.error}>{error}</Text>}
      <FlatList
        testID="request-list"
        data={requests}
        keyExtractor={(item) => item.request_id}
        refreshing={refreshing}
        onRefresh={onRefresh}
        contentContainerStyle={requests.length === 0 ? styles.emptyContainer : undefined}
        ListEmptyComponent={<Text style={styles.empty}>No one needs help right now.</Text>}
        renderItem={({ item }) => (
          <View style={styles.row}>
            <View style={styles.rowText}>
              <Text style={styles.sessionId}>{item.session_id}</Text>
              <Text style={styles.time}>{relativeTime(item.requested_at)}</Text>
            </View>
            <Pressable
              testID={`accept-${item.session_id}`}
              style={styles.acceptButton}
              disabled={acceptingId !== null}
              onPress={() => onAccept(item.session_id)}
            >
              {acceptingId === item.session_id ? (
                <ActivityIndicator color="white" />
              ) : (
                <Text style={styles.acceptText}>Accept</Text>
              )}
            </Pressable>
          </View>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, paddingTop: 64, paddingHorizontal: 16 },
  header: { fontSize: 22, fontWeight: "600", marginBottom: 16 },
  error: { color: "#d93025", marginBottom: 12 },
  emptyContainer: { flex: 1, alignItems: "center", justifyContent: "center" },
  empty: { color: "#666" },
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#ccc",
  },
  rowText: { gap: 4 },
  sessionId: { fontSize: 16, fontWeight: "500" },
  time: { fontSize: 13, color: "#666" },
  acceptButton: {
    backgroundColor: "#1a73e8",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    minWidth: 72,
    alignItems: "center",
  },
  acceptText: { color: "white", fontWeight: "600" },
});
