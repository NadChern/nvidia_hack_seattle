import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Animated,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  Vibration,
  View,
} from "react-native";

import { api } from "../api";
import type { AssistRequest } from "../api/contract";
import { connectAssistEvents } from "../api/events";
import { GatewayRequestError } from "../api/client";
import { clearPairingCredential } from "../storage/credentials";

// Kept as a fallback even with the events WebSocket wired up below: a missed
// or delayed event (a reconnect still in backoff, a dropped frame) would
// otherwise leave the screen stale with no other signal telling us so.
const POLL_INTERVAL_MS = 3000;

// A short double-buzz, like a call notification, rather than one long
// continuous vibration -- this fires once per newly-seen request, not on
// every poll tick, so it should never feel like the phone is stuck buzzing.
const RING_VIBRATION_PATTERN = [0, 400, 250, 400];

interface Props {
  onAccepted: (sessionId: string, serverUrl: string, token: string) => void;
  onUnpaired: () => void;
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

export function RequestListScreen({ onAccepted, onUnpaired }: Props) {
  const [requests, setRequests] = useState<AssistRequest[]>([]);
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());
  const [refreshing, setRefreshing] = useState(false);
  const [accepting, setAccepting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ringingForId = useRef<string | null>(null);
  const pulse = useRef(new Animated.Value(1)).current;

  const load = useCallback(async () => {
    try {
      const list = await api.listRequests();
      const sorted = [...list].sort(
        (a, b) => new Date(b.requested_at).getTime() - new Date(a.requested_at).getTime(),
      );
      setRequests(sorted);
    } catch (e) {
      // The gateway's device-credential store is in-memory: any gateway
      // restart invalidates every previously paired device, and the phone
      // has no other way to find out except a 401 on its next poll. Treat
      // that specific failure as "go pair again," not a transient error.
      if (e instanceof GatewayRequestError && e.status === 401) {
        await clearPairingCredential();
        onUnpaired();
        return;
      }
      setError(e instanceof Error ? e.message : "Could not load requests");
    }
  }, [onUnpaired]);

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

  // The newest request that hasn't been declined on this device -- shown as
  // a single full-screen "incoming call," not a list. Declining is local
  // only (there's no reject endpoint yet): it hides the card here so the
  // next request, if any, can take over, but the request stays "pending"
  // server-side until someone else accepts it or it expires.
  const active = requests.find((r) => !dismissedIds.has(r.request_id)) ?? null;

  useEffect(() => {
    if (active && active.request_id !== ringingForId.current) {
      ringingForId.current = active.request_id;
      Vibration.vibrate(RING_VIBRATION_PATTERN);
    } else if (!active) {
      ringingForId.current = null;
    }
  }, [active]);

  useEffect(() => {
    if (!active) {
      return;
    }
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1.12, duration: 650, useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 1, duration: 650, useNativeDriver: true }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [active, pulse]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  }, [load]);

  const onAccept = useCallback(
    async (sessionId: string) => {
      setAccepting(true);
      setError(null);
      try {
        await api.acceptRequest(sessionId);
      } catch (e) {
        setError(describeAcceptError(e));
        setAccepting(false);
        await load(); // A 409 here means someone else took it -- refresh so it drops off.
        return;
      }
      try {
        const helperToken = await api.getHelperToken(sessionId);
        onAccepted(sessionId, helperToken.livekit_url, helperToken.token);
      } catch (e) {
        setError(describeAcceptError(e));
        setAccepting(false);
      }
    },
    [load, onAccepted],
  );

  const onDecline = useCallback(() => {
    if (!active) {
      return;
    }
    const { request_id } = active;
    setDismissedIds((prev) => {
      const next = new Set(prev);
      next.add(request_id);
      return next;
    });
  }, [active]);

  return (
    <FlatList
      testID="request-list"
      data={active ? [active] : []}
      keyExtractor={(item) => item.request_id}
      refreshing={refreshing}
      onRefresh={onRefresh}
      contentContainerStyle={styles.container}
      ListHeaderComponent={error ? <Text style={styles.error}>{error}</Text> : null}
      ListEmptyComponent={
        <View style={styles.emptyContainer}>
          <Text style={styles.empty}>No one needs help right now.</Text>
        </View>
      }
      renderItem={({ item }) => (
        <View style={styles.callCard}>
          <Animated.View style={[styles.ringBadge, { transform: [{ scale: pulse }] }]}>
            <Text style={styles.ringBadgeText}>📞</Text>
          </Animated.View>
          <Text style={styles.callTitle}>Your Parent Needs Help</Text>
          <Text style={styles.callSubtitle} numberOfLines={1} ellipsizeMode="middle">
            {item.session_id}
          </Text>
          <Text style={styles.callTime}>Requested {relativeTime(item.requested_at)}</Text>
          <View style={styles.actionsRow}>
            <Pressable
              testID={`decline-${item.session_id}`}
              style={[styles.actionButton, styles.declineButton]}
              disabled={accepting}
              onPress={onDecline}
            >
              <Text style={styles.declineText}>Decline</Text>
            </Pressable>
            <Pressable
              testID={`accept-${item.session_id}`}
              style={[styles.actionButton, styles.acceptButton]}
              disabled={accepting}
              onPress={() => onAccept(item.session_id)}
            >
              {accepting ? (
                <ActivityIndicator color="white" />
              ) : (
                <Text style={styles.acceptText}>Accept</Text>
              )}
            </Pressable>
          </View>
        </View>
      )}
    />
  );
}

const styles = StyleSheet.create({
  container: { flexGrow: 1, padding: 24, justifyContent: "center" },
  error: { color: "#d93025", marginBottom: 16, textAlign: "center" },
  emptyContainer: { alignItems: "center", justifyContent: "center", paddingVertical: 80 },
  empty: { color: "#666", fontSize: 16 },
  callCard: { alignItems: "center", gap: 8 },
  ringBadge: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: "#1a73e8",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 8,
  },
  ringBadgeText: { fontSize: 44 },
  callTitle: { fontSize: 26, fontWeight: "700", textAlign: "center" },
  callSubtitle: { fontSize: 15, color: "#666", maxWidth: "90%" },
  callTime: { fontSize: 13, color: "#999", marginBottom: 24 },
  actionsRow: { flexDirection: "row", gap: 16, width: "100%", paddingHorizontal: 24 },
  actionButton: {
    flex: 1,
    paddingVertical: 18,
    borderRadius: 32,
    alignItems: "center",
    justifyContent: "center",
  },
  acceptButton: { backgroundColor: "#1e8e3e" },
  declineButton: { backgroundColor: "#fce8e6", borderWidth: 1, borderColor: "#d93025" },
  acceptText: { color: "white", fontWeight: "700", fontSize: 18 },
  declineText: { color: "#d93025", fontWeight: "700", fontSize: 18 },
});
