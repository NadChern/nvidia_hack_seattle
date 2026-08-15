// Reconnecting client for WS /v1/assist/events. Two failure modes matter
// here that a plain WebSocket does not handle on its own: the gateway
// sleeping or restarting (an ordinary close, reconnect with backoff), and a
// connection that looks open but is dead -- no TCP close ever arrives, so a
// missed keepalive is treated as the same signal. See
// role-prompts/Erin-Remote-Assist.md's "Conditions of defensive programming".
import type { AssistEvent } from "./contract";
import { getPairingCredential, type PairingCredential } from "../storage/credentials";

const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 30_000;
// The gateway's own keepalive interval (media_gateway.config.Settings.ws_keepalive_s)
// defaults to 10s and is not exposed to this app. 3x that default leaves room
// for a slow network without waiting anywhere near as long as a TCP timeout.
const KEEPALIVE_TIMEOUT_MS = 30_000;

export type AssistConnectionState = "connecting" | "open" | "closed";

// Matches the constructor React Native's WebSocket accepts (a `headers`
// option is a React Native extension -- a browser's WebSocket cannot set
// custom headers at all, which is also why authorize_helper_websocket on the
// gateway refuses a device credential from anywhere but that header).
type SocketFactory = (url: string, headers: Record<string, string>) => WebSocket;

// The DOM lib's WebSocket type (what @types resolves to here) only declares
// the 2-argument web constructor; React Native's is a different runtime
// object entirely, whose 3rd-argument `headers` option isn't in that type.
const RNWebSocket = WebSocket as unknown as new (
  url: string,
  protocols: undefined,
  options: { headers: Record<string, string> },
) => WebSocket;

const defaultCreateSocket: SocketFactory = (url, headers) => new RNWebSocket(url, undefined, { headers });

interface Options {
  onEvent: (event: Exclude<AssistEvent, { type: "hello" | "keepalive" }>) => void;
  onStateChange?: (state: AssistConnectionState) => void;
  getCredential?: () => Promise<PairingCredential | null>;
  createSocket?: SocketFactory;
}

export interface AssistEventsConnection {
  close: () => void;
}

function toWebSocketUrl(gatewayUrl: string): string {
  return `${gatewayUrl.replace(/^http/, "ws")}/v1/assist/events`;
}

export function connectAssistEvents({
  onEvent,
  onStateChange,
  getCredential = getPairingCredential,
  createSocket = defaultCreateSocket,
}: Options): AssistEventsConnection {
  let closedByCaller = false;
  let socket: WebSocket | null = null;
  let keepaliveTimer: ReturnType<typeof setTimeout> | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let backoffMs = INITIAL_BACKOFF_MS;

  const resetKeepaliveTimer = () => {
    if (keepaliveTimer) {
      clearTimeout(keepaliveTimer);
    }
    keepaliveTimer = setTimeout(() => {
      // Nothing since the last message, including a keepalive: the socket
      // may still look open while the gateway is long gone. Force a close so
      // the same reconnect path onclose already handles picks this up.
      socket?.close();
    }, KEEPALIVE_TIMEOUT_MS);
  };

  const clearTimers = () => {
    if (keepaliveTimer) {
      clearTimeout(keepaliveTimer);
      keepaliveTimer = null;
    }
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  };

  const scheduleReconnect = () => {
    if (closedByCaller) {
      return;
    }
    reconnectTimer = setTimeout(connect, backoffMs);
    backoffMs = Math.min(backoffMs * 2, MAX_BACKOFF_MS);
  };

  const connect = async () => {
    if (closedByCaller) {
      return;
    }
    const credential = await getCredential();
    if (!credential || closedByCaller) {
      // Not paired (yet, or anymore): nothing to connect to. The caller
      // decides whether to retry connectAssistEvents once pairing completes.
      onStateChange?.("closed");
      return;
    }

    onStateChange?.("connecting");
    socket = createSocket(toWebSocketUrl(credential.gateway_url), {
      Authorization: `Bearer ${credential.credential}`,
    });

    socket.onopen = () => {
      backoffMs = INITIAL_BACKOFF_MS;
      onStateChange?.("open");
      resetKeepaliveTimer();
    };

    socket.onmessage = (message) => {
      resetKeepaliveTimer();
      let event: AssistEvent;
      try {
        event = JSON.parse(String(message.data)) as AssistEvent;
      } catch {
        return; // Malformed frame: drop it, the next keepalive still arrives.
      }
      if (event.type === "hello" || event.type === "keepalive") {
        return;
      }
      onEvent(event);
    };

    socket.onclose = () => {
      if (keepaliveTimer) {
        clearTimeout(keepaliveTimer);
        keepaliveTimer = null;
      }
      onStateChange?.("closed");
      scheduleReconnect();
    };

    socket.onerror = () => {
      socket?.close();
    };
  };

  connect();

  return {
    close: () => {
      closedByCaller = true;
      clearTimers();
      socket?.close();
    },
  };
}
