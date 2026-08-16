import { connectAssistEvents } from "./events";
import type { PairingCredential } from "../storage/credentials";

class FakeSocket {
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  close = jest.fn(() => {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.onclose?.();
  });

  open(): void {
    this.onopen?.();
  }

  receive(payload: unknown): void {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }
}

const credential: PairingCredential = {
  gateway_url: "https://gateway.example",
  device_id: "helper-01",
  credential: "v1.helper-credential",
  expires_at: "2026-08-16T00:00:00Z",
};

// One microtask + timer flush: connect() does `await getCredential()` before
// touching createSocket, so a bare `await` after triggering it isn't always
// enough once fake timers are in play.
const flush = () => jest.advanceTimersByTimeAsync(0);

function harness() {
  const sockets: FakeSocket[] = [];
  const createSocket = jest.fn((_url: string, _headers: Record<string, string>) => {
    const socket = new FakeSocket();
    sockets.push(socket);
    return socket as unknown as WebSocket;
  });
  const getCredential = jest.fn(() => Promise.resolve(credential as PairingCredential | null));
  const onEvent = jest.fn();
  const onStateChange = jest.fn();
  return { sockets, createSocket, getCredential, onEvent, onStateChange };
}

beforeEach(() => {
  jest.useFakeTimers();
});

afterEach(() => {
  jest.useRealTimers();
});

describe("connectAssistEvents", () => {
  test("connects to the assist events URL with the credential as an Authorization header, not the URL", async () => {
    const h = harness();

    connectAssistEvents({ ...h });
    await flush();

    expect(h.createSocket).toHaveBeenCalledTimes(1);
    const [url, headers] = h.createSocket.mock.calls[0];
    expect(url).toBe("wss://gateway.example/v1/assist/events");
    expect(headers).toEqual({ Authorization: "Bearer v1.helper-credential" });
  });

  test("does not attempt to connect when there is no stored credential", async () => {
    const h = harness();
    h.getCredential.mockResolvedValue(null);

    connectAssistEvents({ ...h });
    await flush();

    expect(h.createSocket).not.toHaveBeenCalled();
    expect(h.onStateChange).toHaveBeenCalledWith("closed");
  });

  test("hello and keepalive frames never reach onEvent", async () => {
    const h = harness();
    connectAssistEvents({ ...h });
    await flush();
    const socket = h.sockets[0];
    socket.open();

    socket.receive({ schema_version: "1.0", type: "hello", occurred_at: "2026-08-16T00:00:00Z" });
    socket.receive({ schema_version: "1.0", type: "keepalive", occurred_at: "2026-08-16T00:00:01Z" });

    expect(h.onEvent).not.toHaveBeenCalled();
  });

  test("assist_requested/accepted/ended frames reach onEvent", async () => {
    const h = harness();
    connectAssistEvents({ ...h });
    await flush();
    const socket = h.sockets[0];
    socket.open();

    const requested = {
      schema_version: "1.0",
      type: "assist_requested",
      request_id: "req-1",
      session_id: "sess-1",
      occurred_at: "2026-08-16T00:00:00Z",
    };
    socket.receive(requested);

    expect(h.onEvent).toHaveBeenCalledWith(requested);
  });

  test("a missed keepalive forces the socket closed", async () => {
    const h = harness();
    connectAssistEvents({ ...h });
    await flush();
    const socket = h.sockets[0];
    socket.open();

    await jest.advanceTimersByTimeAsync(30_000);

    expect(socket.close).toHaveBeenCalled();
  });

  test("a message resets the keepalive timeout window", async () => {
    const h = harness();
    connectAssistEvents({ ...h });
    await flush();
    const socket = h.sockets[0];
    socket.open();

    await jest.advanceTimersByTimeAsync(20_000);
    socket.receive({ schema_version: "1.0", type: "keepalive", occurred_at: "2026-08-16T00:00:20Z" });
    await jest.advanceTimersByTimeAsync(20_000);

    // 40s of elapsed time with a reset at 20s: never 30s since a message.
    expect(socket.close).not.toHaveBeenCalled();
  });

  test("onclose schedules a reconnect after the initial backoff", async () => {
    const h = harness();
    connectAssistEvents({ ...h });
    await flush();
    const first = h.sockets[0];
    first.open();

    first.close();
    await jest.advanceTimersByTimeAsync(999);
    expect(h.createSocket).toHaveBeenCalledTimes(1);

    await jest.advanceTimersByTimeAsync(1);
    expect(h.createSocket).toHaveBeenCalledTimes(2);
  });

  test("backoff grows on repeated failures without an intervening open", async () => {
    const h = harness();
    connectAssistEvents({ ...h });
    await flush();

    // Fail three times in a row without ever reaching onopen.
    h.sockets[0].close(); // schedules a reconnect at 1000ms
    await jest.advanceTimersByTimeAsync(1000);
    expect(h.createSocket).toHaveBeenCalledTimes(2);

    h.sockets[1].close(); // schedules a reconnect at 2000ms
    await jest.advanceTimersByTimeAsync(1999);
    expect(h.createSocket).toHaveBeenCalledTimes(2);
    await jest.advanceTimersByTimeAsync(1);
    expect(h.createSocket).toHaveBeenCalledTimes(3);

    h.sockets[2].close(); // schedules a reconnect at 4000ms
    await jest.advanceTimersByTimeAsync(3999);
    expect(h.createSocket).toHaveBeenCalledTimes(3);
    await jest.advanceTimersByTimeAsync(1);
    expect(h.createSocket).toHaveBeenCalledTimes(4);
  });

  test("backoff resets to the initial delay after a successful open", async () => {
    const h = harness();
    connectAssistEvents({ ...h });
    await flush();

    h.sockets[0].close(); // failure #1: next reconnect scheduled at 1000ms
    await jest.advanceTimersByTimeAsync(1000);
    expect(h.createSocket).toHaveBeenCalledTimes(2);

    h.sockets[1].open(); // recovered -- backoff should reset to 1000ms
    h.sockets[1].close();
    await jest.advanceTimersByTimeAsync(999);
    expect(h.createSocket).toHaveBeenCalledTimes(2);
    await jest.advanceTimersByTimeAsync(1);
    expect(h.createSocket).toHaveBeenCalledTimes(3);
  });

  test("close() stops further reconnects", async () => {
    const h = harness();
    const connection = connectAssistEvents({ ...h });
    await flush();
    const socket = h.sockets[0];
    socket.open();

    connection.close();
    await jest.advanceTimersByTimeAsync(60_000);

    expect(h.createSocket).toHaveBeenCalledTimes(1);
  });

  test("close() closes the live socket", async () => {
    const h = harness();
    const connection = connectAssistEvents({ ...h });
    await flush();
    const socket = h.sockets[0];
    socket.open();

    connection.close();

    expect(socket.close).toHaveBeenCalled();
  });
});
