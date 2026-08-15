#!/usr/bin/env python3
"""Does the running LiveKit server accept the credentials we hold?

`check_listeners.py` answers where the server is listening. This answers
whether it will talk to us, which is a different question and fails in a much
more confusing way: the gateway mints a join token locally, the server rejects
it, and the browser reports "could not join the livekit room". Nothing in that
chain says "the credentials disagree" -- the gateway's token is perfectly
well-formed, it is just signed with a secret the server does not have.

That happens easily. A LiveKit container outlives the shell that started it,
so `docker compose up -d livekit` from one set of environment variables and a
gateway started later from another is a normal Sunday. Measured exactly that
way: a container 27 hours old holding `vma-dev` against a `.env` that had been
regenerated as `vma-dev-alex`.

Stdlib only, like its sibling -- HMAC-SHA256 and base64url are all a LiveKit
JWT needs, and this has to run before any virtualenv exists.

    python3 tools/dev-livekit/check_credentials.py

Reads VMA_LIVEKIT_API_KEY and VMA_LIVEKIT_API_SECRET. Exit 0 if the server
accepts them, 1 if it rejects them, 2 if it could not be asked.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:7880"
TIMEOUT_S = 5.0


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _token(api_key: str, api_secret: str) -> str:
    """A short-lived JWT granting only the right to list rooms.

    `roomList` and nothing else: this asks whether the credentials are
    accepted, so it should not be able to do anything if they are.
    """
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    claims = {
        "iss": api_key,
        "sub": api_key,
        # A few seconds of leeway for clock skew, and a minute of life -- this
        # token is used once, immediately.
        "nbf": now - 10,
        "exp": now + 60,
        "video": {"roomList": True},
    }
    signing_input = f"{_b64url(json.dumps(header).encode())}.{_b64url(json.dumps(claims).encode())}"
    signature = hmac.new(api_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(signature)}"


def main() -> int:
    api_key = os.environ.get("VMA_LIVEKIT_API_KEY", "")
    api_secret = os.environ.get("VMA_LIVEKIT_API_SECRET", "")
    if not api_key or not api_secret:
        print("VMA_LIVEKIT_API_KEY and VMA_LIVEKIT_API_SECRET are not set", file=sys.stderr)
        return 2

    base = os.environ.get("VMA_LIVEKIT_URL", DEFAULT_URL)
    # The gateway is configured with a WebSocket URL; the management API is the
    # same host over plain HTTP.
    base = base.replace("wss://", "https://").replace("ws://", "http://").rstrip("/")

    request = urllib.request.Request(
        f"{base}/twirp/livekit.RoomService/ListRooms",
        data=b"{}",
        headers={
            "Authorization": f"Bearer {_token(api_key, api_secret)}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            if response.status == 200:
                print(f"OK: LiveKit accepts the credentials in .env (key {api_key}).")
                return 0
            print(f"unexpected status {response.status} from LiveKit", file=sys.stderr)
            return 2
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            detail = error.read().decode("utf-8", "replace").strip()
            print(
                f"LiveKit rejected the credentials in .env (key {api_key}): {detail}\n"
                "\n"
                "The server is running with a different key/secret pair than the one\n"
                "the gateway will sign join tokens with, so every join returns 401 and\n"
                "the console reports 'could not join the livekit room'. A container\n"
                "outlives the shell that started it, which is usually how this happens.",
                file=sys.stderr,
            )
            return 1
        print(f"LiveKit returned {error.code}: {error.reason}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"could not reach LiveKit at {base}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
