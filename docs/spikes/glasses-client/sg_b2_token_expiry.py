"""SG-B2: is LiveKit token expiry enforced at join, and with how much leeway?

SG-B saw a token 3s past a 1s TTL still join. That is either JWT clock-skew
leeway or expiry not being enforced at all, and the two have very different
consequences for the plan's token-refresh design. This separates them.
"""

import asyncio
import datetime as dt
import os

from livekit import api, rtc

URL = "ws://127.0.0.1:7880"
ROOM = "sg-b2-expiry"
KEY = os.environ["VMA_LIVEKIT_API_KEY"]
SECRET = os.environ["VMA_LIVEKIT_API_SECRET"]


def token(identity, ttl_s):
    return (
        api.AccessToken(KEY, SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_ttl(dt.timedelta(seconds=ttl_s))
        .with_grants(
            api.VideoGrants(room_join=True, room=ROOM, can_publish=True, can_subscribe=True)
        )
        .to_jwt()
    )


async def try_join(identity, ttl_s, wait_s):
    tok = token(identity, ttl_s)
    await asyncio.sleep(wait_s)
    room = rtc.Room()
    try:
        await asyncio.wait_for(room.connect(URL, tok), timeout=15)
        await room.disconnect()
        return "JOINED"
    except Exception as exc:  # noqa: BLE001
        return f"refused ({type(exc).__name__})"


async def main():
    # ttl, seconds waited past issue -> seconds past expiry
    cases = [(2, 0), (2, 10), (2, 30), (2, 90), (2, 300)]
    for ttl, wait in cases:
        result = await try_join(f"exp-{ttl}-{wait}", ttl, wait)
        print(f"ttl={ttl}s waited={wait}s (expired {max(0, wait - ttl)}s ago): {result}", flush=True)


asyncio.run(main())
