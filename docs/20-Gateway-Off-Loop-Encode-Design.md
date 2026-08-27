# Gateway Off-Loop Encode + rgba_raw Fail-Loud — Design

**Status:** built. This records the design and the decision that changed during
implementation (a simpler design than first proposed — see "Design chosen").

**Purpose.** Gate item #5 in `docs/19-Post-Spike-Build-Plan.md` has two remaining gateway parts,
deferred out of PR #9 (which shipped the glasses H.265 publish, the load-bearing half):

- **15b** — move the gateway's JPEG encode **off the asyncio event loop**.
- **15d** — make an **`rgba_raw`** subscriber **fail loudly** instead of silently dropping ~2 frames
  in 3 at full resolution.

15b was attempted during PR #9 and reverted: the naive change drops frames at epoch teardown and
turned a synthetic test to zero relayed frames. This document records *why*, the constraint that
makes it non-trivial, the caller-analysis fact that makes it tractable, and the proposed design —
so the next attempt starts from the diagnosis rather than rediscovering it.

---

## Background: what runs today

The sampler coroutine (`pipeline.py:_sample`) paces at `sample_fps`, takes the newest frame from a
latest-wins slot, and calls `_relay_video` **synchronously**:

```
while True:
    await pacer.wait()          # the ONLY suspension point
    frame = slot.take()
    ...
    self._relay_video(...)      # synchronous: encode + publish, no await
```

`_relay_video` loops over `required_video_encodings()` and calls `encode_video` (`relay/codec.py`)
inline. For `jpeg` that is a PIL `convert("RGB").save(...)` — several ms at 720p, on the event loop,
blocking every other subscriber, control message, and audio chunk while it runs. That blocking is
what 15b removes.

Epoch teardown is driven by `_finish_epoch` (sync): it `task.cancel()`s the sampler, flushes audio,
and emits `track_lost` + `EpochEnded`. `EpochEnded` is **the last word** on an epoch's stream — a
contract enforced by `tests/test_lifecycle.py` (`test_epoch_ended_is_the_last_message`) and relied
on by consumers that reset tracker state on epoch boundaries (`docs/12`).

---

## The 15b root cause

Because `_relay_video` is synchronous, **take → publish is atomic with respect to the event loop**:
the only place the sampler yields is `await pacer.wait()`, where no frame is in hand. That is exactly
why `_finish_epoch`'s hard `task.cancel()` is safe today — cancellation can only land at the pacer
wait.

Moving the encode off-loop (`await asyncio.to_thread(encode_video, …)`) inserts a **suspension point
between take and publish**. A hard cancel at that point drops the in-flight frame.

- **In production:** one frame, at the real end of a continuous track. Immaterial.
- **In the synthetic test** (`test_rgba_raw_subscribers_get_pixel_exact_frames`): **zero** frames
  survive. `immediate_pacer` runs the sampler in lockstep with the scripted source (a `sleep(0)`
  pacer); the new `await` breaks the lockstep, so the source races ahead to end the epoch before any
  frame publishes. `assert frames` then fails.

So the failure is real but its *magnitude* in the test is a lockstep artifact — it is not evidence
of a production frame-loss bug, it is evidence that off-loop encode is incompatible with hard-cancel
teardown.

---

## The constraint

"EpochEnded is the last word" + off-loop (async) encode ⟹ the in-flight frame must be **drained and
published before EpochEnded is emitted** ⟹ the terminal emission can no longer happen synchronously
inside a hard cancel. It must become async/deferred. This is the ripple that made 15b look large.

## The fact that makes it tractable

Tracing every caller of the two teardown entry points:

| Sink method | Callers | Context | Must stay sync? |
|---|---|---|---|
| `epoch_ended` | `room_worker._on_track_unsubscribed`, `_on_participant_disconnected` | **sync LiveKit event callbacks** (`room.on(...)`) | **Yes** |
| `session_ended` | `api/sessions.py` (DELETE endpoint), `transport/scripted.py` | **async** | **No** |

`epoch_ended` is pinned synchronous by the LiveKit SDK's sync event dispatch. But `session_ended` is
called only from async contexts and is **free to become a coroutine**. That asymmetry is exactly
enough for a clean design: the per-epoch terminal work moves onto the sampler (which is already an
async task), and only the session-level ordering needs an `await`, which `session_ended` can now do.

---

## Rejected design — "the sampler owns its terminal emission"

The first proposal made `_finish_epoch` set a cooperative `stop_requested` flag and let the sampler
drain the last frame and emit `EpochEnded` itself, with `session_ended` going async to await the
samplers before `SessionEnded`. It preserves the tail frame, but implementation surfaced a **reconnect
ordering hazard**: on the "new track SID = new epoch" reconnect (the spike's central finding), a
`track_unsubscribed(old)` immediately followed by `track_subscribed(new)` would publish the new epoch's
`EpochStarted` **while the old epoch's sampler is still draining** — so a consumer could see
`EpochStarted(new)` *before* `EpochEnded(old)`, reversing the very boundary it resets tracker state on.
Closing that hole means forcing the prior epoch's synchronous teardown at `epoch_started` time, which
adds exactly the moving parts the simpler design below avoids. Rejected: it trades a real ordering
guarantee for an immaterial tail frame.

## Design chosen — injectable encode runner, teardown unchanged

**15b is just: move the encode to a thread, and keep `EpochEnded` synchronous exactly as today.**

1. `MediaPipeline` takes an injectable `encode_runner`, defaulting to `asyncio.to_thread`.
   `_relay_video` becomes `async` and does `await self._encode_runner(encode_video, …)` per required
   encoding. PIL releases the GIL across the libjpeg encode, so it genuinely overlaps the loop.

2. **Teardown is untouched.** `_finish_epoch` still hard-`cancel()`s the sampler and emits `EpochEnded`
   synchronously. If a cancel lands during the off-loop encode, `CancelledError` is raised at the
   `await` **before** `publish_video`, so the in-flight frame is dropped rather than published after
   `EpochEnded`. **Ordering is preserved in every case** — graceful end, session end, displaced-epoch
   reconnect, and shutdown — with no new code, because a cancelled task simply never reaches the
   publish. The only cost is one frame at a real track end, which is immaterial on a continuous stream.

3. **Tests stay deterministic.** `asyncio.to_thread` composes badly with the lockstep `immediate_pacer`
   (a real thread hop lets the scripted source race ahead of the sampler, which drove the earlier
   "zero frames" failure). The test harness injects an **inline runner** — `async def f(fn,*a,**k):
   return fn(*a,**k)` — which awaits without an inner suspension, so it does not yield to the loop and
   the existing lockstep holds unchanged. A guard test asserts the production default really is
   `asyncio.to_thread`, so a refactor can't silently put the encode back on the loop.

No `session_ended` signature change, no sink-protocol change, no touch to the synchronous LiveKit
ingress. The whole of 15b is the runner injection plus making `_relay_video`/`_sample` `await`.

**Measured (720p, quality 92, `docs/20` bench):** the longest single unbroken event-loop block per
frame drops from **13.7 ms → 6.1 ms**, and the full ~8.5 ms encode no longer blocks the loop on every
one of the 90 frames — only PIL's GIL-held setup remains, and that shrinks further under real I/O load
than under the benchmark's hot spin.

**Tests:** the existing pipeline suite (including the pixel-exact `test_rgba_raw…`) passes unchanged,
because the inline runner keeps relay in lockstep — the design needed no test rewrites, only the harness
runner injection. Added: a guard test that the default runner is `asyncio.to_thread` (`test_encode_is
_off_loaded_from_the_event_loop_by_default`).

---

## Proposed design — 15d (separate, small, independent)

`rgba_raw` exists for pixel-exact work (`relay/codec.py` passes the buffer through untouched). At
720p that is 3.7 MB/frame; the socket cannot flush it at sample rate, so `Subscriber.offer_frame`
latest-wins-evicts ~2 of every 3 — silently corrupting the one use case rgba_raw serves.

**Change (built):** in `offer_frame`, when `encoding == "rgba_raw"` and a pending frame would be
evicted, **close the subscriber loudly** with a dedicated `RGBA_RAW_BACKPRESSURE` reason instead of
dropping — the same shape as the audio path (`hub.py`, `AUDIO_BACKPRESSURE`), which surfaces to the
websocket as a close code + reason (`api/stream.py`). A pixel-exact consumer then gets a failure it
notices, not corrupt data, and `dropped` is *not* incremented (a closed-loud frame is not a silent
drop). JPEG subscribers are unaffected: for lossy preview, latest-wins is correct.

This touches only `relay/hub.py`, has no teardown-ordering interaction, and landed as its own commit
first. Covered by `test_rgba_raw_subscriber_is_closed_rather_than_silently_dropped`.

---

## What shipped

One PR, two commits: **15d first** (rgba_raw fail-loud, `relay/hub.py`), then **15b** (off-loop encode
via the injectable runner, `pipeline.py`). Together they close the gateway half of gate item #5. Full
gateway suite green (`uv run pytest`), `ruff` and `pyright` clean.

## Not done (deliberately)

- A shared encode thread-pool / executor abstraction. A per-call `asyncio.to_thread` was enough; a pool
  is an optimization to justify separately with a measurement that shows the default executor binds.
- Relaxing the "EpochEnded is the last word" contract. The chosen design preserves it by construction.
- Any change to the synchronous LiveKit ingress callbacks or the `session_ended` signature.
