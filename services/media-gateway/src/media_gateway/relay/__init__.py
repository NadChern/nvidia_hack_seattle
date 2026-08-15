"""Fan-out of decoded media to local subscribers.

The gateway encodes each sampled frame once and hands the same immutable bytes
to every subscriber. See `docs/12-Media-Relay-Contract.md`.
"""
