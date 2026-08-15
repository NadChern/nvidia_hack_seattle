"""Transport-independent logic.

Nothing in this package imports `livekit`. That is deliberate: the sampler,
dimension guard, and epoch rules are the parts the S01 spike proved, and
keeping them free of the SDK is what makes them testable without a LiveKit
server or a network.
"""
