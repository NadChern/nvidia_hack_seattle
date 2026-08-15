"""Shared media relay wire contract for the Visual Memory Assistant.

The Media Gateway owns the only LiveKit subscription and relays decoded,
dimension-guarded media over a local WebSocket. Consumers import this package
instead of touching WebRTC:

    from visual_memory_media_contract import MediaClient
    from visual_memory_media_contract.protocol import EpochStarted, VideoFrame

    async for message in MediaClient("ws://localhost:8080/v1/stream/video"):
        match message:
            case EpochStarted():
                tracker.reset(message.epoch_id)
            case VideoFrame():
                tracker.step(message.rgb, message.captured_at)

`docs/12-Media-Relay-Contract.md` is the normative protocol definition.

`fixtures` and `testing` are deliberately not imported here: they are only
needed by test code and keep the import cost of the runtime path down.
"""

from visual_memory_media_contract.client import MediaClient, MediaClientError, ReconnectPolicy
from visual_memory_media_contract.framing import decode_message, encode_message
from visual_memory_media_contract.protocol import PROTOCOL_VERSION, RelayMessage

__all__ = [
    "PROTOCOL_VERSION",
    "MediaClient",
    "MediaClientError",
    "ReconnectPolicy",
    "RelayMessage",
    "decode_message",
    "encode_message",
]
