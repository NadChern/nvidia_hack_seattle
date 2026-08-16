from __future__ import annotations

import pytest
from conftest import confirmed_answer

from agent.config import Settings
from agent.listener import HandsFreeListener, Transcript
from agent.stub import DraftAnswer

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def query(self, text: str, session_id: str | None) -> DraftAnswer:
        self.calls.append((text, session_id))
        result = confirmed_answer()
        return DraftAnswer(text=result.spoken_answer, tool_result=result)


class RegistrationBackend(FakeBackend):
    async def query(self, text: str, session_id: str | None) -> DraftAnswer:
        self.calls.append((text, session_id))
        return DraftAnswer(
            text="model text must not be spoken", tool_result=None, registration_started=True
        )


class FlakyBackend(FakeBackend):
    async def query(self, text: str, session_id: str | None) -> DraftAnswer:
        self.calls.append((text, session_id))
        if len(self.calls) == 1:
            raise RuntimeError("temporary Memory outage")
        result = confirmed_answer()
        return DraftAnswer(text=result.spoken_answer, tool_result=result)


class FakeReply:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def send(self, session_id: str, text: str) -> None:
        self.calls.append((session_id, text))


class FakeEvents:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.consume_calls = 0
        self.transcripts: list[dict[str, object]] = []
        self.replies: list[dict[str, object]] = []

    async def consume_manual_trigger(self, session_id: str) -> bool:
        del session_id
        self.consume_calls += 1
        return False

    async def send_transcript(self, **event: object) -> None:
        if self.fail:
            raise RuntimeError("gateway unavailable")
        self.transcripts.append(event)

    async def send_reply(self, **event: object) -> None:
        if self.fail:
            raise RuntimeError("gateway unavailable")
        self.replies.append(event)


def transcript(text: str, *, session_id: str = "sess_01") -> Transcript:
    return Transcript(
        text=text,
        session_id=session_id,
        epoch_id="TR_AUDIO_1",
        pts_samples_start=0,
        samples=16_000,
        sample_rate=16_000,
    )


@pytest.mark.parametrize(
    "heard",
    [
        "where is my keys",
        "Where's my keys?",
        "remember these keys",
        "could you tell me where I left my wallet",
    ],
)
async def test_every_completed_transcript_reaches_model_intent_router(heard: str) -> None:
    backend = FakeBackend()
    events = FakeEvents()
    listener = HandsFreeListener(Settings(environment="ci"), backend, FakeReply(), events)

    handled = await listener.process(transcript(heard))

    assert handled
    assert backend.calls == [(" ".join(heard.casefold().split()), "sess_01")]
    assert events.transcripts[0]["text"] == heard
    assert events.consume_calls == 0


async def test_wake_prefix_is_optional_and_is_not_stripped_before_model() -> None:
    backend = FakeBackend()
    listener = HandsFreeListener(Settings(environment="ci"), backend, FakeReply(), FakeEvents())

    await listener.process(transcript("Hey memory, where are my keys?"))

    assert backend.calls == [("hey memory, where are my keys?", "sess_01")]


async def test_registration_workflow_owns_audio_without_a_duplicate_model_reply() -> None:
    backend = RegistrationBackend()
    reply = FakeReply()
    events = FakeEvents()
    listener = HandsFreeListener(Settings(environment="ci"), backend, reply, events)

    handled = await listener.process(transcript("remember my keys"))

    assert handled
    assert backend.calls == [("remember my keys", "sess_01")]
    assert reply.calls == []
    assert events.replies == []


async def test_transient_query_failure_does_not_stop_the_stt_message_stream() -> None:
    backend = FlakyBackend()
    reply = FakeReply()
    listener = HandsFreeListener(Settings(environment="ci"), backend, reply, FakeEvents())

    async def messages():  # type: ignore[no-untyped-def]
        yield transcript("Where are my keys?").model_dump_json()
        yield transcript("Where are my keys?").model_dump_json()

    await listener._consume_messages("sess_01", messages())

    assert backend.calls == [
        ("where are my keys?", "sess_01"),
        ("where are my keys?", "sess_01"),
    ]
    assert reply.calls == [("sess_01", confirmed_answer().spoken_answer)]


async def test_socket_session_scope_rejects_a_mismatched_transcript() -> None:
    backend = FakeBackend()
    listener = HandsFreeListener(Settings(environment="ci"), backend, FakeReply(), FakeEvents())

    async def messages():  # type: ignore[no-untyped-def]
        yield transcript("Where are my keys?", session_id="sess_other").model_dump_json()

    await listener._consume_messages("sess_01", messages())

    assert backend.calls == []


async def test_event_delivery_failure_never_blocks_the_guarded_audio_reply() -> None:
    backend = FakeBackend()
    reply = FakeReply()
    listener = HandsFreeListener(Settings(environment="ci"), backend, reply, FakeEvents(fail=True))

    handled = await listener.process(transcript("where are my keys"))

    assert handled
    assert backend.calls == [("where are my keys", "sess_01")]
    assert reply.calls == [("sess_01", confirmed_answer().spoken_answer)]
