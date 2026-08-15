from __future__ import annotations

import pytest
from conftest import confirmed_answer

from agent.config import Settings
from agent.listener import HandsFreeListener, Transcript, triggered_question
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
    def __init__(self, *, fail: bool = False, manual_trigger: bool = False) -> None:
        self.fail = fail
        self.manual_trigger = manual_trigger
        self.transcripts: list[dict[str, object]] = []
        self.replies: list[dict[str, object]] = []

    async def consume_manual_trigger(self, session_id: str) -> bool:
        del session_id
        if self.fail:
            raise RuntimeError("gateway unavailable")
        armed = self.manual_trigger
        self.manual_trigger = False
        return armed

    async def send_transcript(self, **event: object) -> None:
        if self.fail:
            raise RuntimeError("gateway unavailable")
        self.transcripts.append(event)

    async def send_reply(self, **event: object) -> None:
        if self.fail:
            raise RuntimeError("gateway unavailable")
        self.replies.append(event)


def transcript(text: str) -> Transcript:
    return Transcript(
        text=text,
        session_id="sess_01",
        epoch_id="TR_AUDIO_1",
        pts_samples_start=0,
        samples=16_000,
        sample_rate=16_000,
    )


@pytest.mark.parametrize(
    "text",
    [
        "where are my keys",
        "hey memory tell me a joke",
        "hey memories where are my keys",
        "hey memory",
    ],
)
def test_trigger_discipline_rejects_non_matching_speech(text: str) -> None:
    assert triggered_question(text, "hey memory") is None


@pytest.mark.parametrize(
    "text",
    [
        "hey memory remember my keys",
        "hey memory scan this wallet",
        "hey memory learn my bottle",
    ],
)
def test_trigger_accepts_registration_intents(text: str) -> None:
    assert triggered_question(text, "hey memory") is not None


def test_trigger_accepts_wake_prefix_and_where_question_without_punctuation() -> None:
    assert (
        triggered_question("Hey Memory, where did I leave my keys", "hey memory")
        == "where did i leave my keys"
    )


@pytest.mark.parametrize(
    "text",
    [
        "hey memory where did I leave my keys",
        "Hey Memory, where did I leave my keys?",
        "hey memory where's my wallet",
        "hey memory do you know where my keys are",
        "uh hey memory where did i leave my keys",
        "um, hey memory, where are my keys",
        "afterward hey memory where did i leave my keys",
        "on the living-room coffee table at 10:42. hey memory where did i leave my keys",
        "i have no record of the keys. hey memory where did I leave my keys",
        "so anyway hey memory where did i put my glasses",
    ],
)
def test_trigger_scans_for_a_wake_prefix_after_disfluency_or_echo(text: str) -> None:
    assert triggered_question(text, ("hey memory",)) is not None


@pytest.mark.parametrize(
    "text",
    [
        "On the living-room coffee table at 10:42, but they were picked up afterward.",
        "I have no record of the keys.",
        "I saw the wallet on the kitchen counter at 09:15.",
        "I cannot confirm where that is now.",
        "where did i leave my keys",
        "do you know where the coffee is",
        "i was just telling him where i left my badge",
        "hey memory is a cool name for the project",
        "the hey memory demo runs on the spark box",
        "hey, memory usage is climbing on the gpu",
        "can you tell me where the bathroom is",
        "hey there where did you go",
    ],
)
def test_question_shape_gate_rejects_adversarial_non_triggers(text: str) -> None:
    settings = Settings(environment="ci")

    assert triggered_question(text, settings.accepted_wake_prefixes) is None


def test_the_observed_glasses_mishearing_triggers() -> None:
    """Verbatim from the X3 Pro HUD, on a query that failed to trigger.

    Parakeet renders "memory" as two words on this microphone. The wake list
    exists precisely so a measured mishearing can be added without loosening
    the question-shape gate that keeps ordinary speech out of the model.
    """
    settings = Settings(environment="ci")
    heard = "Hey may me, where did I leave my ma monitor?"

    assert triggered_question(heard, settings.accepted_wake_prefixes) == (
        "where did i leave my ma monitor?"
    )


def test_a_wake_variant_without_a_question_still_does_not_fire() -> None:
    settings = Settings(environment="ci")

    assert (
        triggered_question("hey may me is a nice phrase", settings.accepted_wake_prefixes) is None
    )


@pytest.mark.parametrize(
    "text",
    [
        "hay memory where did i leave my keys",
        "he memory where did i leave my keys",
        "hey memories where did i leave my keys",
        "hey mammary where did i leave my keys",
    ],
)
def test_configured_stt_variants_trigger_the_same_bounded_question(text: str) -> None:
    settings = Settings(environment="ci")

    assert triggered_question(text, settings.accepted_wake_prefixes) == "where did i leave my keys"


async def test_non_triggering_transcript_stops_before_model_or_reply() -> None:
    backend = FakeBackend()
    reply = FakeReply()
    events = FakeEvents()
    listener = HandsFreeListener(Settings(environment="ci"), backend, reply, events)

    handled = await listener.process(transcript("I am just talking to someone"))

    assert not handled
    assert backend.calls == []
    assert reply.calls == []
    assert events.transcripts[0]["text"] == "I am just talking to someone"
    assert events.replies == []


async def test_wake_question_runs_guarded_core_and_sends_reply() -> None:
    backend = FakeBackend()
    reply = FakeReply()
    events = FakeEvents()
    listener = HandsFreeListener(Settings(environment="ci"), backend, reply, events)

    handled = await listener.process(transcript("um, hay memory, where are my keys?"))

    assert handled
    assert backend.calls == [("where are my keys?", "sess_01")]
    assert reply.calls == [("sess_01", confirmed_answer().spoken_answer)]
    assert events.replies[0]["question"] == "where are my keys?"
    assert events.replies[0]["answer_status"] == "confirmed"
    assert events.replies[0]["guard"] == "passed"


async def test_registration_workflow_owns_audio_without_a_duplicate_model_reply() -> None:
    backend = RegistrationBackend()
    reply = FakeReply()
    events = FakeEvents()
    listener = HandsFreeListener(Settings(environment="ci"), backend, reply, events)

    handled = await listener.process(transcript("hey memory remember my keys"))

    assert handled
    assert backend.calls == [("remember my keys", "sess_01")]
    assert reply.calls == []
    assert events.replies == []


async def test_transient_query_failure_does_not_stop_the_stt_message_stream() -> None:
    backend = FlakyBackend()
    reply = FakeReply()
    listener = HandsFreeListener(Settings(environment="ci"), backend, reply, FakeEvents())

    async def messages():  # type: ignore[no-untyped-def]
        yield transcript("Hey memory, where are my keys?").model_dump_json()
        yield transcript("Hey memory, where are my keys?").model_dump_json()

    await listener._consume_messages("sess_01", messages())

    assert backend.calls == [
        ("where are my keys?", "sess_01"),
        ("where are my keys?", "sess_01"),
    ]
    assert reply.calls == [("sess_01", confirmed_answer().spoken_answer)]


async def test_manual_trigger_accepts_one_bounded_where_question_without_wake() -> None:
    backend = FakeBackend()
    reply = FakeReply()
    events = FakeEvents(manual_trigger=True)
    listener = HandsFreeListener(Settings(environment="ci"), backend, reply, events)

    handled = await listener.process(transcript("where are my keys"))
    second = await listener.process(transcript("where is my wallet"))

    assert handled
    assert not second
    assert backend.calls == [("where are my keys", "sess_01")]


async def test_a_wake_prefix_survives_the_pause_before_the_question() -> None:
    """ "Hey memory." <pause> "where are my keys" is two utterances, not one.

    The VAD ends an utterance at any pause past its silence window, and a wake
    phrase invites exactly that pause. Reported from the glasses as being cut
    off before the sentence could be finished.
    """
    backend = FakeBackend()
    listener = HandsFreeListener(Settings(environment="ci"), backend, FakeReply(), FakeEvents())

    opened = await listener.process(transcript("hey memory"))
    answered = await listener.process(transcript("where are my keys"))

    assert not opened
    assert answered
    assert backend.calls == [("where are my keys", "sess_01")]


async def test_a_carried_wake_is_single_use() -> None:
    """One prefix answers one question; the next needs its own wake."""
    backend = FakeBackend()
    listener = HandsFreeListener(Settings(environment="ci"), backend, FakeReply(), FakeEvents())

    await listener.process(transcript("hey memory"))
    first = await listener.process(transcript("where are my keys"))
    second = await listener.process(transcript("where is my wallet"))

    assert first
    assert not second
    assert backend.calls == [("where are my keys", "sess_01")]


async def test_a_where_question_alone_never_fires() -> None:
    """The prefix gate is what keeps ordinary conversation out of the model."""
    backend = FakeBackend()
    listener = HandsFreeListener(Settings(environment="ci"), backend, FakeReply(), FakeEvents())

    handled = await listener.process(transcript("where did the coffee go"))

    assert not handled
    assert backend.calls == []


async def test_a_carried_wake_expires() -> None:
    """A prefix from minutes ago is not consent for a later question."""
    backend = FakeBackend()
    listener = HandsFreeListener(
        Settings(environment="ci", wake_carry_over_s=0.0), backend, FakeReply(), FakeEvents()
    )

    await listener.process(transcript("hey memory"))
    handled = await listener.process(transcript("where are my keys"))

    assert not handled
    assert backend.calls == []


async def test_manual_trigger_survives_a_leading_disfluency() -> None:
    """The press already established intent; a stray "um" must not waste it.

    SG-A measured that anchoring the question shape to the start of an
    utterance loses most real questions. This is the fallback for when the wake
    word fails, so it cannot fail the same way.
    """
    backend = FakeBackend()
    listener = HandsFreeListener(
        Settings(environment="ci"), backend, FakeReply(), FakeEvents(manual_trigger=True)
    )

    handled = await listener.process(transcript("um, where are my keys"))

    assert handled
    assert backend.calls == [("where are my keys", "sess_01")]


async def test_manual_trigger_still_refuses_an_unsupported_question() -> None:
    """Scanning is not permission to answer anything that follows a press."""
    backend = FakeBackend()
    events = FakeEvents(manual_trigger=True)
    listener = HandsFreeListener(Settings(environment="ci"), backend, FakeReply(), events)

    handled = await listener.process(transcript("what time is the keynote"))

    assert not handled
    assert backend.calls == []
    # The arm survives: an unrelated sentence must not spend the press, and
    # asking the gateway about every non-question is a round-trip per utterance.
    assert events.manual_trigger

    answered = await listener.process(transcript("um, where are my keys"))

    assert answered
    assert backend.calls == [("where are my keys", "sess_01")]


async def test_event_delivery_failure_never_blocks_the_guarded_audio_reply() -> None:
    backend = FakeBackend()
    reply = FakeReply()
    listener = HandsFreeListener(Settings(environment="ci"), backend, reply, FakeEvents(fail=True))

    handled = await listener.process(transcript("hey memory where are my keys"))

    assert handled
    assert backend.calls == [("where are my keys", "sess_01")]
    assert reply.calls == [("sess_01", confirmed_answer().spoken_answer)]
