"""SG-A: wake-prefix robustness against echo and STT variation.

Pure function spike against agent.listener.triggered_question. No hardware, no
LiveKit, no models. Answers two questions the plan currently assumes:

  1. Can the assistant's own reply audio, transcribed back, self-trigger?
  2. What happens to a legitimate wake when reply audio lands at the head of
     the same utterance?
"""

import sys

sys.path.insert(0, "services/agent/src")

from agent.listener import triggered_question  # noqa: E402

WAKE = "hey memory"


def fires(text: str) -> bool:
    return triggered_question(text, WAKE) is not None


CLEAN = [
    "hey memory where did I leave my keys",
    "Hey Memory, where did I leave my keys?",
    "hey memory. where did i leave my glasses",
    "hey memory where's my wallet",
    "hey memory do you know where my keys are",
    "hey memory could you tell me where the remote is",
]

# Assistant replies, shaped like what the guard actually emits.
REPLIES = [
    "On the living-room coffee table at 10:42, but they were picked up afterward.",
    "I have no record of the keys.",
    "I saw the wallet on the kitchen counter at 09:15.",
    "I cannot confirm where that is now.",
]

# Reply audio bleeding into the head of the wearer's next utterance.
ECHO_PREFIXED = [f"{r.casefold()} {c}" for r in REPLIES[:2] for c in CLEAN[:2]]

# Partial echo: only the tail of the reply survives the VAD boundary.
PARTIAL_ECHO = [
    "afterward hey memory where did i leave my keys",
    "table at ten forty two hey memory where did i leave my keys",
    "uh hey memory where did i leave my keys",
    "um, hey memory, where are my keys",
]

# Plausible Parakeet mishearings of the wake prefix itself.
MISHEARD = [
    "hay memory where did i leave my keys",
    "he memory where did i leave my keys",
    "hey memories where did i leave my keys",
    "hey memory, where did i leave my keys",  # control: should fire
    "a memory where did i leave my keys",
    "hey mammary where did i leave my keys",
]


def report(name, cases, expect):
    hits = [(t, fires(t)) for t in cases]
    good = sum(1 for _, f in hits if f is expect)
    print(f"\n{name}  ({good}/{len(hits)} as expected, expect fire={expect})")
    for text, f in hits:
        mark = "ok " if f is expect else "!! "
        print(f"  {mark}fire={str(f):5} {text[:78]}")
    return good, len(hits)


print("SG-A wake-prefix robustness")
print(f"wake_prefix = {WAKE!r}")
report("1. clean wearer questions", CLEAN, True)
report("2. assistant replies alone (self-trigger?)", REPLIES, False)
report("3. reply audio prefixed to a real wake", ECHO_PREFIXED, True)
report("4. partial echo / disfluency before wake", PARTIAL_ECHO, True)
report("5. misheard wake prefix", MISHEARD, True)
