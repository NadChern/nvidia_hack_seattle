"""SG-A2: does a scan-anywhere wake match recover recall without buying false fires?

Candidate change to triggered_question: instead of requiring the transcript to
START with the wake prefix, find the prefix anywhere in a bounded window and
keep the existing question-shape gate on what follows.
"""

import re
import sys

sys.path.insert(0, "services/agent/src")

from agent.listener import _QUESTION_SHAPE, triggered_question  # noqa: E402

WAKE = "hey memory"
VARIANTS = ("hey memory", "hay memory", "he memory", "hey memories", "hey mammary")


def baseline_startswith_match(text: str, wake_prefix: str = WAKE) -> str | None:
    """The pre-SG-A2 implementation, retained so the comparison is reproducible."""
    normalized = " ".join(text.casefold().split())
    prefix = " ".join(wake_prefix.casefold().split())
    if not normalized.startswith(prefix):
        return None
    if len(normalized) > len(prefix) and normalized[len(prefix)].isalnum():
        return None
    question = normalized[len(prefix) :].lstrip(" ,:;.!?-")
    if not question or _QUESTION_SHAPE.match(question) is None:
        return None
    return question


def scan_match(text: str, variants=(WAKE,)) -> str | None:
    """Wake prefix anywhere; question shape still gates what follows it."""
    normalized = " ".join(text.casefold().split())
    for variant in variants:
        prefix = " ".join(variant.casefold().split())
        start = 0
        while (hit := normalized.find(prefix, start)) != -1:
            end = hit + len(prefix)
            # Same boundary rule as today: no alnum glued to either side.
            before_ok = hit == 0 or not normalized[hit - 1].isalnum()
            after_ok = end == len(normalized) or not normalized[end].isalnum()
            if before_ok and after_ok:
                question = normalized[end:].lstrip(" ,:;.!?-")
                if question and _QUESTION_SHAPE.match(question) is not None:
                    return question
            start = hit + 1
    return None


SHOULD_FIRE = [
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
]

SHOULD_NOT_FIRE = [
    # assistant replies
    "On the living-room coffee table at 10:42, but they were picked up afterward.",
    "I have no record of the keys.",
    "I saw the wallet on the kitchen counter at 09:15.",
    "I cannot confirm where that is now.",
    # ordinary conversation on a hackathon floor
    "where did i leave my keys",
    "do you know where the coffee is",
    "i was just telling him where i left my badge",
    "hey memory is a cool name for the project",
    "the hey memory demo runs on the spark box",
    "hey, memory usage is climbing on the gpu",
    "can you tell me where the bathroom is",
    "hey there where did you go",
]

MISHEARD = [
    "hay memory where did i leave my keys",
    "he memory where did i leave my keys",
    "hey memories where did i leave my keys",
    "hey mammary where did i leave my keys",
]


def score(name, cases, expect, fn):
    hits = [(t, fn(t) is not None) for t in cases]
    good = sum(1 for _, f in hits if f is expect)
    print(f"  {name}: {good}/{len(hits)}")
    for text, f in hits:
        if f is not expect:
            print(f"      !! fire={f} {text[:70]}")
    return good, len(hits)


print("SG-A2 scan-anywhere wake match\n")

print("BASELINE (pre-SG-A2 startswith):")
score("recall  ", SHOULD_FIRE, True, baseline_startswith_match)
score("no-fire ", SHOULD_NOT_FIRE, False, baseline_startswith_match)
score("misheard", MISHEARD, True, baseline_startswith_match)

print("\nCANDIDATE (scan, exact prefix only):")
score("recall  ", SHOULD_FIRE, True, scan_match)
score("no-fire ", SHOULD_NOT_FIRE, False, scan_match)
score("misheard", MISHEARD, True, scan_match)

print("\nCANDIDATE (scan + misheard variant list):")
score("recall  ", SHOULD_FIRE, True, lambda t: scan_match(t, VARIANTS))
score("no-fire ", SHOULD_NOT_FIRE, False, lambda t: scan_match(t, VARIANTS))
score("misheard", MISHEARD, True, lambda t: scan_match(t, VARIANTS))

print("\nPRODUCTION:")
score("recall  ", SHOULD_FIRE, True, lambda t: triggered_question(t, VARIANTS))
score("no-fire ", SHOULD_NOT_FIRE, False, lambda t: triggered_question(t, VARIANTS))
score("misheard", MISHEARD, True, lambda t: triggered_question(t, VARIANTS))
