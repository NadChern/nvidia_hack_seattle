#!/usr/bin/env python3
"""Run the demo scenario against a local database and print the answer.

This is fallback levels 4 and 5 from docs/03, working with no camera, no
vision model, and no network: validated observations go in, the reducer runs,
and the assistant's answer comes out.

    uv run python scripts/seed_demo.py
    uv run python scripts/seed_demo.py --confirmed   # the undisturbed case

The scenario ends with the keys being picked up, so the correct answer names
the coffee table *and* says they are no longer there. **A `confirmed` answer
here is a bug**, and the script exits non-zero if it sees one -- which makes
this a rehearsal and a regression test at the same time.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from visual_memory_memory_contract.fixtures import (
    SESSION,
    keys_placed_and_left,
    keys_placed_then_picked_up,
)

from application_memory.config import Settings
from application_memory.domain.answers import EvidenceRef, answer_for
from application_memory.domain.reducer import PromotionPolicy
from application_memory.evidence.store import EvidenceStore
from application_memory.store import models, repository
from application_memory.store.engine import (
    create_all,
    create_db_engine,
    create_session_factory,
    session_scope,
)

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
OFF = "\033[0m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmed",
        action="store_true",
        help="run the undisturbed scenario, which may legitimately answer 'confirmed'",
    )
    parser.add_argument(
        "--database",
        default="sqlite+pysqlite:///./data/demo.db",
        help="where to write; defaults to a throwaway demo database",
    )
    args = parser.parse_args(argv)

    settings = Settings(
        environment="dev",
        database_url=args.database,
        evidence_dir=Path("./data/demo-evidence"),
    )
    engine = create_db_engine(settings)
    create_all(engine)
    factory = create_session_factory(engine)
    store = EvidenceStore(settings.evidence_dir)
    policy = PromotionPolicy(
        min_event_confidence=settings.promote_min_event_confidence,
        min_identity_confidence=settings.promote_min_identity_confidence,
    )

    observations = list(keys_placed_and_left() if args.confirmed else keys_placed_then_picked_up())

    with session_scope(factory) as db:
        # Start from nothing so repeated runs are identical rather than
        # accumulating a longer and longer timeline.
        repository.delete_session(db, SESSION)

    print(f"\n{BOLD}What the vision service reported{OFF}")
    with session_scope(factory) as db:
        for observation in observations:
            # Real evidence, so a `confirmed` answer is actually supportable.
            # Without a loadable frame the query layer downgrades the claim --
            # correctly, but it would obscure what this script is showing.
            frame = f"demo-frame-{observation.observation_id}".encode()
            digest = hashlib.sha256(frame).hexdigest()
            evidence = observation.evidence[0] if observation.evidence else None
            if evidence is not None:
                store.put(
                    frame,
                    session_id=SESSION,
                    evidence_id=evidence.evidence_id,
                    declared_sha256=digest,
                )
                db.add(
                    models.EvidenceRow(
                        evidence_id=evidence.evidence_id,
                        session_id=SESSION,
                        sha256=digest,
                        media_type="image/jpeg",
                        relative_path=str(Path(SESSION) / f"{evidence.evidence_id}.bin"),
                        size_bytes=len(frame),
                        created_at=repository.utcnow(),
                    )
                )

            result = repository.record_observation(db, observation, policy=policy)
            where = ""
            if observation.location and observation.location.surface:
                where = f" {observation.location.relation} the {observation.location.surface}"
            mark = "promoted" if result.promoted else "history only"
            clock = observation.event.occurred_at.astimezone().strftime("%H:%M:%S")
            print(
                f"  {clock}  {observation.event.action:<10}"
                f"{observation.object.label}{where}  {DIM}({mark}){OFF}"
            )

    print(f"\n{BOLD}Asked: where are my keys?{OFF}")
    with session_scope(factory) as db:
        matches = repository.find_objects_by_label(db, "keys")
        state = repository.state_of(db, matches[0]) if matches else None
        reference = None
        if state is not None and state.last_confirmed_placement is not None:
            evidence_id = state.last_confirmed_placement.evidence_id
            if evidence_id is not None:
                row = db.get(models.EvidenceRow, evidence_id)
                if row is not None and store.exists(row.relative_path):
                    reference = EvidenceRef(
                        evidence_id=evidence_id,
                        url=f"/v1/evidence/{evidence_id}",
                        media_type=row.media_type,
                    )
        answer = answer_for(state, label="keys", evidence=reference)

    print(f"\n  {BOLD}{answer.spoken_answer}{OFF}\n")
    print(f"  {DIM}answer_status  {OFF}{answer.answer_status}")
    print(f"  {DIM}current        {OFF}{answer.current_status}")
    if answer.last_confirmed_placement:
        placement = answer.last_confirmed_placement
        print(
            f"  {DIM}last confirmed {OFF}{placement.relation} the {placement.surface} "
            f"at {placement.occurred_at.astimezone().strftime('%H:%M')}"
        )
        print(f"  {DIM}evidence       {OFF}{placement.evidence_url or placement.evidence_id}")
        print(f"  {DIM}media type     {OFF}{placement.evidence_media_type or '-'}")

    engine.dispose()

    expected = "confirmed" if args.confirmed else "last_confirmed_only"
    if answer.answer_status != expected:
        print(f"\n{RED}FAIL{OFF} expected {expected}, got {answer.answer_status}\n")
        return 1
    print(f"\n{GREEN}OK{OFF}   answer_status is {expected}, as it must be\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
