"""Full memory reset for demos and manual testing.

Clears every persisted row and purges the on-disk evidence and registration
crops, then bumps the registry version so the vision gallery cache refreshes to
empty without a restart. Destructive, and behind the same internal-token gate
as the rest of the control surface -- the console's "Clear memory" button is
the only intended caller.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from application_memory.deps import (
    authorize_request,
    session_factory_of,
    settings_of,
)
from application_memory.store import repository

router = APIRouter(tags=["maintenance"])


def _purge_dir(path: Path) -> int:
    """Remove every child of ``path`` (files and subtrees), keeping the root.

    Missing root is a no-op -- a reset before anything was ever stored is still
    a successful reset.
    """
    if not path.exists():
        return 0
    removed = 0
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
        removed += 1
    return removed


@router.post("/v1/maintenance/reset")
def reset(request: Request) -> dict[str, Any]:
    """Delete all memory state and return the bumped registry version."""
    authorize_request(request)
    settings = settings_of(request)
    factory = session_factory_of(request)

    with factory() as db:
        version = repository.reset_all(db)
        db.commit()

    purged = _purge_dir(settings.evidence_dir) + _purge_dir(settings.registration_crop_dir)
    return {"reset": True, "registry_version": version, "purged_paths": purged}
