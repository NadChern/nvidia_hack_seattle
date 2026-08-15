"""Evidence frames on disk, addressed by id and verified by digest.

Files rather than database blobs, so deletion is a subtree removal and the
store can move to object storage without touching a row. Paths are assigned by
the server: docs/06 is explicit that clients must not submit local file paths,
and a client-supplied path is a directory traversal waiting to happen.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from application_memory.errors import InvalidRequestError


class EvidenceStore:
    """Session-scoped storage under one root directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def _relative(self, session_id: str, evidence_id: str) -> Path:
        # Both components are server-minted identifiers, never caller input, so
        # this cannot escape the root.
        return Path(session_id) / f"{evidence_id}.bin"

    def put(self, data: bytes, *, session_id: str, evidence_id: str, declared_sha256: str) -> Path:
        """Store bytes after checking they are what the caller said they were.

        A mismatch is refused rather than stored. Accepting it would leave an
        unverifiable frame behind a confident answer, which is precisely the
        "unsupported confident answer" docs/04 measures -- and the digest is
        the only thing that makes a stored frame evidence rather than an image.
        """
        actual = hashlib.sha256(data).hexdigest()
        if actual != declared_sha256.lower():
            raise InvalidRequestError(
                "evidence digest does not match the bytes received",
                declared=declared_sha256[:12],
                actual=actual[:12],
            )

        relative = self._relative(session_id, evidence_id)
        target = self._root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return relative

    def get(self, relative_path: str) -> bytes | None:
        """Read stored bytes, or None when the file is gone.

        Returning None rather than raising is deliberate: a missing file is a
        normal outcome after retention has run, and the query layer turns it
        into a weaker answer rather than an error.
        """
        target = self._root / relative_path
        if not target.is_file():
            return None
        return target.read_bytes()

    def exists(self, relative_path: str) -> bool:
        return (self._root / relative_path).is_file()

    def delete_session(self, session_id: str) -> int:
        """Remove every frame for a session, returning how many were deleted."""
        directory = self._root / session_id
        if not directory.is_dir():
            return 0
        count = sum(1 for path in directory.rglob("*") if path.is_file())
        shutil.rmtree(directory, ignore_errors=True)
        return count


__all__ = ["EvidenceStore"]
