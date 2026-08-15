"""Long-lived registered-object crops, isolated from session retention.

`EvidenceStore` is intentionally session-scoped and swept after 24 hours.
Reference crops are the durable input to identity matching and VLM escalation,
so putting them there would silently erase every registration overnight.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from application_memory.errors import InvalidRequestError


class RegistrationCropStore:
    """Object-scoped crop storage under its own configured root."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def _relative(self, object_id: str, view_id: str) -> Path:
        return Path(object_id) / f"{view_id}.bin"

    def put(
        self,
        data: bytes,
        *,
        object_id: str,
        view_id: str,
        declared_sha256: str,
    ) -> Path:
        actual = hashlib.sha256(data).hexdigest()
        if actual != declared_sha256.lower():
            raise InvalidRequestError(
                "registration crop digest does not match the bytes received",
                declared=declared_sha256[:12],
                actual=actual[:12],
            )
        relative = self._relative(object_id, view_id)
        target = self._root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return relative

    def get(self, relative_path: str) -> bytes | None:
        target = self._root / relative_path
        return target.read_bytes() if target.is_file() else None

    def exists(self, relative_path: str) -> bool:
        return (self._root / relative_path).is_file()

    def delete(self, relative_path: str) -> bool:
        target = self._root / relative_path
        if not target.is_file():
            return False
        target.unlink()
        return True

    def delete_object(self, object_id: str) -> int:
        directory = self._root / object_id
        if not directory.is_dir():
            return 0
        count = sum(1 for path in directory.rglob("*") if path.is_file())
        shutil.rmtree(directory, ignore_errors=True)
        return count


__all__ = ["RegistrationCropStore"]
