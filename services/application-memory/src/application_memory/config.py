"""Service configuration.

All settings come from the environment with a `VMA_` prefix. Settings are
frozen and validated at startup so a misconfiguration fails the process rather
than surfacing later as a wrong answer.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["dev", "ci", "deploy"]

MIN_SECRET_LENGTH = 32


def _env_file() -> str | None:
    """Load a local .env outside deploy only.

    A .env baked into a production image is the secret-handling anti-pattern
    docs/07-Privacy-and-Security.md forbids.
    """
    return None if os.getenv("VMA_ENVIRONMENT") == "deploy" else ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VMA_",
        env_file=_env_file(),
        extra="ignore",
        frozen=True,
    )

    service_name: str = "application-memory"
    environment: Environment = "dev"
    log_level: str = "INFO"

    # --- Storage ----------------------------------------------------------
    #: SQLite by default. docs/08 allows combining logical services "when
    #: hackathon simplicity or latency justifies it", and dropping a database
    #: container removes a volume, credentials, and a readiness dependency from
    #: day one on the GN100. Swapping to Postgres is this URL plus a migration.
    database_url: str = "sqlite+pysqlite:///./data/memory.db"
    #: Evidence lives outside the database so deletion is a subtree removal and
    #: the file store can later move to object storage unchanged.
    evidence_dir: Path = Path("./data/evidence")
    #: Registered crops outlive sessions and therefore must never share the
    #: session-scoped evidence root swept by retention.
    registration_crop_dir: Path = Path("./data/registration-crops")
    #: Largest evidence upload accepted, in bytes. `request.body()` reads the
    #: whole payload into memory, so without a cap one oversized POST -- a long
    #: clip, a mistake, or a deliberate one -- can exhaust the process. 25 MB
    #: is generous for a frame and enough for a short clip.
    max_evidence_bytes: int = Field(default=25_000_000, gt=0)
    #: Prefix for the `evidence_url` returned in answers. Left unset, answers
    #: carry a root-relative path (`/v1/evidence/{id}`), which is correct for a
    #: caller on the same origin and wrong for anyone else -- so set this to
    #: the address clients actually reach this service on once that differs.
    public_base_url: str | None = None

    # --- Object registry --------------------------------------------------
    max_registration_crop_bytes: int = Field(default=5_000_000, gt=0)
    registry_max_views_per_object: int = Field(default=8, ge=2, le=32)
    registry_max_embedding_dim: int = Field(default=4096, ge=1, le=16384)

    # --- Promotion --------------------------------------------------------
    #: Thresholds are configuration, not model constants: docs/04 requires the
    #: threshold set used for an evaluation run to be recorded, so /v1/status
    #: reports these.
    promote_min_event_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    promote_min_identity_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    #: A placement with no evidence can still be promoted, but the query layer
    #: refuses to call it `confirmed`. Turning this on refuses it outright.
    require_evidence_for_placement: bool = False

    # --- Retention --------------------------------------------------------
    #: docs/07: evidence and structured event metadata are session-scoped and
    #: deleted after 24 hours by default. Configurable, never silently longer.
    retention_hours: int = Field(default=24, ge=1)
    retention_sweep_interval_s: float = Field(default=900.0, gt=0)

    # --- Authorization ----------------------------------------------------
    internal_api_token: SecretStr | None = None
    #: Only these devices may write observations. Comma-separated, not JSON.
    device_id_allowlist: Annotated[tuple[str, ...], NoDecode] = ()

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("device_id_allowlist", mode="before")
    @classmethod
    def _split_allowlist(cls, value: object) -> object:
        """Accept a comma-separated string so the env var stays readable.

        `NoDecode` on the field is required: pydantic-settings JSON-decodes
        complex types straight from the environment before any validator runs,
        so without it the documented comma form fails and only JSON works.
        """
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                raise ValueError(
                    "device_id_allowlist is comma-separated, not JSON: "
                    'use VMA_DEVICE_ID_ALLOWLIST="glasses-01,glasses-02"'
                )
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @model_validator(mode="after")
    def _deploy_requires_hardening(self) -> Self:
        if self.environment != "deploy":
            return self
        missing: list[str] = []
        if self.internal_api_token is None:
            missing.append("internal_api_token")
        if not self.device_id_allowlist:
            missing.append("device_id_allowlist")
        if missing:
            raise ValueError(f"environment=deploy requires {', '.join(missing)}")
        return self

    @model_validator(mode="after")
    def _token_is_long_enough(self) -> Self:
        token = self.internal_api_token
        if token is not None and len(token.get_secret_value()) < MIN_SECRET_LENGTH:
            raise ValueError(f"internal_api_token must be at least {MIN_SECRET_LENGTH} characters")
        return self

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def sqlite_path(self) -> Path | None:
        """The database file, when the backend is SQLite.

        Used to create the parent directory at startup; a missing directory is
        otherwise reported as an opaque "unable to open database file".
        """
        if not self.is_sqlite:
            return None
        _, _, tail = self.database_url.partition(":///")
        return Path(tail) if tail and tail != ":memory:" else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
