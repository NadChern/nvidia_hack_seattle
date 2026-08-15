"""Engine and session construction.

Synchronous on purpose. The workload is a handful of small reads and writes per
observation against a local file; async SQLAlchemy would add `aiosqlite`, a
second execution model, and a class of "greenlet has no context" errors, in
exchange for concurrency this service does not need. FastAPI runs `def`
endpoints in a threadpool, so nothing blocks the event loop.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from application_memory.config import Settings
from application_memory.store.models import Base


def create_db_engine(settings: Settings) -> Engine:
    """Build an engine, creating the SQLite parent directory if needed."""
    path = settings.sqlite_path
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        settings.database_url,
        # SQLite refuses cross-thread use by default, and FastAPI's threadpool
        # hands each request a different thread. The pool still serialises
        # access, so this is safe here in a way it would not be with several
        # writer processes -- which is also why the Dockerfile pins one worker.
        connect_args={"check_same_thread": False} if settings.is_sqlite else {},
        future=True,
    )

    if settings.is_sqlite:
        # Registered for its side effect; pyright cannot see that a
        # decorator consumes it, hence the suppression.
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(  # pyright: ignore[reportUnusedFunction]
            dbapi_connection: Any, _: object
        ) -> None:
            cursor: Any = dbapi_connection.cursor()
            # WAL keeps a reader from blocking the writer, which matters as
            # soon as a query arrives while an observation is being ingested.
            cursor.execute("PRAGMA journal_mode=WAL")
            # Foreign keys are off by default in SQLite, which would silently
            # allow an observation to reference a session that never existed.
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def create_all(engine: Engine) -> None:
    """Create the schema directly.

    Used by tests and by first run in dev. Alembic owns schema evolution; this
    exists so a test does not need a migration run to get a table.
    """
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Generator[Session]:
    """A transaction that commits on success and rolls back on any exception."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = ["create_all", "create_db_engine", "create_session_factory", "session_scope"]
