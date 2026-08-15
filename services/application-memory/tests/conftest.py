"""Shared fixtures.

Tests run against a real SQLite file rather than an in-memory database. The
difference matters: SQLite round-trips datetimes through strings and hands them
back naive, so an in-memory-only suite would miss the whole class of
naive-versus-aware comparison failures that only appear once data has actually
been stored.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from application_memory.config import Settings
from application_memory.store.engine import create_all, create_db_engine, create_session_factory


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="ci",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'memory.db'}",
        evidence_dir=tmp_path / "evidence",
        internal_api_token=None,
    )


@pytest.fixture
def engine(settings: Settings) -> Iterator[Engine]:
    engine = create_db_engine(settings)
    create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def sessions(engine: Engine) -> sessionmaker[DbSession]:
    return create_session_factory(engine)


@pytest.fixture
def db(sessions: sessionmaker[DbSession]) -> Iterator[DbSession]:
    session = sessions()
    try:
        yield session
        session.commit()
    finally:
        session.close()
