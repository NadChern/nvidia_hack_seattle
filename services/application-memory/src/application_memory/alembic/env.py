"""Alembic environment.

The database URL comes from the service settings rather than alembic.ini, so a
migration always runs against the same database the service uses. Two sources
of truth for a connection string is how a migration silently upgrades the wrong
file.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from application_memory.config import Settings
from application_memory.store.models import Base

config = context.config
target_metadata = Base.metadata

settings = Settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

# Alembic builds its own engine, so it bypasses the directory creation in
# store.engine. Without this, `alembic upgrade head` on a fresh checkout fails
# with an opaque "unable to open database file" that names neither the path nor
# the reason.
if (sqlite_path := settings.sqlite_path) is not None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot ALTER most things in place; batch mode rewrites the
            # table instead, so a column change is possible at all.
            render_as_batch=settings.is_sqlite,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
