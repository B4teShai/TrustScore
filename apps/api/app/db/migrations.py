"""Best-effort startup migrations for the optional Postgres persistence layer.

Applies db/migrations/*.sql in filename order and records each applied file in
a schema_migrations table, so the API creates or upgrades its own schema on
startup — no manual psql step and no reliance on the postgres image's
first-init hook. Every migration file is written with IF NOT EXISTS guards, so
running them against a database that was initialized before migration tracking
existed is safe.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import settings
from app.db.session import get_engine, report_engine_connection_failure


logger = logging.getLogger(__name__)

_MIGRATIONS_TABLE_DDL = """
create table if not exists schema_migrations (
    filename text primary key,
    applied_at timestamptz not null default now()
)
"""


def run_startup_migrations() -> list[str]:
    """Apply pending migration files and return their names.

    Best-effort like the rest of persistence: no database configured, an
    unreachable server, or a failing migration must never prevent the API
    from starting — the failure is logged and scoring continues without
    durable persistence.
    """
    engine = get_engine()
    if engine is None:
        logger.info("startup_migrations_skipped_no_database")
        return []

    directory = Path(settings.migrations_dir)
    if not directory.is_dir():
        logger.warning(
            "startup_migrations_directory_missing",
            extra={"migrations_dir": str(directory)},
        )
        return []
    files = sorted(path for path in directory.glob("*.sql") if path.is_file())
    if not files:
        logger.info(
            "startup_migrations_no_files",
            extra={"migrations_dir": str(directory)},
        )
        return []

    applied: list[str] = []
    try:
        from sqlalchemy import text

        with engine.begin() as connection:
            connection.exec_driver_sql(_MIGRATIONS_TABLE_DDL)
            already_applied = {
                row[0]
                for row in connection.execute(text("select filename from schema_migrations"))
            }

        for path in files:
            if path.name in already_applied:
                continue
            sql = path.read_text(encoding="utf-8").strip()
            if not sql:
                continue
            # One transaction per file: the migration and its tracking row
            # commit together, so a failure leaves earlier files applied and
            # this file cleanly unapplied.
            with engine.begin() as connection:
                connection.exec_driver_sql(sql)
                connection.execute(
                    text(
                        "insert into schema_migrations (filename) values (:filename) "
                        "on conflict (filename) do nothing"
                    ),
                    {"filename": path.name},
                )
            applied.append(path.name)
            logger.info("startup_migration_applied", extra={"filename": path.name})
    except Exception as exc:
        if "OperationalError" in type(exc).__name__:
            report_engine_connection_failure()
        logger.warning(
            "startup_migrations_failed",
            extra={"error_type": type(exc).__name__, "error": str(exc), "applied": applied},
            exc_info=True,
        )
        return applied

    logger.info(
        "startup_migrations_complete",
        extra={"applied": applied, "total_files": len(files)},
    )
    return applied
