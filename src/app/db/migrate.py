"""`python -m app.db.migrate` (`make migrate`): apply `migrations/NNNN_name.sql` in order.

Plain SQL files, no Alembic (DECISION-26). Each file runs in its own transaction together
with its row in `schema_migrations`, so a failed file leaves nothing half-applied.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MIGRATIONS = Path(__file__).with_name("migrations")


def pending(folder: Path, applied: set[str]) -> list[Path]:
    files = sorted(folder.glob("[0-9][0-9][0-9][0-9]_*.sql"))
    seen: dict[str, str] = {}
    for f in files:
        v = f.name[:4]
        if v in seen:
            raise ValueError(f"migration version {v} used twice: {seen[v]}, {f.name}")
        seen[v] = f.name
    return [f for f in files if f.stem not in applied]


def migrate(dsn: str, folder: Path = MIGRATIONS) -> list[str]:
    """Apply what is missing; returns the versions applied now."""
    import psycopg

    done: list[str] = []
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        conn.commit()
        applied = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
        for f in pending(folder, applied):
            with conn.transaction():
                conn.execute(f.read_text("utf-8").encode())  # a file, not a literal query
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (f.stem,))
            done.append(f.stem)
    return done


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()
    for v in migrate(os.environ["DATABASE_URL"]) or ["(nothing to apply)"]:
        print(f"applied {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
