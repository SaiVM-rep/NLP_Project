"""Migration runner ordering. Applying them is covered live in test_pg_store.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.db.migrate import MIGRATIONS, pending


def test_pending_is_unapplied_files_in_version_order(tmp_path: Path) -> None:
    for name in ("0002_b.sql", "0001_a.sql", "0010_c.sql", "notes.txt"):
        (tmp_path / name).write_text("SELECT 1;")
    got = pending(tmp_path, applied={"0001_a"})
    assert [p.name for p in got] == ["0002_b.sql", "0010_c.sql"]


def test_versions_must_be_unique(tmp_path: Path) -> None:
    (tmp_path / "0001_a.sql").write_text("")
    (tmp_path / "0001_b.sql").write_text("")
    with pytest.raises(ValueError, match="0001"):
        pending(tmp_path, applied=set())


def test_shipped_migrations_start_with_the_prd_schema() -> None:
    first, second = pending(MIGRATIONS, applied=set())[:2]
    assert first.name == "0001_init.sql" and "CREATE TABLE llm_calls" in first.read_text("utf-8")
    assert second.name == "0002_service.sql"
