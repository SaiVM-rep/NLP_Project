"""PgStore against real Postgres (`make up`), same contract as MemoryStore.

Each test gets a fresh schema (`dsn` in conftest) with every migration applied.
"""

from __future__ import annotations

import pytest

pytest.importorskip("psycopg")

from app.db.migrate import migrate
from app.db.pg import PgStore
from app.db.store import Store
from tests.store_contract import StoreContract

pytestmark = pytest.mark.integration


class TestPgStore(StoreContract):
    @pytest.fixture
    def store(self, dsn: str) -> Store:
        migrate(dsn)
        return PgStore(dsn)


def test_migrate_is_idempotent(dsn: str) -> None:
    assert migrate(dsn) == ["0001_init", "0002_service"]
    assert migrate(dsn) == []
