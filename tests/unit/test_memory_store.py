"""The in-memory Store (API/worker unit tests, demo mode) against the shared contract."""

from __future__ import annotations

import pytest

from app.db.memory import MemoryStore
from app.db.store import Store
from tests.store_contract import StoreContract


class TestMemoryStore(StoreContract):
    @pytest.fixture
    def store(self) -> Store:
        return MemoryStore()
