"""Shared fixtures for the Postgres-backed tests (`make up` first)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.environ.get("DATABASE_URL", "postgresql://app:app@localhost:5432/app")


@pytest.fixture
def dsn() -> Iterator[str]:
    """A fresh schema with nothing in it, dropped afterwards."""
    psycopg = pytest.importorskip("psycopg")
    try:
        admin = psycopg.connect(DB_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"no postgres at {DB_URL}: {e}")
    schema = f"t_{uuid4().hex[:12]}"
    with admin:
        admin.execute(f"CREATE SCHEMA {schema}")
        try:
            yield f"{DB_URL}?options=-csearch_path%3D{schema}"
        finally:
            admin.execute(f"DROP SCHEMA {schema} CASCADE")
