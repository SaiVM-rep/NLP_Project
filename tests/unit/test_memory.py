"""Working-set trimming (DECISION-41): hand unused pages back so free RAM recovers between
models instead of the worker keeping ~2 GB after each one (measured 2026-11-25)."""

from __future__ import annotations

from typing import Any

import pytest

from app.core import memory


class FakeKernel:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, int, int]] = []

    def GetCurrentProcess(self) -> int:
        return -1

    def SetProcessWorkingSetSize(self, h: Any, lo: Any, hi: Any) -> int:
        self.calls.append((h, lo.value, hi.value))
        return 1


def test_trim_asks_windows_to_empty_the_working_set(monkeypatch: pytest.MonkeyPatch) -> None:
    k = FakeKernel()
    monkeypatch.setattr(memory.sys, "platform", "win32")
    monkeypatch.setattr(memory, "_kernel32", lambda: k)
    assert memory.trim_working_set() is True
    ((_, lo, hi),) = k.calls
    assert lo == hi == memory.ctypes.c_size_t(-1).value  # (SIZE_T)-1, -1 = empty it


def test_trim_is_a_no_op_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory.sys, "platform", "linux")
    monkeypatch.setattr(memory, "_kernel32", lambda: pytest.fail("no Win32 call off Windows"))
    assert memory.trim_working_set() is False
