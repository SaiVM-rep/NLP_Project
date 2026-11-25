"""GpuResident: load/close and cache trimming (DECISION-3)."""

from __future__ import annotations

from typing import Any

import pytest

from app.verify.residency import GpuResident


class FakeModel:
    def to(self, device: str) -> FakeModel:
        return self

    def eval(self) -> FakeModel:
        return self


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    torch: Any = pytest.importorskip("torch")
    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    return calls


def test_trim_returns_cached_gpu_memory_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy(monkeypatch)
    GpuResident(FakeModel(), device="cuda").trim()
    assert calls == [1]


def test_trim_is_a_no_op_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy(monkeypatch)
    GpuResident(FakeModel(), device="cpu").trim()
    assert calls == []
