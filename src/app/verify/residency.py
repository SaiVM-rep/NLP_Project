"""Load, score, close: one verifier on the GPU at a time (DECISION-3, 8 GB card).

torch is imported lazily so torch-free installs can still import the verify package.
"""

from __future__ import annotations

import gc
from types import TracebackType
from typing import Any, Self

from app.core.memory import trim_working_set


class GpuResident:
    def __init__(self, model: Any, *, device: str) -> None:
        self._model: Any = model.to(device).eval()
        self._device = device

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError(f"{type(self).__name__} is closed")
        return self._model

    def trim(self) -> None:
        """Hand cached-but-free GPU blocks back to the driver."""
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    def close(self) -> None:
        if self._model is None:
            return
        self._model = None
        gc.collect()
        self.trim()
        trim_working_set()  # the freed weights would otherwise stay counted against RAM

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
