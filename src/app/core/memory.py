"""Give unused RAM back to Windows (DECISION-41).

After a model is closed, Python/torch keep its freed pages in the process working set, so
the worker held ~2 GB after each model and the next stage stacked on top (measured
2026-11-25). `SetProcessWorkingSetSize(h, -1, -1)` empties the working set: pages still in
use fault straight back in, the rest become available to other programs.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any


def _kernel32() -> Any:
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    return k


def trim_working_set() -> bool:
    """True when Windows was asked to trim; no-op elsewhere."""
    if sys.platform != "win32":
        return False
    k = _kernel32()
    minus_one = ctypes.c_size_t(-1)
    return bool(
        k.SetProcessWorkingSetSize(ctypes.c_void_p(k.GetCurrentProcess()), minus_one, minus_one)
    )
