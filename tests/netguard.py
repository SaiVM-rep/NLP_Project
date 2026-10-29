"""Socket-level network guard for FR-5.6 ("verification runs locally; no API calls").

Blocks every non-loopback connection and DNS lookup, records the attempt, and fails the
way an offline machine does (OSError / gaierror), so callers' own fallbacks are exercised.
Loopback stays open: Ollama is localhost, and asyncio on Windows uses loopback socket pairs.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

LOCAL_NAMES = {"localhost", "::1", "0:0:0:0:0:0:0:1"}


def _is_local(host: Any) -> bool:
    if host is None:  # getaddrinfo(None, port): bind-style lookup, no network
        return True
    if isinstance(host, bytes):
        host = host.decode()
    host = str(host).lower()
    return host in LOCAL_NAMES or host.startswith("127.")


@contextmanager
def no_network() -> Iterator[list[str]]:
    """Yields the list of blocked attempts (`host:port`, or `dns:host`)."""
    attempts: list[str] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def _blocked(address: Any) -> str | None:
        if not isinstance(address, tuple):  # AF_UNIX path
            return None
        host, port = address[0], address[1]
        return None if _is_local(host) else f"{host}:{port}"

    def connect(self: socket.socket, address: Any) -> None:
        if (where := _blocked(address)) is not None:
            attempts.append(where)
            raise ConnectionRefusedError(f"network disabled (FR-5.6 test): {where}")
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        if (where := _blocked(address)) is not None:
            attempts.append(where)
            return 111  # ECONNREFUSED
        return real_connect_ex(self, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_local(host):
            attempts.append(f"dns:{host}")
            raise socket.gaierror(socket.EAI_NONAME, f"network disabled (FR-5.6 test): {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = getaddrinfo
    try:
        yield attempts
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex  # type: ignore[method-assign]
        socket.getaddrinfo = real_getaddrinfo
