"""The network guard behind the FR-5.6 test must itself be trustworthy."""

from __future__ import annotations

import socket

import httpx
import pytest

from tests.netguard import no_network


def test_external_connect_is_refused_and_recorded() -> None:
    with no_network() as attempts:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError):
                s.connect(("93.184.216.34", 443))
            assert s.connect_ex(("8.8.8.8", 53)) != 0
        finally:
            s.close()
    assert attempts == ["93.184.216.34:443", "8.8.8.8:53"]


def test_dns_lookup_of_external_host_is_refused() -> None:
    with no_network() as attempts, pytest.raises(socket.gaierror):
        socket.getaddrinfo("huggingface.co", 443)
    assert attempts == ["dns:huggingface.co"]


def test_real_http_client_fails_like_an_offline_machine() -> None:
    with no_network() as attempts, pytest.raises(httpx.ConnectError):
        httpx.get("https://huggingface.co/api/models", timeout=5)
    assert attempts  # recorded, whichever layer it hit first


def test_loopback_still_works() -> None:
    """Ollama is localhost; asyncio on Windows uses loopback socket pairs."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        with no_network() as attempts:
            socket.getaddrinfo("localhost", port)
            c = socket.create_connection(("127.0.0.1", port), timeout=5)
            c.close()
        assert attempts == []
    finally:
        server.close()


def test_guard_is_removed_on_exit() -> None:
    orig = (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo)
    with no_network():
        pass
    assert (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo) == orig
