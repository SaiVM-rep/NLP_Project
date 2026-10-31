"""FR-2.1 / FR-2.2 / FR-2.3: token estimate and the Route A/B boundary."""

from __future__ import annotations

from app.core.config import Limits, ModelProfile
from app.core.schemas import Route
from app.route.router import route, tau_for
from app.route.tokens import estimate_tokens

PROFILE = ModelProfile(
    provider="gemini", model="m", effective_context=10_000, limits=Limits(rpm=1, tpm=1, rpd=1)
)


def test_estimate_is_chars_over_four_plus_margin() -> None:
    assert estimate_tokens("abcd" * 100, margin=0.0) == 100
    assert estimate_tokens("abcd" * 100, margin=0.1) == 110


def test_estimate_rounds_up_so_it_never_undercounts() -> None:
    assert estimate_tokens("abcde", margin=0.0) == 2
    assert estimate_tokens("", margin=0.1) == 0


def test_tau_is_fraction_of_effective_context() -> None:
    assert tau_for(PROFILE, 0.8) == 8000


def test_override_beats_fraction() -> None:
    assert tau_for(PROFILE, 0.8, tau_override=123) == 123


def test_boundary_tokens_equal_tau_is_route_a() -> None:
    assert route(8000, tau=8000) is Route.A


def test_one_over_tau_is_route_b() -> None:
    assert route(8001, tau=8000) is Route.B


def test_small_context_picks_the_smallest_window_that_fits() -> None:
    from app.route.tokens import small_context

    short, mid = "x" * 4 * 1000, "x" * 4 * 5000  # ~1.1k and ~5.5k tokens with margin
    assert small_context([short], [8192, 4096], ceiling=16384) == 4096
    assert small_context([short, mid], [4096, 8192], ceiling=16384) == 8192  # the largest decides


def test_small_context_falls_back_to_the_profile_window() -> None:
    from app.route.tokens import small_context

    big = "x" * 4 * 9000
    assert small_context([big], [4096, 8192], ceiling=16384) is None  # too big for any choice
    assert small_context(["hi"], [], ceiling=16384) is None  # feature off for this profile
    assert small_context(["hi"], [4096], ceiling=4096) is None  # not smaller than the window
    assert small_context([], [4096], ceiling=16384) is None
