"""Decision logic of the one-command showcase launcher (scripts/showcase.py)."""

from scripts.showcase import CLAIMS_MIN_GB, MemoryGuard, pdf_verdict, ram_verdict


def test_ram_verdict_allows_claim_checking_only_with_enough_headroom() -> None:
    ok, msg = ram_verdict(CLAIMS_MIN_GB + 0.1)
    assert ok and "Check claims" in msg
    ok, msg = ram_verdict(CLAIMS_MIN_GB - 0.1)
    assert not ok and "untick" in msg


def test_guard_ignores_one_dip_and_trips_on_a_sustained_low() -> None:
    g = MemoryGuard(floor_mb=400, samples=3)
    assert not g.update(300)
    assert not g.update(900)  # recovered: the count starts again
    assert not g.update(300)
    assert not g.update(350)
    assert g.update(390)


def test_guard_resets_after_tripping() -> None:
    g = MemoryGuard(floor_mb=400, samples=1)
    assert g.update(100)
    g.reset()
    assert not g.update(500)


def test_pdf_verdict_short_paper_is_route_a() -> None:
    v = pdf_verdict(tokens=7000, tau=13107)
    assert v.route == "A" and v.ok
    assert "Route A" in v.message


def test_pdf_verdict_long_paper_warns_about_route_b_time() -> None:
    v = pdf_verdict(tokens=20000, tau=13107)
    assert v.route == "B" and not v.ok
    assert "prewarm" in v.message.lower()


def test_pdf_verdict_boundary_is_route_a() -> None:
    assert pdf_verdict(tokens=13107, tau=13107).route == "A"  # FR-2.2: tokens <= tau


def test_pdf_verdict_empty_text_is_not_ok() -> None:
    v = pdf_verdict(tokens=0, tau=13107)
    assert not v.ok and "no text" in v.message.lower()


def test_display_leak_is_flagged_with_what_to_do() -> None:
    # 2026-11-25: dwm held 4.4 GB of RAM as shared GPU memory after 1 day of uptime and a
    # long-paper upload stopped at 0.32 GB free; normal is a few hundred MB.
    from scripts.showcase import display_leak_verdict

    assert display_leak_verdict(0.4) is None
    msg = display_leak_verdict(4.4)
    assert msg is not None and "4.4 GB" in msg and "restart" in msg.lower()
