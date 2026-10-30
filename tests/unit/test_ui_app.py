"""P7 page wiring: the pick -> highlight path. (Building a gr.Blocks leaks asyncio sockets,
which filterwarnings=error rejects, so the page itself is checked live: tests/integration.)"""

from app.ui.app import _highlight
from app.ui.view import Bundle, make_view
from tests.unit.test_ui_view import DOC, RESULT


def _view():  # type: ignore[no-untyped-def]
    return make_view(Bundle(title="SBERT", document=DOC, result=RESULT))


def test_picking_a_claim_highlights_its_evidence() -> None:
    v = _view()
    out = _highlight(v, "claim", v.claims[1].label)
    assert "<mark" in out and "Training took 20 minutes." in out.split("<mark")[1]


def test_picking_an_unattributed_key_point_highlights_nothing() -> None:
    v = _view()
    assert "<mark" not in _highlight(v, "kp", v.key_points[1].label)


def test_nothing_picked_or_no_view_is_safe() -> None:
    assert _highlight(None, "kp", "x") == ""
    assert "<mark" not in _highlight(_view(), "claim", None)
