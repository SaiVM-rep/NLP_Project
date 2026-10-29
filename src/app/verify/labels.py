"""FR-5.4 label rule. PRD §11.8 step 6. Thresholds come from config, tuned on AggreFact dev."""

from __future__ import annotations

from app.core.schemas import ClaimLabel


def label(
    max_entail: float, max_contra: float | None, *, theta_e: float, theta_c: float
) -> ClaimLabel:
    """`max_contra=None` for single-score verifiers (MiniCheck): they can only support or not."""
    if max_entail >= theta_e:
        return ClaimLabel.SUPPORTED
    if max_contra is not None and max_contra >= theta_c:
        return ClaimLabel.CONTRADICTED
    return ClaimLabel.UNSUPPORTED
