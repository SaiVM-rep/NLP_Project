"""Route A vs B. PRD FR-2.2, FR-2.3, §11.5."""

from __future__ import annotations

from app.core.config import ModelProfile
from app.core.schemas import Route


def tau_for(profile: ModelProfile, tau_fraction: float, tau_override: int | None = None) -> int:
    return tau_override or int(tau_fraction * profile.effective_context)


def route(tokens: int, tau: int) -> Route:
    return Route.A if tokens <= tau else Route.B
