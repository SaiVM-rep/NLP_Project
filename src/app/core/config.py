"""Typed config loading. PRD §13.

Rate limits and context budgets live in configs/models.yaml and are never
hard-coded (PRD §11.3) — they change at the provider's discretion.

`extra="forbid"` throughout: a mistyped yaml key must fail the run, not be
silently dropped into a default.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Limits(_Strict):
    rpm: int = Field(gt=0)
    tpm: int = Field(gt=0)
    rpd: int = Field(gt=0)


class ModelProfile(_Strict):
    """A model plus an *experimental* context budget. PRD §8.1."""

    provider: Literal["gemini", "ollama"]
    model: str
    effective_context: int = Field(gt=0)
    limits: Limits
    # Route B map calls in flight. A local model on one GPU serves one request at a time,
    # so extra in-flight calls only queue (and can outlive the client timeout).
    max_concurrency: int = Field(default=4, ge=1)
    # Smaller windows for stages with short prompts (key points, claim decomposition).
    # Ollama only: at 16384 qwen3:8b keeps 6 of 37 layers on the CPU (25 tok/s, ~1 GB host
    # RAM); at 4096/8192 it fits the 8 GB GPU (43 tok/s). DECISION-35.
    small_ctx: list[int] = Field(default_factory=list)


class RouteCfg(_Strict):
    tau_fraction: float = Field(gt=0.0, le=1.0)
    token_margin: float = Field(ge=0.0, lt=1.0)


class ChunkCfg(_Strict):
    chunk_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)
    fan_in_k: int = Field(gt=1)

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> ChunkCfg:
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be smaller than "
                f"chunk_tokens ({self.chunk_tokens}); otherwise chunking never advances"
            )
        return self


class DecodingCfg(_Strict):
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    seed: int


class SummaryCfg(_Strict):
    compression_pct: float = Field(gt=0.0, le=100.0)
    # Qwen3 hidden reasoning for summary calls (Ollama `think`); None = model default.
    # Off since DECISION-37: thinking cost ~13.7 tokens per summary word on this GPU.
    think: bool | None = None
    # Upper bound on any summary target (single pass, chunk, merge); None = PRD 10% rule.
    max_words: int | None = Field(default=None, gt=0)


class KeyPointsCfg(_Strict):
    top_k: int = Field(gt=0)
    candidates_per_section: int = Field(gt=0)  # `{n}` in keypoints.v1
    theta_attr: float = Field(ge=0.0, le=1.0)
    dup_distance: float = Field(ge=0.0, le=1.0)
    lambda_graph: float = Field(ge=0.0, le=1.0)


class ModelRef(_Strict):
    """A Hugging Face checkpoint pinned to a commit. A branch name drifts; a sha reproduces."""

    checkpoint: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


# Thresholds are per verifier: DeBERTa and MiniCheck scores are not on one scale, so
# AggreFact-dev tuning yields one pair each (PRD §11.8 step 6).
class NliCfg(ModelRef):
    batch_size: int = Field(gt=0)
    theta_e: float = Field(ge=0.0, le=1.0)
    theta_c: float = Field(ge=0.0, le=1.0)


class MiniCheckCfg(ModelRef):
    batch_size: int = Field(gt=0)
    chunk_words: int = Field(gt=0)
    theta_e: float = Field(ge=0.0, le=1.0)  # single support score: no contradiction threshold
    # Score each claim against its k most similar chunks only; None = every chunk (upstream).
    top_k_chunks: int | None = Field(default=None, gt=0)


class VerifyCfg(_Strict):
    top_n_evidence: int = Field(gt=0)
    decomposer: Literal["ollama", "gemini"]
    embedder: ModelRef
    nli: NliCfg
    minicheck: MiniCheckCfg


class BertScoreCfg(ModelRef):
    num_layers: int = Field(gt=0)
    baseline: str  # rescale_baseline/en/<baseline>.tsv shipped with bert-score
    batch_size: int = Field(gt=0)


class MetricsCfg(_Strict):
    bertscore: BertScoreCfg


class CacheCfg(_Strict):
    enabled: bool
    dir: str


class AppConfig(_Strict):
    profile: str
    route: RouteCfg
    chunk: ChunkCfg
    decoding: DecodingCfg
    summary: SummaryCfg
    keypoints: KeyPointsCfg
    verify: VerifyCfg
    metrics: MetricsCfg
    fallback: Literal["fail", "ollama"]
    cache: CacheCfg

    # Populated by load_config; not part of the yaml surface.
    profiles: dict[str, ModelProfile] = Field(default_factory=dict, exclude=True)

    @property
    def active_profile(self) -> ModelProfile:
        return self.profiles[self.profile]

    def fingerprint(self) -> str:
        """Stable sha256 of the config, for the run manifest (FR-7.3)."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_profiles(path: str | Path) -> dict[str, ModelProfile]:
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return {name: ModelProfile(**body) for name, body in raw["profiles"].items()}


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Override leaves without wiping sibling keys."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    config_dir: str | Path = "configs",
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    config_dir = Path(config_dir)
    with (config_dir / "default.yaml").open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if overrides:
        raw = _deep_merge(raw, overrides)

    profiles = load_profiles(config_dir / "models.yaml")
    if raw.get("profile") not in profiles:
        raise ValueError(f"unknown profile {raw.get('profile')!r}; available: {sorted(profiles)}")

    cfg = AppConfig(**raw)
    cfg.profiles = profiles
    return cfg
