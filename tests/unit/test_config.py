"""Config loading. PRD §13.

Limits and context budgets must come from configs/models.yaml and never be
hard-coded (PRD §11.3). These tests pin that contract.
"""

import pytest
from pydantic import ValidationError

from app.core.config import AppConfig, ModelProfile, load_config, load_profiles

MODELS_YAML = """
profiles:
  gemini_full:
    provider: gemini
    model: gemini-2.5-flash
    effective_context: 1000000
    limits: {rpm: 10, tpm: 250000, rpd: 250}
  sim_8k:
    provider: gemini
    model: gemini-2.5-flash
    effective_context: 8000
    limits: {rpm: 10, tpm: 250000, rpd: 250}
"""

DEFAULT_YAML = """
profile: gemini_full
route: {tau_fraction: 0.8, token_margin: 0.10}
chunk: {chunk_tokens: 3000, overlap_tokens: 150, fan_in_k: 4}
decoding: {temperature: 0.2, top_p: 0.9, seed: 1234}
summary: {compression_pct: 10}
keypoints:
  {top_k: 8, candidates_per_section: 5, theta_attr: 0.55, dup_distance: 0.15, lambda_graph: 0.5}
verify:
  top_n_evidence: 5
  decomposer: ollama
  embedder: {checkpoint: org/emb, revision: "1111111111111111111111111111111111111111"}
  nli:
    checkpoint: org/nli
    revision: "2222222222222222222222222222222222222222"
    batch_size: 32
    theta_e: 0.5
    theta_c: 0.6
  minicheck:
    theta_e: 0.7
    checkpoint: org/mc
    revision: "3333333333333333333333333333333333333333"
    batch_size: 16
    chunk_words: 500
metrics:
  bertscore:
    checkpoint: org/rob
    revision: "4444444444444444444444444444444444444444"
    num_layers: 17
    baseline: roberta-large
    batch_size: 32
fallback: fail
cache: {enabled: true, dir: .cache/llm}
"""


@pytest.fixture
def cfg_dir(tmp_path):
    (tmp_path / "models.yaml").write_text(MODELS_YAML, encoding="utf-8")
    (tmp_path / "default.yaml").write_text(DEFAULT_YAML, encoding="utf-8")
    return tmp_path


# ------------------------------------------------------------ profiles ---


def test_loads_every_profile(cfg_dir):
    profiles = load_profiles(cfg_dir / "models.yaml")
    assert set(profiles) == {"gemini_full", "sim_8k"}


def test_profile_carries_context_and_limits(cfg_dir):
    p = load_profiles(cfg_dir / "models.yaml")["sim_8k"]
    assert p.effective_context == 8000
    assert p.limits.rpm == 10
    assert p.limits.rpd == 250


def test_simulated_profile_shrinks_context_not_the_model(cfg_dir):
    """PRD §8.1: sim_* is a controlled variable, same model, smaller budget."""
    profiles = load_profiles(cfg_dir / "models.yaml")
    assert profiles["sim_8k"].model == profiles["gemini_full"].model
    assert profiles["sim_8k"].effective_context < profiles["gemini_full"].effective_context


def test_profile_rejects_nonpositive_context():
    with pytest.raises(ValidationError):
        ModelProfile(
            provider="gemini", model="m", effective_context=0, limits={"rpm": 1, "tpm": 1, "rpd": 1}
        )


# -------------------------------------------------------------- config ---


def test_loads_default_config(cfg_dir):
    cfg = load_config(cfg_dir)
    assert cfg.profile == "gemini_full"
    assert cfg.chunk.chunk_tokens == 3000
    assert cfg.chunk.fan_in_k == 4
    assert (cfg.verify.nli.theta_e, cfg.verify.nli.theta_c) == (0.5, 0.6)
    assert cfg.verify.minicheck.theta_e == 0.7
    assert cfg.cache.enabled is True


def test_active_profile_resolves_from_the_profile_name(cfg_dir):
    cfg = load_config(cfg_dir)
    assert cfg.active_profile.effective_context == 1000000


def test_unknown_profile_name_fails_loudly(cfg_dir):
    with pytest.raises(ValueError, match="unknown profile"):
        load_config(cfg_dir, overrides={"profile": "sim_999k"})


def test_overrides_win_over_the_yaml(cfg_dir):
    cfg = load_config(cfg_dir, overrides={"profile": "sim_8k"})
    assert cfg.active_profile.effective_context == 8000


def test_nested_override_does_not_wipe_its_siblings(cfg_dir):
    cfg = load_config(cfg_dir, overrides={"chunk": {"chunk_tokens": 500}})
    assert cfg.chunk.chunk_tokens == 500
    assert cfg.chunk.overlap_tokens == 150  # untouched
    assert cfg.chunk.fan_in_k == 4


def test_fallback_policy_defaults_to_fail(cfg_dir):
    """PRD §11.3: a run must never silently mix models."""
    assert load_config(cfg_dir).fallback == "fail"


def test_fallback_policy_rejects_unknown_values(cfg_dir):
    with pytest.raises(ValidationError):
        load_config(cfg_dir, overrides={"fallback": "pray"})


def test_overlap_must_be_smaller_than_chunk_size(cfg_dir):
    with pytest.raises(ValidationError, match="overlap"):
        load_config(cfg_dir, overrides={"chunk": {"chunk_tokens": 100, "overlap_tokens": 100}})


def test_tau_fraction_must_be_between_zero_and_one(cfg_dir):
    with pytest.raises(ValidationError):
        load_config(cfg_dir, overrides={"route": {"tau_fraction": 1.5}})


def test_config_is_hashable_for_the_run_manifest(cfg_dir):
    """FR-7.3: the manifest records the config, so it must serialise stably."""
    a = load_config(cfg_dir).fingerprint()
    b = load_config(cfg_dir).fingerprint()
    assert a == b and len(a) == 64


def test_fingerprint_changes_when_config_changes(cfg_dir):
    a = load_config(cfg_dir).fingerprint()
    b = load_config(cfg_dir, overrides={"chunk": {"chunk_tokens": 500}}).fingerprint()
    assert a != b


# --------------------------------------------------------------- types ---


def test_appconfig_rejects_unknown_keys(cfg_dir):
    """A typo in a yaml key must fail, not be silently ignored."""
    with pytest.raises(ValidationError):
        load_config(cfg_dir, overrides={"chnk": {"chunk_tokens": 1}})


def test_appconfig_is_importable_as_a_type():
    assert AppConfig.__name__ == "AppConfig"


# -------------------------------------------------------------- verifiers ---


def test_verifier_checkpoints_come_from_config(cfg_dir):
    v = load_config(cfg_dir).verify
    assert (v.nli.checkpoint, v.nli.batch_size) == ("org/nli", 32)
    assert v.minicheck.chunk_words == 500
    assert v.embedder.revision == "1" * 40


@pytest.mark.parametrize("rev", ["main", "v1.0", "b3546ea", "G" * 40])
def test_checkpoint_revision_must_be_a_full_commit_sha(cfg_dir, rev):
    """A branch name drifts under us; only a commit sha reproduces a run (FR-7.3)."""
    with pytest.raises(ValidationError, match="revision"):
        load_config(cfg_dir, overrides={"verify": {"nli": {"revision": rev}}})


def test_real_configs_pin_the_verified_checkpoints():
    """CLAUDE.md §8: checkpoints confirmed live on 2026-10-29."""
    v = load_config("configs").verify
    assert v.nli.checkpoint == "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    assert v.nli.revision == "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
    assert v.minicheck.checkpoint == "lytang/MiniCheck-Flan-T5-Large"
    assert v.minicheck.revision == "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"
    assert v.embedder.checkpoint == "sentence-transformers/all-MiniLM-L6-v2"
    assert v.embedder.revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def test_real_configs_pin_bertscore():
    """bert-score's own default (roberta-large, layer 17), pinned to a commit."""
    b = load_config("configs").metrics.bertscore
    assert (b.checkpoint, b.revision) == (
        "FacebookAI/roberta-large",
        "722cf37b1afa9454edce342e7895e588b6ff1d59",
    )
    assert (b.num_layers, b.baseline) == (17, "roberta-large")


def test_real_default_profile_is_local_qwen():
    """DECISION-27: no API key or internet needed for a default run."""
    cfg = load_config("configs")
    assert cfg.profile == "qwen3_8b_local"
    assert cfg.active_profile.provider == "ollama"


def test_real_qwen_sim_profiles_fit_inside_the_real_window():
    """E2 needs budgets that make Route B trigger, below Qwen's real 16k window."""
    profiles = load_config("configs").profiles
    real = profiles["qwen3_8b_local"]
    for name in ("qwen_sim_4k", "qwen_sim_8k", "qwen_sim_12k"):
        p = profiles[name]
        assert (p.provider, p.model) == (real.provider, real.model)
        assert p.effective_context < real.effective_context
        assert p.limits == real.limits


def test_real_thresholds_are_the_frozen_aggrefact_dev_values():
    """DECISION-28: tuned on AggreFact dev only (results/tuning/aggrefact_dev.summary.json).

    MiniCheck stays at its upstream 0.5: the dev curve is flat from 0.25 to 0.6 and the
    sweep's best (0.22) beats 0.5 by 0.0005 balanced accuracy, which is noise.
    theta_c is not tunable from binary labels and stays 0.5.
    """
    v = load_config("configs").verify
    assert (v.nli.theta_e, v.nli.theta_c) == (0.10, 0.5)
    assert v.minicheck.theta_e == 0.5


def test_real_local_profiles_send_one_request_at_a_time():
    """E1 run 1 died: 4 queued Route B calls on one GPU waited past Ollama's 600 s timeout."""
    profiles = load_config("configs").profiles
    for name, p in profiles.items():
        assert p.max_concurrency == (1 if p.provider == "ollama" else 4), name


def test_local_qwen_profile_has_small_windows_below_its_context() -> None:
    cfg = load_config("configs")
    p = cfg.profiles["qwen3_8b_local"]
    assert p.small_ctx == [4096, 8192]
    assert all(c < p.effective_context for c in p.small_ctx)
    assert cfg.profiles["gemini_full"].small_ctx == []


def test_summaries_run_without_thinking_and_uncapped_by_default() -> None:
    # DECISION-37: Qwen3 thinking cost 13.7 tokens per summary word and one runaway
    # 15-min call; key points and decomposition were already think-off.
    s = load_config("configs").summary
    assert s.think is False
    assert s.max_words is None  # experiments keep the PRD 10% rule; uploads cap per job


def test_summary_word_cap_must_be_positive(cfg_dir):
    with pytest.raises(ValidationError):
        load_config(cfg_dir, overrides={"summary": {"max_words": 0}})
