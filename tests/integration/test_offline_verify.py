"""FR-5.6: verification runs locally, no API calls. PRD §6 acceptance: network-disabled test.

The whole verify stage — Qwen3 decomposition via local Ollama, embedder, DeBERTa,
MiniCheck — runs with every non-loopback connection refused, from a cold LLM cache.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

torch = pytest.importorskip("torch")

from app.cli import make_provider  # noqa: E402
from app.core.config import load_config  # noqa: E402
from app.core.schemas import ClaimLabel, SummaryNode  # noqa: E402
from app.ingest.spans import build_document  # noqa: E402
from app.ingest.tei import parse_tei  # noqa: E402
from app.llm.cache import DiskCache  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.verify.pipeline import openers_from_config, verify_nodes  # noqa: E402
from tests.netguard import no_network  # noqa: E402

load_dotenv()  # HF_HOME, OLLAMA_URL
ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs")


def _ollama_up() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not _ollama_up(), reason="needs local ollama"),
]

SUMMARY = (
    "Sentence-BERT uses siamese and triplet networks to derive sentence embeddings. "
    "SBERT was pre-trained on ImageNet."
)


def test_default_decomposer_is_local() -> None:
    """A Gemini decomposer would be an API call inside verification."""
    assert CFG.verify.decomposer == "ollama"
    assert CFG.profiles["qwen3_8b_local"].provider == "ollama"


async def test_verify_stage_completes_with_network_disabled(tmp_path: Path) -> None:
    tei = parse_tei((ROOT / "tests/fixtures/tei/sbert.tei.xml").read_bytes())
    doc = build_document("sbert", "pdf", tei.title, tei.sections, tei.references_text)
    profile = CFG.profiles["qwen3_8b_local"]
    provider = make_provider(profile)
    client = LLMClient(provider, profile, DiskCache(tmp_path))  # cold: real local calls

    async def release() -> None:
        await provider.unload(profile.model)  # type: ignore[attr-defined]

    try:
        with no_network() as attempts:
            out = await verify_nodes(
                doc,
                [SummaryNode(id="root", level=0, text=SUMMARY)],
                client,
                openers_from_config(CFG.verify),
                CFG.verify,
                release_llm=release,
                prompt_dir=ROOT / "prompts",
            )
    finally:
        await provider.aclose()

    assert attempts == [], f"verification tried the network: {attempts}"
    assert out, "no claims verified"
    planted = [r for r in out if "ImageNet" in r.claim]
    assert planted and planted[0].label_deberta is ClaimLabel.CONTRADICTED
