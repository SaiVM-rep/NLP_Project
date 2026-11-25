"""One job end to end: ingest -> summarize -> key points -> verify (PRD §8, FR-6.3).

`make_pipeline` owns order, progress and metrics; the model-touching work sits behind
`Stages` so it is unit-testable with fakes. `LiveStages` is the real thing, built per job
from the job's options, and closed whatever happens.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import AppConfig
from app.core.logging import doc_context
from app.core.memory import trim_working_set
from app.core.schemas import ClaimLabel, Document, KeyPoint
from app.db.store import Job, JobOutput
from app.ingest.objects import PaperObject
from app.route.tokens import estimate_tokens
from app.summarize.common import SummaryResult
from app.verify.pipeline import VerifiedSummary, summary_nodes
from app.worker.runner import JobContext, Pipeline

ROOT = Path(__file__).resolve().parents[3]


class JobOptions(BaseModel):
    """`options` accepted by POST /v1/jobs (§11.10). Unknown keys are an error, not ignored."""

    model_config = ConfigDict(extra="forbid")

    profile: str | None = None
    tau: int | None = Field(default=None, gt=0)
    keypoints: bool = True
    verify: bool = True
    # DECISION-37, uploads: verify the summary the user reads (not every merge node) and
    # cap its length; `"all"` / null restore the experiment behaviour (per level, PRD 10%).
    verify_scope: Literal["final", "all"] = "final"
    max_summary_words: int | None = Field(default=500, gt=0)
    # DECISION-39, uploads: MiniCheck reads each claim's 5 most similar chunks; null = all.
    minicheck_top_k: int | None = Field(default=5, gt=0)
    # DECISION-40: PDF tables & formulas, shown beside page crops; never fed to the summary.
    tables: bool = True


class Stages(Protocol):
    def ingest(self, input: dict[str, Any]) -> Document: ...
    def objects(self, input: dict[str, Any]) -> list[PaperObject]: ...
    async def summarize(self, doc: Document, tau: int | None) -> SummaryResult: ...
    async def keypoints(self, doc: Document) -> list[KeyPoint]: ...
    async def verify(
        self, doc: Document, result: SummaryResult, *, final_only: bool = False
    ) -> VerifiedSummary: ...
    async def aclose(self) -> None: ...


def metrics(
    doc: Document, result: SummaryResult, verified: VerifiedSummary | None
) -> dict[str, float]:
    """FR-5.5. No claims -> no faithfulness key (JSONB has no NaN)."""
    margin = 0.0  # same estimator both sides; the margin would cancel
    out = {
        "compression_ratio": estimate_tokens(result.summary, margin)
        / max(1, estimate_tokens(doc.full_text, margin))
    }
    if verified is not None and verified.claims:
        n = len(verified.claims)
        out["faithfulness_deberta"] = (
            sum(c.label_deberta is ClaimLabel.SUPPORTED for c in verified.claims) / n
        )
        out["faithfulness_minicheck"] = (
            sum(c.label_minicheck is ClaimLabel.SUPPORTED for c in verified.claims) / n
        )
    return out


def make_pipeline(
    load_cfg: Callable[[dict[str, Any]], AppConfig],
    stages: Callable[[AppConfig], Stages],
) -> Pipeline:
    async def run(job: Job, ctx: JobContext) -> JobOutput:
        opts = JobOptions(**job.config)
        overrides: dict[str, Any] = {
            "summary": {"max_words": opts.max_summary_words},
            "verify": {"minicheck": {"top_k_chunks": opts.minicheck_top_k}},
        }
        if opts.profile:
            overrides["profile"] = opts.profile
        cfg = load_cfg(overrides)
        cfg.active_profile  # unknown profile fails here, before any work  # noqa: B018
        st = stages(cfg)
        try:
            ctx.progress("ingest", 0.05, "")
            doc = st.ingest(job.input)
            ctx.attach_document(doc)
            objects: list[PaperObject] = []
            if opts.tables and "path" in job.input:
                ctx.progress("tables", 0.1, "tables and formulas")
                objects = st.objects(job.input)
            with doc_context(doc.doc_id):
                ctx.progress("summarize", 0.15, f"{len(doc.full_text)} chars")
                result = await st.summarize(doc, opts.tau)
                kps: list[KeyPoint] = []
                if opts.keypoints:
                    ctx.progress(
                        "keypoints", 0.5, f"route {result.route.value}, {len(result.nodes)} nodes"
                    )
                    kps = await st.keypoints(doc)
                verified = None
                if opts.verify:
                    ctx.progress(
                        "verify", 0.65, f"route {result.route.value}, {len(kps)} key points"
                    )
                    verified = await st.verify(doc, result, final_only=opts.verify_scope == "final")
        finally:
            await st.aclose()
        return JobOutput(
            summary=result,
            nodes=verified.nodes if verified else summary_nodes(result),
            claims=verified.claims if verified else [],
            key_points=kps,
            metrics=metrics(doc, result, verified),
            objects=objects,
        )

    return run


class LiveStages:
    """Real models for one job. LLM client opened here, closed in `aclose`."""

    def __init__(self, cfg: AppConfig) -> None:
        from app.cli import make_provider
        from app.llm.cache import DiskCache
        from app.llm.client import LLMClient
        from app.llm.limiter import SlidingWindow

        self.cfg = cfg
        profile = cfg.active_profile
        self.provider = make_provider(profile)
        self.client = LLMClient(
            self.provider,
            profile,
            DiskCache(Path(os.getenv("LLM_CACHE_DIR", ROOT / cfg.cache.dir))),
            rpm_bucket=SlidingWindow(profile.limits.rpm),
            tpm_bucket=SlidingWindow(profile.limits.tpm),
            fallback=cfg.fallback,
        )

    async def _release(self) -> None:
        unload = getattr(self.provider, "unload", None)  # Ollama only (DECISION-3)
        if unload is not None:
            await unload(self.cfg.active_profile.model)

    def ingest(self, input: dict[str, Any]) -> Document:
        from app.ingest.grobid import GrobidClient, ingest_pdf
        from app.ingest.web import ingest_url

        if "url" in input:
            with httpx.Client(
                timeout=30, headers={"user-agent": "semantic-compression/0.1"}
            ) as http:
                return ingest_url(input["url"], http)
        with httpx.Client(timeout=300) as http:  # owned, so its socket is closed
            grobid = GrobidClient(os.getenv("GROBID_URL", "http://localhost:8070"), client=http)
            return ingest_pdf(Path(input["path"]), grobid)

    def objects(self, input: dict[str, Any]) -> list[PaperObject]:
        from app.ingest.objects import extract_objects

        trim_working_set()  # Docling needs ~2 GB; free what ingest left behind first
        return extract_objects(Path(input["path"]), cache_dir=ROOT / ".cache" / "objects")

    async def summarize(self, doc: Document, tau: int | None) -> SummaryResult:
        from app.summarize.single_pass import summarize

        return await summarize(
            doc, self.client, self.cfg, prompt_dir=ROOT / "prompts", tau_override=tau
        )

    async def keypoints(self, doc: Document) -> list[KeyPoint]:
        from app.keypoints.pipeline import extract_keypoints
        from app.verify.pipeline import openers_from_config

        return await extract_keypoints(
            doc,
            self.client,
            openers_from_config(self.cfg.verify).embedder,
            self.cfg,
            release_llm=self._release,
            prompt_dir=ROOT / "prompts",
        )

    async def verify(
        self, doc: Document, result: SummaryResult, *, final_only: bool = False
    ) -> VerifiedSummary:
        from app.verify.pipeline import openers_from_config, verify_summary

        return await verify_summary(
            doc,
            result,
            self.client,
            openers_from_config(self.cfg.verify),
            self.cfg.verify,
            release_llm=self._release,
            prompt_dir=ROOT / "prompts",
            final_only=final_only,
        )

    async def aclose(self) -> None:
        from app.llm.base import TransientError

        try:
            # the model stays loaded between calls (DECISION-35); best effort, so a dead
            # Ollama at close time never fails a finished job or hides the real error
            with contextlib.suppress(TransientError):
                await self._release()
        finally:
            await self.provider.aclose()
            trim_working_set()  # an idle worker between uploads holds almost no RAM
