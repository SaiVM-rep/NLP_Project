"""BERTScore F1 (PRD §12.5): comparability only, never evidence of faithfulness.

bert-score loads models by name with no revision, so the pinned snapshot is downloaded
first and its local path handed over; layer and rescale baseline are passed explicitly
because bert-score can no longer infer them from a path.

bert-score silently truncates inputs at the model max (512 tokens for RoBERTa). A 10%
summary of a median arXiv paper is ~660 tokens, so truncation is counted and logged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.config import BertScoreCfg
from app.verify.residency import GpuResident

log = logging.getLogger(__name__)

FILES = [
    "config.json",
    "model.safetensors",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
]


class BertScore(GpuResident):
    def __init__(self, scorer: Any, *, device: str) -> None:
        super().__init__(scorer._model, device=device)
        self._scorer = scorer
        self._max = int(scorer._tokenizer.model_max_length)
        self.truncated = 0  # inputs cut in the last call

    @classmethod
    def from_config(cls, cfg: BertScoreCfg, *, device: str = "cuda") -> BertScore:
        import bert_score
        from huggingface_hub import snapshot_download

        path = snapshot_download(cfg.checkpoint, revision=cfg.revision, allow_patterns=FILES)
        # bert-score switches to a T5 loader if "t5" appears anywhere in model_type
        if "t5" in path.lower():
            raise ValueError(f"snapshot path {path!r} contains 't5'; move HF_HOME")
        baseline = Path(bert_score.__file__).parent / "rescale_baseline" / "en"
        scorer = bert_score.BERTScorer(
            model_type=path,
            num_layers=cfg.num_layers,
            lang="en",
            rescale_with_baseline=True,
            baseline_path=str(baseline / f"{cfg.baseline}.tsv"),
            batch_size=cfg.batch_size,
            device=device,
        )
        return cls(scorer, device=device)

    def _too_long(self, text: str) -> bool:
        return len(self._scorer._tokenizer.encode(text, add_special_tokens=True)) > self._max

    def __call__(self, cands: list[str], refs: list[str]) -> list[float]:
        _ = self.model  # raises once closed
        if len(cands) != len(refs):
            raise ValueError(f"{len(cands)} candidates vs {len(refs)} references")
        if not cands:
            return []
        self.truncated = sum(self._too_long(t) for t in [*cands, *refs])
        if self.truncated:
            log.warning(
                "bertscore truncated %d of %d inputs at %d tokens",
                self.truncated,
                2 * len(cands),
                self._max,
                extra={"truncated": self.truncated},
            )
        _, _, f1 = self._scorer.score(cands, refs)
        return [float(v) for v in f1.tolist()]

    def close(self) -> None:
        self._scorer = None
        super().close()
