"""FR-5.3: DeBERTa-v3 NLI verifier. PRD §11.8 step 4.

Each claim is scored against its top-n evidence sentences (premise = source sentence,
hypothesis = claim). Max entail and max contradict are taken independently across premises.

The model call is injected (`ProbsFn`) so the aggregation is testable without torch.
Label order differs between checkpoints, so it is read from `config.id2label`, never assumed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from app.verify.residency import GpuResident

# (premises, hypotheses) -> (N, num_labels) softmax probabilities
ProbsFn = Callable[[list[str], list[str]], NDArray[np.float32]]


class LabelIdx(NamedTuple):
    entail: int
    contra: int


def label_indices(id2label: Mapping[int, str] | Mapping[str, str]) -> LabelIdx:
    by_name = {str(name).lower(): int(i) for i, name in id2label.items()}
    missing = [n for n in ("entailment", "contradiction") if n not in by_name]
    if missing:
        raise ValueError(f"NLI head has no {missing} label: {dict(id2label)}")
    return LabelIdx(entail=by_name["entailment"], contra=by_name["contradiction"])


def score_claim(
    claim: str, premises: list[str], probs: ProbsFn, idx: LabelIdx
) -> tuple[float, float]:
    """(max_entail, max_contra) over premises; (0, 0) when there is no evidence."""
    if not premises:
        return 0.0, 0.0
    p = probs(premises, [claim] * len(premises))
    entail = float(np.clip(p[:, idx.entail].max(), 0.0, 1.0))
    contra = float(np.clip(p[:, idx.contra].max(), 0.0, 1.0))
    return entail, contra


class DebertaNLI(GpuResident):
    """Callable `ProbsFn` over a HF sequence-classification NLI model."""

    def __init__(self, model: Any, tokenizer: Any, *, device: str, batch_size: int = 32) -> None:
        super().__init__(model, device=device)
        self._tok = tokenizer
        self._batch = batch_size
        self.label_idx = label_indices(model.config.id2label)
        self._num_labels = int(model.config.num_labels)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        revision: str | None = None,
        device: str = "cuda",
        local_files_only: bool = False,
        **kw: Any,
    ) -> DebertaNLI:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        tok = AutoTokenizer.from_pretrained(
            checkpoint, revision=revision, local_files_only=local_files_only
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, revision=revision, dtype=dtype, local_files_only=local_files_only
        )
        return cls(model, tok, device=device, **kw)

    def __call__(self, premises: list[str], hyps: list[str]) -> NDArray[np.float32]:
        model = self.model
        if not premises:
            return np.zeros((0, self._num_labels), dtype=np.float32)
        import torch

        out = []
        with torch.inference_mode():
            for i in range(0, len(premises), self._batch):
                enc = self._tok(
                    premises[i : i + self._batch],
                    hyps[i : i + self._batch],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                logits = model(**{k: v.to(self._device) for k, v in enc.items()}).logits
                out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
