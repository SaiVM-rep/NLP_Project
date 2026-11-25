"""FR-5.3: MiniCheck verifier (Flan-T5-Large variant). PRD §11.8 step 5.

Reimplements upstream Liyan06/MiniCheck@b58b9fa (Apache-2.0) `Inferencer` for
`lytang/MiniCheck-Flan-T5-Large` rather than installing the package, which pulls
openai/nltk/datasets and uses device_map="auto" (DECISION-13). Differences, on purpose:
sentences come from our pySBD spans instead of nltk, and chunks are spans over
`full_text` so they can be highlighted (SC2).

Upstream behaviour kept: chunks of <= 500 whitespace words, input
"predict: {chunk}</s>{claim}", one decoder step, softmax over label tokens 3 / 209,
max over chunks.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.core.schemas import Document, Span
from app.verify.residency import GpuResident

# (chunks, claims) -> (N,) probability that the chunk supports the claim
SupportFn = Callable[[list[str], list[str]], NDArray[np.float32]]

CHUNK_WORDS = 500
MAX_INPUT_TOKENS = 2048
# Flan-T5 vocab ids upstream reads at the first decoder step: 3 = unsupported, 209 = supported.
LABEL_TOKENS = [3, 209]


def chunk_spans(doc: Document, max_words: int = CHUNK_WORDS) -> list[Span]:
    if max_words <= 0:
        raise ValueError(f"max_words must be positive, got {max_words}")
    chunks: list[Span] = []
    first: Span | None = None
    last: Span | None = None
    words = 0
    for s in doc.sentences:
        n = len(doc.full_text[s.start : s.end].split())
        if first is not None and last is not None and words + n > max_words:
            chunks.append(Span(start=first.start, end=last.end))
            first, words = None, 0
        if first is None:
            first = s
        last = s
        words += n
    if first is not None and last is not None:
        chunks.append(Span(start=first.start, end=last.end))
    return chunks


def pick_chunks(
    sim: NDArray[np.float32], doc: Document, chunks: list[Span], k: int
) -> list[list[int]]:
    """Per claim, the k chunks holding its most similar sentence, best first.

    `sim` is claims x doc.sentences cosine similarity; ties keep document order.
    """
    starts = np.array([c.start for c in chunks])
    owner = np.searchsorted(starts, [s.start for s in doc.sentences], side="right") - 1
    best = np.full((sim.shape[0], len(chunks)), -np.inf)
    for i in range(sim.shape[0]):
        np.maximum.at(best[i], owner, sim[i])
    order = np.argsort(-best, axis=1, kind="stable")[:, :k]
    return [[int(j) for j in row] for row in order]


def support_score(claim: str, chunks: list[str], support: SupportFn) -> float:
    """Max support probability over chunks; 0.0 when there is nothing to check against."""
    if not chunks:
        return 0.0
    p = support(chunks, [claim] * len(chunks))
    return float(np.clip(p.max(), 0.0, 1.0))


class MiniCheckT5(GpuResident):
    """Callable `SupportFn` over MiniCheck-Flan-T5-Large."""

    def __init__(self, model: Any, tokenizer: Any, *, device: str, batch_size: int = 16) -> None:
        super().__init__(model, device=device)
        self._tok = tokenizer
        self._batch = batch_size

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        revision: str | None = None,
        device: str = "cuda",
        local_files_only: bool = False,
        **kw: Any,
    ) -> MiniCheckT5:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # T5 overflows in fp16 (NaN); bf16 keeps fp32's range at half the memory.
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        # The checkpoint is .bin-only: transformers would otherwise start a background
        # thread asking huggingface.co for a safetensors conversion PR, ignoring
        # local_files_only (FR-5.6), and possibly requesting one on someone else's repo.
        os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
        tok = AutoTokenizer.from_pretrained(
            checkpoint, revision=revision, local_files_only=local_files_only
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint, revision=revision, dtype=dtype, local_files_only=local_files_only
        )
        return cls(model, tok, device=device, **kw)

    def __call__(self, chunks: list[str], claims: list[str]) -> NDArray[np.float32]:
        model = self.model
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        import torch

        texts = [
            f"predict: {c}{self._tok.eos_token}{h}" for c, h in zip(chunks, claims, strict=True)
        ]
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self._batch):
                enc = self._tok(
                    texts[i : i + self._batch],
                    max_length=MAX_INPUT_TOKENS,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                start = torch.zeros(
                    (enc["input_ids"].size(0), 1), dtype=torch.long, device=self._device
                )
                logits = model(**enc, decoder_input_ids=start).logits[:, 0]
                probs = torch.softmax(logits[:, LABEL_TOKENS].float(), dim=-1)[:, 1]
                out.append(probs.cpu().numpy())
        # Every call pads to a new shape; without this the cached reserve grew past the
        # 8 GB card (2.9 -> 6.5 GB in 8 claims) and spilled into RAM, ~10x slower.
        self.trim()
        return np.concatenate(out).astype(np.float32)
