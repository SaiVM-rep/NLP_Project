"""Sentence-transformer embedder: the real `Embed` for `evidence.retrieve_evidence` (FR-5.2).

Normalisation is left to the caller: `evidence.top_n` already takes cosine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.verify.residency import GpuResident


class SentenceEmbedder(GpuResident):
    def __init__(self, model: Any, *, device: str, batch_size: int = 64) -> None:
        super().__init__(model, device=device)
        self._batch = batch_size
        # sentence-transformers 6 name; the old get_sentence_embedding_dimension warns
        self._dim = int(model.get_embedding_dimension())

    @classmethod
    def from_pretrained(
        cls, checkpoint: str, *, revision: str | None = None, device: str = "cuda", **kw: Any
    ) -> SentenceEmbedder:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(checkpoint, revision=revision, device=device)
        return cls(model, device=device, **kw)

    def __call__(self, texts: list[str]) -> NDArray[np.float32]:
        model = self.model
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vecs = model.encode(
            texts, batch_size=self._batch, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)
