"""EXPLORATORY: rerank MiniLM's top-5 attribution candidates with DeBERTa NLI.

    .venv/Scripts/python.exe scripts/attr_rerank.py [sbert frank true]
    .venv/Scripts/python.exe scripts/attr_score.py --system rerank

Writes results/attribution/<name>.rerank.json (key point rank -> sentence id), the same
shape as <name>.system.json. Premise = candidate sentence, hypothesis = key point. Scored on
the same 24 labelled key points the baseline was measured on, so there is no held-out set:
report it as exploratory (DECISION-32), not as a tuned improvement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.core.config import load_config  # noqa: E402
from app.verify.embedder import SentenceEmbedder  # noqa: E402
from app.verify.nli import DebertaNLI, label_indices  # noqa: E402
from eval.attribution import rerank  # noqa: E402

OUT = ROOT / "results" / "attribution"
TOP = 5


def main() -> int:
    names = sys.argv[1:] or ["sbert", "frank", "true"]
    v = load_config(ROOT / "configs").verify
    device = "cuda"
    papers = {}
    for name in names:
        b = json.loads((ROOT / "demo" / f"{name}.json").read_text(encoding="utf-8"))
        full = b["document"]["full_text"]
        sents = [full[s["start"] : s["end"]] for s in b["document"]["sentences"]]
        papers[name] = (sents, b["result"]["key_points"])

    # phase 1: retriever (same as the system and the labelling sheets)
    cands: dict[str, list[list[int]]] = {}
    with SentenceEmbedder.from_pretrained(
        v.embedder.checkpoint, revision=v.embedder.revision, device=device, local_files_only=True
    ) as embed:
        for name, (sents, kps) in papers.items():
            sv = embed(sents)
            kv = embed([k["text"] for k in kps])
            sv = sv / np.linalg.norm(sv, axis=1, keepdims=True)
            kv = kv / np.linalg.norm(kv, axis=1, keepdims=True)
            sims = kv @ sv.T
            cands[name] = [[int(j) for j in np.argsort(-row)[:TOP]] for row in sims]

    # phase 2: NLI over the candidates (one model resident at a time, DECISION-3)
    with DebertaNLI.from_pretrained(
        v.nli.checkpoint, revision=v.nli.revision, device=device, local_files_only=True
    ) as nli:
        entail_ix = label_indices(nli.model.config.id2label).entail
        for name, (sents, kps) in papers.items():
            picks: dict[str, int] = {}
            for k, cs in zip(kps, cands[name], strict=True):
                probs = nli([sents[j] for j in cs], [k["text"]] * len(cs))
                picks[str(k["rank"])] = rerank(cs, [float(p) for p in probs[:, entail_ix]])
            (OUT / f"{name}.rerank.json").write_text(json.dumps(picks), encoding="utf-8")
            print(f"{name}: {picks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
