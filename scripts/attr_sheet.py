"""Build key-point attribution sheets from the demo bundles (blind: the system's pick is hidden).

    .venv/Scripts/python.exe scripts/attr_sheet.py [sbert frank true]

Per paper, in results/attribution/:
  <name>.sheet.csv      key point + 5 candidate sentences (shuffled), columns to fill
  <name>.sentences.csv  every sentence with its id, for searching when no candidate fits
  <name>.system.json    the system's pick per key point (sentence id or null): DO NOT open
                        while labelling; scripts/attr_score.py reads it
Candidates = top 5 by MiniLM cosine (the same retriever the system uses), run on CPU.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.core.config import load_config  # noqa: E402
from app.verify.embedder import SentenceEmbedder  # noqa: E402
from eval.attribution import sentence_index  # noqa: E402

OUT = ROOT / "results" / "attribution"
TOP = 5


def main() -> int:
    names = sys.argv[1:] or ["sbert", "frank", "true"]
    e = load_config(ROOT / "configs").verify.embedder
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    with SentenceEmbedder.from_pretrained(
        e.checkpoint, revision=e.revision, device="cpu", local_files_only=True
    ) as embed:
        for name in names:
            b = json.loads((ROOT / "demo" / f"{name}.json").read_text(encoding="utf-8"))
            full, spans = b["document"]["full_text"], b["document"]["sentences"]
            sents = [full[s["start"] : s["end"]] for s in spans]
            kps = b["result"]["key_points"]
            sv = embed(sents)
            kv = embed([k["text"] for k in kps])
            sv = sv / np.linalg.norm(sv, axis=1, keepdims=True)
            kv = kv / np.linalg.norm(kv, axis=1, keepdims=True)
            sims = kv @ sv.T
            with (OUT / f"{name}.sheet.csv").open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["kp", "key_point", "cand_id", "candidate_sentence", "supports"])
                for i, k in enumerate(kps):
                    cands = [int(j) for j in np.argsort(-sims[i])[:TOP]]
                    rng.shuffle(cands)
                    for n, j in enumerate(cands):
                        w.writerow([k["rank"], k["text"] if n == 0 else "", j, sents[j], ""])
                    w.writerow([k["rank"], "", "", "other_ids: | unsupported: | notes:", ""])
            with (OUT / f"{name}.sentences.csv").open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["id", "sentence"])
                w.writerows(enumerate(sents))
            system = {str(k["rank"]): sentence_index(spans, k["evidence"]) for k in kps}
            (OUT / f"{name}.system.json").write_text(json.dumps(system), encoding="utf-8")
            print(f"{name}: {len(kps)} key points, {len(sents)} sentences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
