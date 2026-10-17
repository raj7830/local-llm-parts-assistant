"""Proxy evaluation of the parts-similarity variants (there is no ground truth).

For every part with a real DESCRIPTION we check how often its top-5 alternatives
agree with it on the structured attributes an engineer would require for a
drop-in replacement: rated current, blow characteristic, mounting type and size.
These columns are *not* used by the matcher for parts that have a description,
so they act as an independent sanity check.

* per-attribute columns: share of top-5 alternatives with the same value
* ``all_match``: share of top-5 alternatives matching on *every* known attribute
  (queries with at least 3 of the 4 attributes known)
* ``oracle``: best achievable ``all_match`` - for most parts fewer than five
  genuine drop-in replacements exist in the data

Run:  python -m src.parts.evaluate [--embeddings]
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from .similarity import PartsSimilarity, _ratio_similarity, load_parts

ATTRS = ["Rated Current (A)", "Characteristic", "Mounting", "Size"]
K = 5


def _norm_attr(v):
    if pd.isna(v):
        return np.nan
    return re.sub(r"\s*x\s*", "x", str(v).strip().lower())


def evaluate(score: np.ndarray, attrs: np.ndarray, queries, invalid) -> dict[str, float]:
    per_attr = {a: [] for a in ATTRS}
    joint, oracle = [], []
    for i in queries:
        s = score[i].copy()
        s[i] = -np.inf
        s[invalid] = -np.inf
        top = np.argsort(-s, kind="stable")[:K]
        known = [j for j in range(len(ATTRS)) if not pd.isna(attrs[i, j])]
        for j in known:
            vals = attrs[top, j]
            vals = vals[~pd.isna(vals)]
            if len(vals):
                per_attr[ATTRS[j]].append(np.mean(vals == attrs[i, j]))
        if len(known) >= 3:
            ok = ~invalid.copy()
            for j in known:
                ok &= attrs[:, j] == attrs[i, j]
            ok[i] = False
            joint.append(ok[top].mean())
            oracle.append(min(ok.sum(), K) / K)
    out = {a: np.mean(v) for a, v in per_attr.items()}
    out["all_match"] = np.mean(joint)
    out["oracle"] = np.mean(oracle)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", action="store_true", help="also evaluate Ollama embeddinggemma")
    args = ap.parse_args()

    model = PartsSimilarity().fit(load_parts())
    parts = model.parts
    attrs = parts[ATTRS].apply(lambda c: c.map(_norm_attr)).to_numpy(dtype=object)
    queries = np.flatnonzero(parts["description_source"].eq("description"))
    invalid = parts["description_source"].eq("empty").to_numpy()

    text = (model.matrix @ model.matrix.T).toarray()
    current = np.array([_ratio_similarity(model.current, c) for c in model.current])
    current[np.isnan(model.current)] = 0.0  # query current unknown -> no numeric signal

    variants = {
        "random baseline": np.random.default_rng(0).random(text.shape),
        "tfidf word+char (text only)": text,
        "text 0.8 / current 0.2": 0.8 * text + 0.2 * current,
        "text 0.7 / current 0.3 (default)": 0.7 * text + 0.3 * current,
        "text 0.6 / current 0.4": 0.6 * text + 0.4 * current,
    }
    if args.embeddings:
        import ollama

        texts = [t or "unknown" for t in parts["text"]]  # API rejects empty strings
        emb = np.array(ollama.embed(model="embeddinggemma-2:270m", input=texts)["embeddings"])
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        semantic = emb @ emb.T
        variants["embeddinggemma (semantic only)"] = semantic
        variants["0.5 tfidf + 0.5 emb"] = 0.5 * text + 0.5 * semantic
        variants["0.7*(0.5 tfidf + 0.5 emb) + 0.3 current"] = 0.7 * (0.5 * text + 0.5 * semantic) + 0.3 * current

    table = pd.DataFrame({n: evaluate(s, attrs, queries, invalid) for n, s in variants.items()}).T
    print(f"Top-{K} agreement with structured attributes ({len(queries)} queries with a real description):\n")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
