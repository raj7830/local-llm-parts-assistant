"""Find alternative parts based on the DESCRIPTION column of Parts.csv.

Approach (hybrid lexical + numeric):
1. Normalise descriptions (case, unit spelling, "5 X 20mm" vs "5x20mm", synonyms).
   Parts without a description get a synthetic one built from their structured
   attributes so that every part can still be matched (flagged as lower confidence).
2. TF-IDF on word 1-2-grams + character 3-5-grams; cosine similarity.
   Char n-grams make the model robust to spelling/format variants and partially
   match codes such as "cULus" / "UL" or "0.25a" / "0.2a".
3. The rated current is parsed from the text and compared as a min/max ratio -
   plain bag-of-words cannot tell that 1.6 A is closer to 2 A than to 10 A, and
   the current rating is the first thing a replacement part has to match.
4. Final score = 0.7 * text similarity + 0.3 * current similarity.
   The weights were chosen with the proxy evaluation in ``evaluate.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "Parts.csv"

# Structured columns used only to synthesise a text for parts with no DESCRIPTION.
FALLBACK_COLUMNS = [
    "Attribut1", "Characteristic", "Rated Current (A)", "Rating", "Rated Voltage (V)",
    "Maximum AC Voltage Rating", "Maximum DC Voltage Rating", "Rated Breaking Capacity (A)",
    "Mounting", "Mounting Feature", "Size", "Material", "Application",
]

SYNONYMS = [
    (r"\bslow blow\b", "time lag"),
    (r"\btime delay\b", "time lag"),
    (r"\bsubred\b", "sub red"),
    (r"\bsmd\b", "surface mount"),
    (r"\bindicators\b", "indicator"),
    (r"\bt/r\b", "tape reel"),
]

_NUM = r"(\d+(?:\.\d+)?)"


def load_parts(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", dtype=str)


def normalise(text: str) -> str:
    t = text.lower()
    t = re.sub(r"(\d)\s*x\s*(\d)", r"\1x\2", t)          # 5 X 20mm -> 5x20mm
    t = re.sub(r"(\d)\s+(mm|a|v|vac|vdc|w|j|ms)\b", r"\1\2", t)  # 250 V -> 250v
    t = t.replace("/", " / ")
    for pattern, repl in SYNONYMS:
        t = re.sub(pattern, repl, t)
    t = re.sub(r"[,;()]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def synthesise_description(row: pd.Series) -> str:
    values = [str(row[c]) for c in FALLBACK_COLUMNS if c in row and pd.notna(row[c])]
    return " ".join(dict.fromkeys(values))  # de-duplicate, keep order


def parse_current(text: str) -> float | None:
    """Rated current in A: first "<number>a" that is not a breaking capacity "(IR)"."""
    m = re.search(_NUM + r"a\b(?!\s*\(?ir)", text.lower())
    return float(m.group(1)) if m else None


def _ratio_similarity(values: np.ndarray, target: float) -> np.ndarray:
    """min/max ratio: 1 for equal values, 0.5 when a factor of two apart; 0 if unknown."""
    target = max(target, 1e-6)
    v = np.maximum(np.nan_to_num(values, nan=0.0), 0.0)
    sim = np.minimum(v, target) / np.maximum(v, target)
    return np.where(np.isnan(values), 0.0, sim)


class PartsSimilarity:
    def __init__(self, text_weight: float = 0.7):
        self.text_weight = text_weight

    # ------------------------------------------------------------------ fitting
    def fit(self, parts: pd.DataFrame) -> "PartsSimilarity":
        self.parts = parts.reset_index(drop=True).copy()
        has_desc = self.parts["DESCRIPTION"].notna() & (self.parts["DESCRIPTION"].str.strip() != "")
        raw = self.parts["DESCRIPTION"].where(has_desc, self.parts.apply(synthesise_description, axis=1))
        self.parts["description_source"] = np.where(has_desc, "description", "attributes")
        self.parts.loc[raw.str.strip().eq(""), "description_source"] = "empty"
        self.parts["text"] = raw.fillna("").map(normalise)

        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[^\s]+")
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self._vectorise(self.parts["text"], fit=True)

        self.current = self.parts["text"].map(parse_current).to_numpy(dtype=float)  # NaN if unknown
        self.index_of = {pid: i for i, pid in enumerate(self.parts["ID"])}
        return self

    def _vectorise(self, texts, fit: bool = False):
        if fit:
            w, c = self.word_vec.fit_transform(texts), self.char_vec.fit_transform(texts)
        else:
            w, c = self.word_vec.transform(texts), self.char_vec.transform(texts)
        return normalize(hstack([w, c]).tocsr())

    # ---------------------------------------------------------------- scoring
    def _scores(self, query_vec, query_current: float | None) -> np.ndarray:
        text_sim = (self.matrix @ query_vec.T).toarray().ravel()
        if query_current is None or np.isnan(query_current):
            return text_sim  # nothing to compare numerically -> text only
        current_sim = _ratio_similarity(self.current, query_current)
        return self.text_weight * text_sim + (1 - self.text_weight) * current_sim

    def _top_k(self, score, k, exclude=None) -> pd.DataFrame:
        score = score.copy()
        if exclude is not None:
            score[exclude] = -np.inf
        score[self.parts["description_source"].eq("empty").to_numpy()] = -np.inf
        idx = np.argsort(-score, kind="stable")[:k]
        out = self.parts.loc[idx, ["ID", "DESCRIPTION", "description_source"]].copy()
        out.insert(1, "score", score[idx].round(4))
        return out.reset_index(drop=True)

    def alternatives(self, part_id: str, k: int = 5) -> pd.DataFrame:
        """Top-k alternative parts for an existing part ID (the part itself excluded)."""
        if part_id not in self.index_of:
            raise KeyError(f"Unknown part ID {part_id!r}")
        i = self.index_of[part_id]
        score = self._scores(self.matrix[i], self.current[i])
        return self._top_k(score, k, exclude=i)

    def search(self, query: str, k: int = 5) -> pd.DataFrame:
        """Top-k parts for a free-text description (used by the chatbot)."""
        text = normalise(query)
        score = self._scores(self._vectorise([text]), parse_current(text))
        return self._top_k(score, k)

    def all_alternatives(self, k: int = 5) -> pd.DataFrame:
        rows = []
        for pid, source in zip(self.parts["ID"], self.parts["description_source"]):
            if source == "empty":
                rows.append({"ID": pid, "rank": None, "alternative_ID": None, "score": None, "query_source": source})
                continue
            for rank, alt in enumerate(self.alternatives(pid, k).itertuples(), 1):
                rows.append({"ID": pid, "rank": rank, "alternative_ID": alt.ID, "score": alt.score, "query_source": source})
        return pd.DataFrame(rows).astype({"rank": "Int64"})
