"""Write the 5 most similar alternatives for every part.

Run:  python -m src.parts.find_alternatives [--k 5]
Output: reports/task3/alternatives.csv (long format, one row per part/alternative)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .similarity import PartsSimilarity, load_parts

OUT = Path(__file__).resolve().parents[2] / "reports" / "task3" / "alternatives.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    model = PartsSimilarity().fit(load_parts())
    table = model.all_alternatives(args.k)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"Wrote {len(table)} rows for {table['ID'].nunique()} parts to {OUT}")
    print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
