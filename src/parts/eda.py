"""Descriptive analysis of Parts.csv (Task 3).  Run:  python -m src.parts.eda"""

from __future__ import annotations

import re

import pandas as pd

from .similarity import load_parts


def main() -> None:
    p = load_parts()
    d = p["DESCRIPTION"].dropna()
    print(f"Rows: {len(p)}, columns: {p.shape[1]}, unique IDs: {p['ID'].nunique()}")
    print(f"DESCRIPTION missing: {p['DESCRIPTION'].isna().sum()} ({p['DESCRIPTION'].isna().mean():.1%})")
    print(f"Rows with no information at all besides ID: {(p.drop(columns='ID').notna().sum(axis=1) == 0).sum()}")
    print(f"Non-empty descriptions: {len(d)}, distinct: {d.nunique()}, "
          f"rows sharing an identical description with another part: {d.duplicated(keep=False).sum()}")
    print(f"Description length (chars): median {d.str.len().median():.0f}, "
          f"min {d.str.len().min()}, max {d.str.len().max()}")

    print("\nMissing values per column (%):")
    print((p.isna().mean() * 100).round(1).sort_values().to_string())

    print("\nPart families (first two words of DESCRIPTION):")
    print(d.str.split().str[:2].str.join(" ").value_counts().head(8).to_string())

    # Each description looks like two concatenated sources ("... Movement 1.6A ..." and
    # "Electric Indicator, Very Fast Blow, 1.6A, ...") - check if they agree.
    first = d.str.extract(r"Movement (\d+\.?\d*)A")[0].astype(float)
    second = d.str.extract(r"Blow, (\d+\.?\d*)A")[0].astype(float)
    both = first.notna() & second.notna()
    print(f"\nCurrent stated twice in {both.sum()} descriptions, contradicting in {(first[both] != second[both]).sum()}:")
    print(d[both & (first != second)].head(3).to_string())

    speed1 = d.str.extract(r"Indicator \w+ (Very Fast|Fast|Slow Blow|Medium Time Delay) Movement")[0]
    speed2 = d.str.extract(r"Indicator, (Very Fast|Fast|Time Lag|Medium Time Lag) Blow")[0]
    mapping = {"Slow Blow": "Time Lag", "Medium Time Delay": "Medium Time Lag"}
    both = speed1.notna() & speed2.notna()
    print(f"Blow speed stated twice in {both.sum()} descriptions, contradicting in "
          f"{(speed1[both].replace(mapping) != speed2[both]).sum()}")

    print("\nInconsistent unit formatting, e.g. Size column:")
    print(p["Size"].value_counts().loc[lambda s: s.index.str.contains(r"5\s*[xX]\s*20")].to_string())
    print("Units embedded in values, e.g.:", list(p["Operating Temperature-Max (Cel)"].dropna().unique()[:3]),
          list(p["Rated Current (A)"].dropna().unique()[:3]))
    cur = p["Rated Current (A)"].dropna().map(lambda v: float(re.sub(r"[^\d.]", "", v) or "nan"))
    print(f"Rated current spans {cur.min()}A - {cur.max()}A ({cur.nunique()} distinct values)")


if __name__ == "__main__":
    main()
