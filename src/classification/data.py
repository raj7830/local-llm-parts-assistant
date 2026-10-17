"""Load, clean and merge the two tables for the binary "Type" classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TARGET = "Type"
POSITIVE = "n"  # minority class (~7.5 %) - treated as the class of interest

# Columns found to be exact re-encodings of other columns (see EDA in the report).
REDUNDANT = {
    "ERZ": "1:1 relabelling of GJAH (oooo<->www, rrr<->pppp, ii<->nnn, same NaNs)",
    "LPI": "exactly NKJUD * 10 000",
}
CATEGORICAL = ["GJAH", "ZIK", "HUI", "BJZHD", "BJKG", "ZUB", "VOL", "UIO", "VBNM"]
NUMERIC = ["UKL", "CDx", "NKJUD", "POUG", "TRE", "OIN"]


def load_tables(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    t1 = pd.read_csv(data_dir / "table_1.csv", sep=";")
    t2 = pd.read_csv(data_dir / "table_2.csv", sep=";")
    return t1, t2


def merge_tables(t1: pd.DataFrame, t2: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, check ID uniqueness, then inner-join on ID.

    Both tables contain 370 fully duplicated rows. Joining them as-is would create
    a many-to-many join (4 rows for each duplicated ID) and leak identical rows
    into both train and test splits, so they are removed first.
    """
    t1 = t1.drop_duplicates()
    t2 = t2.drop_duplicates()
    for name, t in (("table_1", t1), ("table_2", t2)):
        if t["ID"].duplicated().any():
            raise ValueError(f"{name} has conflicting rows for the same ID")
    merged = t1.merge(t2, on="ID", how="inner", validate="one_to_one")
    return merged.sort_values("ID").reset_index(drop=True)


def load_dataset(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    merged = merge_tables(*load_tables(data_dir))
    X = merged[CATEGORICAL + NUMERIC].copy()
    X[CATEGORICAL] = X[CATEGORICAL].astype(object).where(X[CATEGORICAL].notna(), "missing")
    y = (merged[TARGET] == POSITIVE).astype(int).rename("is_n")
    return X, y
