"""Train and evaluate classifiers for the "Type" column (Task 2).

Run:  python -m src.classification.train
Writes metrics and figures to reports/task2/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC, load_dataset

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "task2"
SEED = 42


def make_preprocessor(scale: bool) -> ColumnTransformer:
    numeric = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        # CDx / NKJUD are heavily right-skewed -> log1p before scaling for linear models
        from sklearn.preprocessing import FunctionTransformer

        numeric += [("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATEGORICAL),
        ]
    )


def make_models() -> dict[str, Pipeline]:
    return {
        "baseline_majority": Pipeline(
            [("prep", make_preprocessor(False)), ("clf", DummyClassifier(strategy="prior"))]
        ),
        "logistic_regression": Pipeline(
            [
                ("prep", make_preprocessor(True)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", make_preprocessor(False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", make_preprocessor(False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=400,
                        early_stopping=True,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def scores(y_true, proba, threshold) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "f1_n": f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "threshold": threshold,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset()
    print(f"Merged dataset: {X.shape[0]} rows, {X.shape[1]} features, positive ('n') rate {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results: dict[str, dict] = {}
    for name, model in make_models().items():
        # Out-of-fold probabilities on the training set -> CV metrics + threshold choice
        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold = 0.5 if name == "baseline_majority" else best_f1_threshold(y_train, oof)
        results[name] = {"cv": scores(y_train, oof, threshold)}
        print(f"{name:24s} CV  " + "  ".join(f"{k}={v:.3f}" for k, v in results[name]["cv"].items()))

    best_name = max(
        (n for n in results if n != "baseline_majority"), key=lambda n: results[n]["cv"]["pr_auc"]
    )
    print(f"\nSelected model (best CV PR-AUC): {best_name}")

    # Refit every model on the full training split; report hold-out test metrics
    fitted = {}
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model
        proba = model.predict_proba(X_test)[:, 1]
        results[name]["test"] = scores(y_test, proba, results[name]["cv"]["threshold"])

    best = fitted[best_name]
    proba = best.predict_proba(X_test)[:, 1]
    threshold = results[best_name]["cv"]["threshold"]
    pred = (proba >= threshold).astype(int)
    report = classification_report(y_test, pred, target_names=["y", "n"], digits=3)
    cm = confusion_matrix(y_test, pred)
    print(report)
    print("Confusion matrix (rows=true y/n, cols=pred y/n):\n", cm)

    imp = permutation_importance(
        best, X_test, y_test, scoring="average_precision", n_repeats=20, random_state=SEED, n_jobs=-1
    )
    importance = (
        pd.DataFrame({"feature": X.columns, "mean": imp.importances_mean, "std": imp.importances_std})
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )
    print(importance.to_string(index=False))

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, model in fitted.items():
        if name == "baseline_majority":
            continue
        p = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, p)
        axes[0].plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, p):.3f})")
    axes[0].axhline(y_test.mean(), ls="--", c="grey", label="chance")
    axes[0].set(xlabel="Recall (class n)", ylabel="Precision (class n)", title="Precision-recall, hold-out test")
    axes[0].legend(fontsize=8)
    axes[1].barh(importance.feature[::-1], importance["mean"][::-1], xerr=importance["std"][::-1])
    axes[1].set(title=f"Permutation importance ({best_name})", xlabel="Drop in average precision")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task2_results.png", dpi=120)

    (OUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "results": results,
                "confusion_matrix_test": cm.tolist(),
                "classification_report_test": report,
                "permutation_importance": importance.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    print(f"\nWrote {OUT_DIR / 'metrics.json'} and task2_results.png")


if __name__ == "__main__":
    main()
