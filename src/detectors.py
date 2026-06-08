"""Train detectors and report 5-fold stratified CV.

Imbalance handling (dataset is ~1:4.44 phish:ham; we do NOT subsample):
  - LogisticRegression, RandomForest: class_weight="balanced".
  - GradientBoosting: no class_weight param, so per-sample weights via
    compute_sample_weight("balanced", y) are passed to fit().

Cross-validation is done manually (StratifiedKFold) so that GradientBoosting's
sample_weight is recomputed and applied correctly within each fold — avoiding the
metadata-routing pitfalls of cross_validate with fit params.

Outputs:
  results/tables/cv_baseline.csv   (model x metric: mean, std over 5 folds)
  results/models/{logreg,random_forest,gradient_boosting}.joblib  (refit on full train)
"""

from __future__ import annotations

from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from . import config, features
from .metrics import METRIC_NAMES, compute_metrics

# Models that need explicit per-sample weights (no class_weight param).
USES_SAMPLE_WEIGHT = {"gradient_boosting"}


def model_factories() -> dict[str, callable]:
    """Fresh, unfitted estimators (seeded). Factories so CV gets clean instances."""
    return {
        "logreg": lambda: LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=config.SEED,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=config.SEED,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=200,
            max_features="sqrt",
            random_state=config.SEED,
        ),
    }


def _fit(name: str, clf, x, y):
    if name in USES_SAMPLE_WEIGHT:
        clf.fit(x, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        clf.fit(x, y)
    return clf


def cross_validate() -> pd.DataFrame:
    x_train, y_train, _, _ = features.load_features()
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    factories = model_factories()

    rows: list[dict] = []
    for name, factory in factories.items():
        per_metric: dict[str, list[float]] = defaultdict(list)
        for fold, (tr, va) in enumerate(skf.split(x_train, y_train)):
            clf = _fit(name, factory(), x_train[tr], y_train[tr])
            proba = clf.predict_proba(x_train[va])[:, 1]
            for k, v in compute_metrics(y_train[va], proba).items():
                per_metric[k].append(v)
        for metric in METRIC_NAMES:
            vals = per_metric[metric]
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)),
                }
            )
        print(
            f"  {name}: "
            + ", ".join(
                f"{m}={np.mean(per_metric[m]):.4f}±{np.std(per_metric[m], ddof=1):.4f}"
                for m in METRIC_NAMES
            )
        )

    df = pd.DataFrame(rows)
    out = config.TABLES_DIR / "cv_baseline.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    return df


def fit_full_and_save() -> None:
    x_train, y_train, _, _ = features.load_features()
    for name, factory in model_factories().items():
        clf = _fit(name, factory(), x_train, y_train)
        path = config.MODELS_DIR / f"{name}.joblib"
        joblib.dump(clf, path)
        print(f"  refit on full train -> {path}")


def main() -> None:
    config.ensure_dirs()
    features.ensure_features()
    print(
        f"5-fold stratified CV (seed={config.SEED}, imbalance handled via "
        f"class_weight / sample_weight):"
    )
    cross_validate()
    print("\nRefitting on full train and persisting models:")
    fit_full_and_save()


if __name__ == "__main__":
    main()
