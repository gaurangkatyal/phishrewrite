"""URL-ablation + fixed-FPR robustness analysis (companion to evaluate.py).

Why this exists
---------------
The label-validity contract FREEZES every URL in a rewrite. A detector that
keys on the URLs themselves (domain tokens like "paypal"/"ebay", or the raw
url-count feature) therefore keeps a stable signal even when the surrounding prose
is fully rewritten — which can make the *text-only* weakness look smaller than it
is. To isolate the text signal we train URL-BLIND detectors: every URL is masked
out of the text (and the handcrafted url-count consequently goes to zero) on the
SAME train split, then we re-run the degradation curve with URLs masked
everywhere. Comparing "original" vs "url-masked" shows how much of each detector's
apparent robustness was propped up by the frozen links.

We also report detection rate at a FIXED false-positive rate (1% on test ham),
which is the operationally meaningful threshold — far more so than the arbitrary
0.5 used for the headline F1/recall.

This module is pure compute (no API calls). It caches URL-blind models so re-runs
are fast. `score_condition` is reused by src.mitigate for the adversarial retrain.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from . import config, features
from .detectors import _fit, model_factories
from .evaluate import (
    CURVE_SEVERITIES,
    _orig_text_map,
    index_rewrites,
    intersection_ids,
    load_rewrites,
)
from .metrics import compute_metrics

# Operating point for the fixed-FPR analysis (fraction of ham wrongly flagged).
TARGET_FPR: float = 0.01

# URL-blind model + transformer cache paths.
MASKED_VEC = config.MODELS_DIR / "tfidf_vectorizer_urlmasked.joblib"
MASKED_SCALER = config.MODELS_DIR / "handcrafted_scaler_urlmasked.joblib"


def _masked_model_path(name: str):
    return config.MODELS_DIR / f"{name}_urlmasked.joblib"


# --------------------------------------------------------------------------- #
# URL masking
# --------------------------------------------------------------------------- #
def mask_urls(text: str) -> str:
    """Remove every URL (same regex the handcrafted features use). Replaced with a
    space so token boundaries are preserved but no scheme/domain/path survives."""
    return features.URL_RE.sub(" ", text or "")


# --------------------------------------------------------------------------- #
# URL-blind detector training (cached)
# --------------------------------------------------------------------------- #
def _combine(tfidf: sparse.spmatrix, hc_scaled: np.ndarray) -> sparse.csr_matrix:
    return sparse.hstack([tfidf, sparse.csr_matrix(hc_scaled)]).tocsr()


def build_masked_models(force: bool = False) -> None:
    """Fit a URL-blind TF-IDF + scaler on masked train text and retrain all three
    detectors on it. Cached; re-fits only when missing or force=True."""
    paths = [MASKED_VEC, MASKED_SCALER, *(_masked_model_path(n) for n in model_factories())]
    if not force and all(p.exists() for p in paths):
        return

    print("  [url-ablation] fitting URL-blind features + retraining detectors...")
    df = features.load_dataset()
    train = df[df["split"] == "train"].reset_index(drop=True)
    train_masked = train["text"].map(mask_urls)

    vec = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    tfidf_tr = vec.fit_transform(train_masked)
    scaler = StandardScaler(with_mean=False)
    hc_tr = scaler.fit_transform(features.handcrafted_matrix(train.assign(text=train_masked)))
    x_tr = _combine(tfidf_tr, hc_tr)
    y_tr = train["label"].to_numpy()

    joblib.dump(vec, MASKED_VEC)
    joblib.dump(scaler, MASKED_SCALER)
    for name, factory in model_factories().items():
        clf = _fit(name, factory(), x_tr, y_tr)
        joblib.dump(clf, _masked_model_path(name))
    print(f"  [url-ablation] URL-blind train matrix {x_tr.shape}; models cached")


def masked_transform(texts: list[str], had_html) -> sparse.csr_matrix:
    """Featurize arbitrary texts with the URL-blind transformers (masks first)."""
    vec = joblib.load(MASKED_VEC)
    scaler = joblib.load(MASKED_SCALER)
    masked = [mask_urls(t) for t in texts]
    frame = pd.DataFrame({"text": masked, "had_html": list(had_html)})
    tfidf = vec.transform(frame["text"])
    hc = scaler.transform(features.handcrafted_matrix(frame))
    return _combine(tfidf, hc)


# --------------------------------------------------------------------------- #
# Generic scorer reused by both the ablation and the mitigation rescoring
# --------------------------------------------------------------------------- #
def _phish_texts(oids, by_id, severity, orig_text):
    texts, html = [], []
    for oid in oids:
        if severity == 0.0:
            texts.append(orig_text[oid]["text"])
            html.append(orig_text[oid]["had_html"])
        else:
            rec = by_id[oid][severity]
            texts.append(rec["rewrite_text"])
            html.append(bool(rec.get("had_html", False)))
    return texts, html


def score_condition(
    models: dict[str, object],
    transform,
    inter_ids: list[str],
    by_id: dict,
    orig_text: dict,
    *,
    target_fpr: float = TARGET_FPR,
) -> pd.DataFrame:
    """Score one (models, featurizer) condition across the curve on a FIXED
    phishing set. Returns rows with recall@0.5, recall@target-FPR, pr_auc, roc_auc.

    Ham (negative class) is the full test ham and its text never changes, so the
    fixed-FPR threshold per model is computed once from clean ham scores.
    """
    df = features.load_dataset()
    test = df[df["split"] == "test"].reset_index(drop=True)
    ham = test[test["label"] == config.LABEL_HAM]
    ham_X = transform(ham["text"].tolist(), ham["had_html"].to_numpy())
    n_ham = ham_X.shape[0]

    # Per-model fixed-FPR threshold from ham scores (severity-invariant).
    ham_scores = {name: clf.predict_proba(ham_X)[:, 1] for name, clf in models.items()}
    thr = {name: float(np.quantile(s, 1.0 - target_fpr)) for name, s in ham_scores.items()}

    rows: list[dict] = []
    for severity in CURVE_SEVERITIES:
        texts, html = _phish_texts(inter_ids, by_id, severity, orig_text)
        phish_X = transform(texts, html)
        X = sparse.vstack([ham_X, phish_X]).tocsr()
        y = np.concatenate([np.zeros(n_ham, int), np.ones(len(inter_ids), int)])
        for name, clf in models.items():
            score = clf.predict_proba(X)[:, 1]
            phish_score = score[n_ham:]
            m = compute_metrics(y, score)
            rows.append(
                {
                    "model": name,
                    "severity": severity,
                    "n_phish": len(inter_ids),
                    "recall_05": m["recall"],
                    "recall_fpr": float((phish_score >= thr[name]).mean()),
                    "pr_auc": m["pr_auc"],
                    "roc_auc": m["roc_auc"],
                    "fpr_threshold": thr[name],
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_ablation() -> None:
    config.ensure_dirs()
    features.ensure_features()
    build_masked_models()

    recs = load_rewrites()
    by_id = index_rewrites(recs)
    orig_text = _orig_text_map()
    inter = intersection_ids(by_id)
    print(
        f"=== URL ablation + fixed-FPR ({int(TARGET_FPR*100)}% FPR) on intersection "
        f"set (n_phish={len(inter)}, ham fixed) ==="
    )

    orig_models = {n: joblib.load(config.MODELS_DIR / f"{n}.joblib") for n in model_factories()}
    masked_models = {n: joblib.load(_masked_model_path(n)) for n in model_factories()}

    df_orig = score_condition(orig_models, features.transform_texts, inter, by_id, orig_text)
    df_orig["condition"] = "original"
    df_mask = score_condition(masked_models, masked_transform, inter, by_id, orig_text)
    df_mask["condition"] = "url_masked"

    out = pd.concat([df_orig, df_mask], ignore_index=True)
    path = config.TABLES_DIR / "url_ablation_degradation.csv"
    out.to_csv(path, index=False)
    _print_side_by_side(df_orig, df_mask)
    print(f"\n  wrote {path.name}")


def _print_side_by_side(df_orig: pd.DataFrame, df_mask: pd.DataFrame) -> None:
    for name in sorted(df_orig["model"].unique()):
        o = df_orig[df_orig["model"] == name].set_index("severity")
        m = df_mask[df_mask["model"] == name].set_index("severity")
        print(f"\n  {name}   (recall@0.5  |  detection@{int(TARGET_FPR*100)}%FPR  |  PR-AUC)")
        print(
            f"    {'sev':>4}   {'orig':>6} {'masked':>7}    "
            f"{'orig':>6} {'masked':>7}    {'orig':>6} {'masked':>7}"
        )
        for s in CURVE_SEVERITIES:
            print(
                f"    {s:>4.2f}   "
                f"{o.loc[s,'recall_05']:>6.3f} {m.loc[s,'recall_05']:>7.3f}    "
                f"{o.loc[s,'recall_fpr']:>6.3f} {m.loc[s,'recall_fpr']:>7.3f}    "
                f"{o.loc[s,'pr_auc']:>6.3f} {m.loc[s,'pr_auc']:>7.3f}"
            )
        do05 = o.loc[1.0, "recall_05"] - o.loc[0.0, "recall_05"]
        dm05 = m.loc[1.0, "recall_05"] - m.loc[0.0, "recall_05"]
        dofp = o.loc[1.0, "recall_fpr"] - o.loc[0.0, "recall_fpr"]
        dmfp = m.loc[1.0, "recall_fpr"] - m.loc[0.0, "recall_fpr"]
        print(
            f"    drop 0->1.0  recall@0.5: orig {do05:+.3f} / masked {dm05:+.3f}   "
            f"det@FPR: orig {dofp:+.3f} / masked {dmfp:+.3f}"
        )


if __name__ == "__main__":
    run_ablation()
