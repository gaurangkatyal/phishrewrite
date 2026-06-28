"""Reverse-direction adversarial-training transfer (Gemini -> Haiku).

The forward mitigation (src.mitigate) trains on HAIKU rewrites and the cross-
generator check (mitigation_cross_gemini.csv) scores those Haiku-trained detectors
on Gemini rewrites. This module closes the loop in the other direction: TRAIN on
GEMINI rewrites, then TEST on the canonical HAIKU test rewrites. If recovery holds
both ways, adversarial training is generator-agnostic rather than keyed to the
specific rewriter it saw.

Pipeline (mirrors src.mitigate exactly, only the rewriter and the cache/model
paths differ so nothing collides with the Haiku-direction artifacts):
  1. Sample the SAME 500 seeded TRAIN phishing emails (src.mitigate.sample_phish_train).
  2. Rewrite each at severities {0.5, 1.0} with Gemini = 1,000 calls. Gemini has no
     Batch API here, so this runs synchronous; pace with ATTACK_RPM on paid tier.
     Cached to data/processed/rewrites_train_<gemini-slug>.jsonl (separate file).
  3. Augment train with the passing (URL-retained, non-refused) Gemini rewrites.
  4. Retrain all three detectors, URL-intact and URL-blind, into *_advgem paths.
  5. Rescore vs the EXISTING Haiku test rewrites (strict intersection) — baseline
     vs Gemini-adv-trained, both URL conditions — and apply the Task-C McNemar
     exact paired test to the sev0->sev1 detection drop of every cell.

    python -m src.mitigate_reverse --estimate          # projected calls + cost
    ATTACK_PROVIDER=gemini python -m src.mitigate_reverse --run --yes   # 1,000 calls
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from . import ablation, attack, config, features, mitigate
from .detectors import _fit, model_factories
from .metrics import compute_metrics


def _gem_slug() -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", config.model_for()).strip("-")


def _gem_cache_path():
    return config.PROCESSED_DIR / f"rewrites_train_{_gem_slug()}.jsonl"


def _advgem_vec(masked: bool):
    s = "_urlmasked" if masked else ""
    return (
        config.MODELS_DIR / f"tfidf_vectorizer_advgem{s}.joblib",
        config.MODELS_DIR / f"handcrafted_scaler_advgem{s}.joblib",
    )


def _advgem_model_path(name: str, masked: bool):
    return config.MODELS_DIR / f"{name}_advgem{'_urlmasked' if masked else ''}.joblib"


# Rewrite (Gemini), separate cache (API)
def _load_cache() -> dict:
    path, cache = _gem_cache_path(), {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                cache[attack._cache_key(rec["original_id"], float(rec["severity"]))] = rec
    return cache


def _append(rec: dict) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with _gem_cache_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def estimate() -> None:
    # Reuse the project estimator (resolves Gemini pricing via model_for()).
    mitigate.estimate()


def rewrite_train_gem(sample: pd.DataFrame | None = None) -> list[dict]:
    config.require_api_key(config.ATTACK_PROVIDER)
    if config.ATTACK_PROVIDER != "gemini":
        raise SystemExit("Reverse mitigation must run with ATTACK_PROVIDER=gemini.")
    sample = mitigate.sample_phish_train() if sample is None else sample
    system = attack.load_system_prompt()
    all_sevs = attack.load_severity_prompts()
    sevs = {s: all_sevs[s] for s in mitigate.MITIGATE_SEVERITIES}
    rewriter = attack.Rewriter(
        config.ATTACK_PROVIDER,
        config.model_for(),
        config.ATTACK_TEMPERATURE,
        config.ATTACK_MAX_TOKENS,
    )
    header = (
        f"=== Gemini TRAIN rewrite: {len(sample)} x {len(sevs)} "
        f"= {len(sample) * len(sevs)} calls ==="
    )
    return attack.run_rewrite_loop(
        sample,
        sevs,
        rewriter=rewriter,
        system=system,
        cache=_load_cache(),
        append_fn=_append,
        header=header,
        label="Gemini train rewrite",
        retention_summary=True,
    )


# Augment + retrain into *_advgem paths (no API)
def _build_and_fit(text: pd.Series, had_html: pd.Series, y: np.ndarray, masked: bool) -> None:
    src_text = text.map(ablation.mask_urls) if masked else text
    vec = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    tfidf = vec.fit_transform(src_text)
    scaler = StandardScaler(with_mean=False)
    frame = pd.DataFrame({"text": src_text.to_numpy(), "had_html": had_html.to_numpy()})
    hc = scaler.fit_transform(features.handcrafted_matrix(frame))
    x = sparse.hstack([tfidf, sparse.csr_matrix(hc)]).tocsr()
    vp, sp = _advgem_vec(masked)
    joblib.dump(vec, vp)
    joblib.dump(scaler, sp)
    for name, factory in model_factories().items():
        joblib.dump(_fit(name, factory(), x, y), _advgem_model_path(name, masked))
    print(f"  [{'url_masked' if masked else 'original'}] augmented {x.shape}; 3 models cached")


def augment_and_retrain(records: list[dict]) -> int:
    df = features.load_dataset()
    train = df[df["split"] == "train"].reset_index(drop=True)
    passing = mitigate._passing(records)
    aug = pd.DataFrame(
        {
            "text": [r["rewrite_text"] for r in passing],
            "had_html": [bool(r.get("had_html", False)) for r in passing],
            "label": [config.LABEL_PHISHING] * len(passing),
        }
    )
    full = pd.concat([train[["text", "had_html", "label"]], aug], ignore_index=True)
    y = full["label"].to_numpy()
    print("=== augment + retrain (Gemini-direction) ===")
    print(f"  train {len(train)} + passing Gemini rewrites {len(aug)} -> {len(full)}")
    _build_and_fit(full["text"], full["had_html"], y, masked=False)
    _build_and_fit(full["text"], full["had_html"], y, masked=True)
    return len(aug)


# Rescore vs Haiku test rewrites, with McNemar (no API)
def _advgem_transform(masked: bool):
    vp, sp = _advgem_vec(masked)
    vec, scaler = joblib.load(vp), joblib.load(sp)

    def tf(texts, had_html):
        src = [ablation.mask_urls(t) for t in texts] if masked else list(texts)
        frame = pd.DataFrame({"text": src, "had_html": list(had_html)})
        tfidf = vec.transform(frame["text"])
        hc = scaler.transform(features.handcrafted_matrix(frame))
        return sparse.hstack([tfidf, sparse.csr_matrix(hc)]).tocsr()

    return tf


def _score_cell(models, transform, inter, by_id, orig_text):
    """Return (rows_df, det_dict) for one (models, featurizer) cell across the curve.
    det_dict[model][severity] -> per-email detection vector @0.5 (for McNemar)."""
    df = features.load_dataset()
    test = df[df["split"] == "test"].reset_index(drop=True)
    ham = test[test["label"] == config.LABEL_HAM]
    ham_X = transform(ham["text"].tolist(), ham["had_html"].to_numpy())
    n_ham = ham_X.shape[0]
    ham_scores = {n: clf.predict_proba(ham_X)[:, 1] for n, clf in models.items()}
    thr = {n: float(np.quantile(s, 1 - ablation.TARGET_FPR)) for n, s in ham_scores.items()}

    det = {n: {} for n in models}
    rows = []
    for sev in ablation.CURVE_SEVERITIES:
        texts, html = ablation._phish_texts(inter, by_id, sev, orig_text)
        phish_X = transform(texts, html)
        X = sparse.vstack([ham_X, phish_X]).tocsr()
        y = np.concatenate([np.zeros(n_ham, int), np.ones(len(inter), int)])
        for name, clf in models.items():
            s = clf.predict_proba(X)[:, 1]
            pscore = s[n_ham:]
            det[name][sev] = (pscore >= 0.5).astype(int)
            m = compute_metrics(y, s)
            rows.append(
                {
                    "model": name,
                    "severity": sev,
                    "n_phish": len(inter),
                    "recall_05": m["recall"],
                    "recall_fpr": float((pscore >= thr[name]).mean()),
                    "pr_auc": m["pr_auc"],
                    "roc_auc": m["roc_auc"],
                    "fpr_threshold": thr[name],
                }
            )
    return pd.DataFrame(rows), det


def rescore_vs_haiku() -> None:
    test_recs = ablation.load_rewrites()  # canonical Haiku test rewrites
    by_id = ablation.index_rewrites(test_recs)
    orig_text = ablation._orig_text_map()
    inter = ablation.intersection_ids(by_id)
    print(f"=== rescore Gemini-adv vs Haiku test (strict intersection n={len(inter)}) ===")

    base = {n: joblib.load(config.MODELS_DIR / f"{n}.joblib") for n in model_factories()}
    base_m = {n: joblib.load(ablation._masked_model_path(n)) for n in model_factories()}
    adv = {n: joblib.load(_advgem_model_path(n, False)) for n in model_factories()}
    adv_m = {n: joblib.load(_advgem_model_path(n, True)) for n in model_factories()}

    cells = [
        ("original", "baseline", base, features.transform_texts),
        ("original", "advgem_trained", adv, _advgem_transform(False)),
        ("url_masked", "baseline", base_m, ablation.masked_transform),
        ("url_masked", "advgem_trained", adv_m, _advgem_transform(True)),
    ]
    frames, mcn = [], []
    for url_cond, train_cond, models, tf in cells:
        d, det = _score_cell(models, tf, inter, by_id, orig_text)
        d["url_condition"] = url_cond
        d["training"] = train_cond
        frames.append(d)
        for name, sevmap in det.items():
            d0, d1 = sevmap[0.0], sevmap[1.0]
            b = int(((d0 == 1) & (d1 == 0)).sum())
            c = int(((d0 == 0) & (d1 == 1)).sum())
            nd = b + c
            p = 1.0 if nd == 0 else float(binomtest(min(b, c), nd, 0.5, "two-sided").pvalue)
            mcn.append(
                {
                    "url_condition": url_cond,
                    "training": train_cond,
                    "model": name,
                    "n": len(d0),
                    "det_sev0": float(d0.mean()),
                    "det_sev1": float(d1.mean()),
                    "effect_drop": float(d0.mean() - d1.mean()),
                    "mcnemar_b_lost": b,
                    "mcnemar_c_gained": c,
                    "mcnemar_exact_p": p,
                }
            )

    out = pd.concat(frames, ignore_index=True)
    path = config.TABLES_DIR / "mitigation_cross_haiku.csv"
    out.to_csv(path, index=False)
    spath = config.TABLES_DIR / "mitigation_cross_haiku_significance.csv"
    pd.DataFrame(mcn).to_csv(spath, index=False)
    _print(out)
    print(f"\n  wrote {path.name} and {spath.name}")


def _print(out: pd.DataFrame) -> None:
    fpr = int(ablation.TARGET_FPR * 100)
    for uc in ["original", "url_masked"]:
        print(f"\n  === {uc} (detection@{fpr}%FPR: baseline -> advgem) ===")
        for name in sorted(out["model"].unique()):
            b = out[
                (out.url_condition == uc) & (out.training == "baseline") & (out.model == name)
            ].set_index("severity")
            a = out[
                (out.url_condition == uc) & (out.training == "advgem_trained") & (out.model == name)
            ].set_index("severity")
            print(
                f"    {name}: "
                + "  ".join(
                    f"s{s:.2f} {b.loc[s,'recall_fpr']:.3f}->{a.loc[s,'recall_fpr']:.3f}"
                    for s in ablation.CURVE_SEVERITIES
                )
            )


def run_full() -> None:
    config.require_api_key(config.ATTACK_PROVIDER)
    records = rewrite_train_gem()
    augment_and_retrain(records)
    rescore_vs_haiku()


def main(argv=None):
    p = argparse.ArgumentParser(description="Gemini->Haiku mitigation transfer")
    p.add_argument("--estimate", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument(
        "--rescore", action="store_true", help="retrain+rescore from cached rewrites (no API)"
    )
    a = p.parse_args(argv)
    config.ensure_dirs()
    if a.estimate:
        estimate()
        return
    if a.rescore:
        recs = list(_load_cache().values())
        if not recs:
            raise SystemExit("No Gemini train rewrites cached; run --run first.")
        augment_and_retrain(recs)
        rescore_vs_haiku()
        return
    if a.run:
        if not a.yes:
            print("Refusing to spend without --yes. See --estimate first.")
            sys.exit(2)
        run_full()
        return
    p.print_help()


if __name__ == "__main__":
    main()
