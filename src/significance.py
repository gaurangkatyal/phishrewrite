"""Paired significance test on the headline degradation.

Replaces the informal "non-overlapping CIs" reading with a proper paired test
over MATCHED emails. For each model and URL condition we take the strict-270
intersection set and, per email, the detection outcome (score >= 0.5) at
severity 0.0 (clean) vs severity 1.0 (most aggressive rewrite). Because the same
270 emails appear in both conditions, the comparison is paired.

We report, per (model x url_condition):
  - n, detection rate @0.0 and @1.0 (these equal the recall@0.5 cells in the
    headline tables — a built-in cross-check),
  - effect size = det@0.0 - det@1.0 (the per-email mean detection drop),
  - paired bootstrap 95% CI on that drop (1000 seeded resamples of emails),
  - McNemar discordant counts b (det@0 only) / c (det@1 only),
  - paired permutation p-value (sign-flip within each email, 10000 seeded perms),
  - exact McNemar two-sided p (binomial on discordant pairs).

Pure compute, no API. Output: results/tables/significance_paired.csv
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import ablation, config, features
from .detectors import model_factories
from .evaluate import (
    _orig_text_map,
    index_rewrites,
    intersection_ids,
    load_rewrites,
)

N_BOOT = config.N_BOOTSTRAP  # 1000, matches the CI tables
CI = config.BOOTSTRAP_CI  # 0.95
N_PERM = 10000


def _detection_vector(model, transform, oids, by_id, orig_text, severity) -> np.ndarray:
    """Per-email detection indicator (score >= 0.5) for `oids` at `severity`."""
    texts, html = ablation._phish_texts(oids, by_id, severity, orig_text)
    X = transform(texts, html)
    score = model.predict_proba(X)[:, 1]
    return (score >= 0.5).astype(int)


def _paired_stats(d0: np.ndarray, d1: np.ndarray, seed: int) -> dict:
    n = len(d0)
    diff = d0 - d1  # +1: lost at sev1; -1: gained; 0: same
    effect = float(diff.mean())  # = det@0 - det@1 (recall@0.5 drop)

    rng = np.random.default_rng(seed)
    # Paired bootstrap CI: resample emails with replacement.
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot[i] = diff[idx].mean()
    lo = float(np.quantile(boot, (1 - CI) / 2))
    hi = float(np.quantile(boot, 1 - (1 - CI) / 2))

    # McNemar discordant pairs.
    b = int(((d0 == 1) & (d1 == 0)).sum())  # detected clean, missed at sev1
    c = int(((d0 == 0) & (d1 == 1)).sum())  # missed clean, detected at sev1
    # Paired permutation p-value: within each email, randomly swap (d0,d1).
    obs = abs(effect)
    rng2 = np.random.default_rng(seed + 1)
    ge = 0
    for _ in range(N_PERM):
        flip = rng2.integers(0, 2, size=n) * 2 - 1  # +1 / -1 per email
        if abs((diff * flip).mean()) >= obs - 1e-12:
            ge += 1
    perm_p = (ge + 1) / (N_PERM + 1)

    # Exact two-sided McNemar (binomial on discordant pairs, p=0.5).
    from scipy.stats import binomtest

    nd = b + c
    mcnemar_p = (
        1.0 if nd == 0 else float(binomtest(min(b, c), nd, 0.5, alternative="two-sided").pvalue)
    )

    return {
        "n": n,
        "det_sev0": float(d0.mean()),
        "det_sev1": float(d1.mean()),
        "effect_drop": effect,
        "drop_lo": lo,
        "drop_hi": hi,
        "mcnemar_b_lost": b,
        "mcnemar_c_gained": c,
        "perm_p_value": perm_p,
        "mcnemar_exact_p": mcnemar_p,
    }


def run() -> None:
    config.ensure_dirs()
    features.ensure_features()
    ablation.build_masked_models()

    recs = load_rewrites()
    by_id = index_rewrites(recs)
    orig_text = _orig_text_map()
    inter = intersection_ids(by_id)
    print(f"=== paired significance on strict intersection (n={len(inter)}) ===")

    base = {n: joblib.load(config.MODELS_DIR / f"{n}.joblib") for n in model_factories()}
    base_m = {n: joblib.load(ablation._masked_model_path(n)) for n in model_factories()}

    conditions = [
        ("original", base, features.transform_texts),
        ("url_masked", base_m, ablation.masked_transform),
    ]
    rows = []
    for url_cond, models, tf in conditions:
        for name, clf in models.items():
            d0 = _detection_vector(clf, tf, inter, by_id, orig_text, 0.0)
            d1 = _detection_vector(clf, tf, inter, by_id, orig_text, 1.0)
            st = _paired_stats(d0, d1, config.SEED)
            st.update({"model": name, "url_condition": url_cond})
            rows.append(st)
            print(
                f"  [{url_cond:10s}] {name:18s} drop {st['effect_drop']:+.3f} "
                f"[{st['drop_lo']:+.3f},{st['drop_hi']:+.3f}]  "
                f"perm_p={st['perm_p_value']:.4g}  mcnemar_p={st['mcnemar_exact_p']:.4g}  "
                f"(b={st['mcnemar_b_lost']}, c={st['mcnemar_c_gained']})"
            )

    cols = [
        "url_condition",
        "model",
        "n",
        "det_sev0",
        "det_sev1",
        "effect_drop",
        "drop_lo",
        "drop_hi",
        "mcnemar_b_lost",
        "mcnemar_c_gained",
        "perm_p_value",
        "mcnemar_exact_p",
    ]
    out = pd.DataFrame(rows)[cols]
    path = config.TABLES_DIR / "significance_paired.csv"
    out.to_csv(path, index=False)
    print(f"\n  wrote {path.name}")


if __name__ == "__main__":
    run()
