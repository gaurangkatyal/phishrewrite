"""Evaluation: clean baseline and degradation curve.

--clean        score persisted detectors on the clean test set (the severity=0.0
               anchor of the degradation curve) -> results/tables/test_clean.csv
--degradation  recompute detector metrics on the LLM-rewritten phishing
               per severity. The headline curve is computed on the retention
               INTERSECTION set (emails whose URLs were retained AND that were
               not refused at ALL four severities) so the comparison is on a
               fixed set of emails. A per-severity full-PASS sensitivity check is
               also reported. Both include bootstrap 95% CIs and a source/era
               ablation. Also exports spot-check pairs (PASS + FAIL) for manual
               verification and writes results/figures/degradation_curve.png.

Eval-set construction (shared by every curve): the negative class is the FULL
test-set ham (fixed across all severities); the positive class is the phishing
emails in the set under study, with their text replaced by the severity-s rewrite
(severity 0.0 uses the original phishing text). Only the phishing text varies with
severity, so any metric change is attributable to the rewrite alone. `recall` at
threshold 0.5 is exactly the phishing detection rate (fraction of phish flagged).
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from . import config, features
from .detectors import model_factories
from .metrics import METRIC_NAMES, compute_metrics

# All severities plotted on the curve, including the 0.0 clean anchor.
CURVE_SEVERITIES: tuple[float, ...] = config.SEVERITIES  # (0.0, 0.25, 0.5, 0.75, 1.0)


# --------------------------------------------------------------------------- #
# Clean baseline
# --------------------------------------------------------------------------- #
def evaluate_clean() -> pd.DataFrame:
    _, _, x_test, y_test = features.load_features()
    rows: list[dict] = []
    for name in model_factories():
        path = config.MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run `python -m src.detectors` first")
        clf = joblib.load(path)
        proba = clf.predict_proba(x_test)[:, 1]
        rows.append({"model": name, **compute_metrics(y_test, proba)})

    df = pd.DataFrame(rows)[["model", *METRIC_NAMES]]
    out = config.TABLES_DIR / "test_clean.csv"
    df.to_csv(out, index=False)
    print("Clean test-set metrics (severity 0.0 anchor):")
    print(df.to_string(index=False))
    print(f"\nWrote {out}")
    return df


# --------------------------------------------------------------------------- #
# Rewrite-evaluation helpers
# --------------------------------------------------------------------------- #
def load_rewrites(path=None) -> list[dict]:
    path = path or config.REWRITES_JSONL
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m src.attack --run` first")
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _is_valid(rec: dict) -> bool:
    """A rewrite is label-valid if the model did not refuse AND every original URL
    was retained (retained_urls is True). No-URL emails (retained_urls is None)
    cannot have URL retention auto-verified, so they are conservatively treated as
    NOT valid for the headline set."""
    return (rec.get("retained_urls") is True) and (not rec.get("refused"))


def index_rewrites(recs: list[dict]) -> dict[str, dict[float, dict]]:
    """original_id -> {severity -> record}."""
    by_id: dict[str, dict[float, dict]] = {}
    for r in recs:
        by_id.setdefault(r["original_id"], {})[float(r["severity"])] = r
    return by_id


def intersection_ids(by_id: dict[str, dict[float, dict]]) -> list[str]:
    """original_ids that are label-valid at EVERY non-zero severity."""
    out = []
    for oid, per_sev in by_id.items():
        if all(s in per_sev and _is_valid(per_sev[s]) for s in config.ATTACK_SEVERITIES):
            out.append(oid)
    return sorted(out)


def per_severity_pass_ids(by_id: dict[str, dict[float, dict]]) -> dict[float, list[str]]:
    """For each severity, the original_ids that are label-valid at THAT severity."""
    out: dict[float, list[str]] = {}
    for s in config.ATTACK_SEVERITIES:
        out[s] = sorted(
            oid for oid, per_sev in by_id.items() if s in per_sev and _is_valid(per_sev[s])
        )
    return out


def _era_of(original_id: str) -> str:
    """Bucket: legacy Nazario mbox corpus vs. modern year-tagged phishing."""
    parts = original_id.split(":")
    tag = parts[1] if len(parts) > 1 else original_id
    return "legacy_mbox" if tag.endswith(".mbox") else "modern_year"


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #
def _ham_features() -> sparse.csr_matrix:
    """Featurize the full test-set ham once (the fixed negative class)."""
    df = features.load_dataset()
    test = df[df["split"] == "test"].reset_index(drop=True)
    ham = test[test["label"] == config.LABEL_HAM]
    return features.transform_texts(ham["text"].tolist(), ham["had_html"].to_numpy())


def _phish_features_for(
    oids: list[str],
    by_id: dict[str, dict[float, dict]],
    severity: float,
    orig_text: dict[str, dict],
) -> sparse.csr_matrix:
    """Featurize the phishing emails in `oids` at the given severity.
    severity 0.0 -> original processed text; >0 -> the severity-s rewrite text."""
    texts: list[str] = []
    had_html: list[bool] = []
    for oid in oids:
        if severity == 0.0:
            row = orig_text[oid]
            texts.append(row["text"])
            had_html.append(bool(row["had_html"]))
        else:
            rec = by_id[oid][severity]
            texts.append(rec["rewrite_text"])
            had_html.append(bool(rec.get("had_html", False)))
    return features.transform_texts(texts, had_html)


def _orig_text_map() -> dict[str, dict]:
    """original_id -> {text, had_html} from the processed dataset (clean anchor)."""
    df = features.load_dataset()
    sub = df[df["split"] == "test"]
    return {
        r["original_id"]: {"text": r["text"], "had_html": bool(r["had_html"])}
        for _, r in sub.iterrows()
    }


# --------------------------------------------------------------------------- #
# Metrics + bootstrap
# --------------------------------------------------------------------------- #
def _bootstrap_cis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int,
    ci: float,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs for each metric, resampling rows with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:  # degenerate resample; skip
            continue
        for k, v in compute_metrics(yt, ys).items():
            acc[k].append(v)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return {
        m: (float(np.quantile(vals, lo_q)), float(np.quantile(vals, hi_q)))
        for m, vals in acc.items()
        if vals
    }


def _score_curve(
    label: str,
    models: dict[str, object],
    ham_X: sparse.csr_matrix,
    ids_by_sev: dict[float, list[str]],
    by_id: dict[str, dict[float, dict]],
    orig_text: dict[str, dict],
) -> list[dict]:
    """Compute every metric (+ bootstrap CI) for each model x severity.

    ids_by_sev maps each severity (incl. 0.0) to the phishing original_ids to
    score at that severity. For the fixed intersection curve this is the same id
    list at every severity; for the sensitivity curve it varies per severity.
    """
    n_ham = ham_X.shape[0]
    rows: list[dict] = []
    for severity in CURVE_SEVERITIES:
        oids = ids_by_sev[severity]
        phish_X = _phish_features_for(oids, by_id, severity, orig_text)
        X = sparse.vstack([ham_X, phish_X]).tocsr()
        y = np.concatenate([np.zeros(n_ham, dtype=int), np.ones(len(oids), dtype=int)])
        for name, clf in models.items():
            score = clf.predict_proba(X)[:, 1]
            m = compute_metrics(y, score)
            cis = _bootstrap_cis(y, score, config.N_BOOTSTRAP, config.BOOTSTRAP_CI, config.SEED)
            row = {"set": label, "model": name, "severity": severity, "n_phish": len(oids)}
            for k in METRIC_NAMES:
                row[k] = m[k]
                lo, hi = cis.get(k, (float("nan"), float("nan")))
                row[f"{k}_lo"] = lo
                row[f"{k}_hi"] = hi
            rows.append(row)
    return rows


def _era_ablation(
    models: dict[str, object],
    ham_X: sparse.csr_matrix,
    inter_ids: list[str],
    by_id: dict[str, dict[float, dict]],
    orig_text: dict[str, dict],
) -> list[dict]:
    """Detection rate (recall) per era bucket x model x severity on the
    intersection set — confirms the degradation isn't driven by a single era."""
    buckets: dict[str, list[str]] = {}
    for oid in inter_ids:
        buckets.setdefault(_era_of(oid), []).append(oid)
    rows: list[dict] = []
    for era, oids in sorted(buckets.items()):
        for severity in CURVE_SEVERITIES:
            phish_X = _phish_features_for(oids, by_id, severity, orig_text)
            for name, clf in models.items():
                score = clf.predict_proba(phish_X)[:, 1]
                det = float((score >= 0.5).mean())  # recall on phishing-only
                rows.append(
                    {
                        "era": era,
                        "model": name,
                        "severity": severity,
                        "n_phish": len(oids),
                        "detection_rate": det,
                    }
                )
    return rows


# --------------------------------------------------------------------------- #
# Spot-check export
# --------------------------------------------------------------------------- #
def _export_spotcheck(
    recs: list[dict], orig_text: dict[str, dict], n_pass: int = 30, n_fail: int = 10
) -> int:
    rng = np.random.default_rng(config.SEED)
    passes = [r for r in recs if _is_valid(r)]
    fails = [r for r in recs if (r.get("retained_urls") is False) and not r.get("refused")]
    pick_pass = list(rng.choice(len(passes), size=min(n_pass, len(passes)), replace=False))
    pick_fail = list(rng.choice(len(fails), size=min(n_fail, len(fails)), replace=False))

    def to_row(r: dict, verdict: str) -> dict:
        ot = orig_text.get(r["original_id"], {})
        return {
            "verdict": verdict,
            "original_id": r["original_id"],
            "severity": r["severity"],
            "n_orig_urls": r.get("n_orig_urls"),
            "missing_urls": "; ".join(r.get("missing_urls") or []),
            "refused": r.get("refused"),
            "original_text": (ot.get("text") or "")[:4000],
            "rewrite_text": (r.get("rewrite_text") or "")[:4000],
        }

    rows = [to_row(passes[i], "PASS") for i in pick_pass]
    rows += [to_row(fails[i], "FAIL") for i in pick_fail]
    out = config.TABLES_DIR / "spotcheck_pairs.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  spot-check: {len(pick_pass)} PASS + {len(pick_fail)} FAIL pairs -> {out.name}")
    return len(rows)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _plot_curve(headline: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_to_plot = [("recall", "Phishing detection rate (recall @0.5)"), ("pr_auc", "PR-AUC")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    models = sorted(headline["model"].unique())
    colors = dict(zip(models, ["#1f77b4", "#d62728", "#2ca02c"]))

    for ax, (metric, title) in zip(axes, metrics_to_plot):
        for name in models:
            sub = headline[headline["model"] == name].sort_values("severity")
            x = sub["severity"].to_numpy()
            y = sub[metric].to_numpy()
            lo = sub[f"{metric}_lo"].to_numpy()
            hi = sub[f"{metric}_hi"].to_numpy()
            ax.plot(x, y, marker="o", label=name, color=colors.get(name))
            ax.fill_between(x, lo, hi, alpha=0.15, color=colors.get(name))
        ax.set_title(title)
        ax.set_xlabel("Rewrite severity")
        ax.set_ylabel(metric)
        ax.set_xticks(list(CURVE_SEVERITIES))
        ax.grid(True, alpha=0.3)
        ax.legend()

    n_phish = int(headline["n_phish"].iloc[0])
    fig.suptitle(
        f"Detector degradation under LLM rewrite — intersection set "
        f"(n_phish={n_phish}, ham fixed; 95% bootstrap CI)"
    )
    fig.tight_layout()
    out = config.FIGURES_DIR / "degradation_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  figure -> {out}")


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def evaluate_degradation() -> None:
    config.ensure_dirs()
    recs = load_rewrites()
    by_id = index_rewrites(recs)
    orig_text = _orig_text_map()

    inter = intersection_ids(by_id)
    per_sev = per_severity_pass_ids(by_id)

    print("=== retention sets ===")
    print(f"  total sampled phishing emails : {len(by_id)}")
    print("  per-severity full-PASS (valid @ that severity):")
    for s in config.ATTACK_SEVERITIES:
        print(f"      severity {s:.2f}: {len(per_sev[s])}")
    print(
        f"  HEADLINE intersection (valid @ ALL {len(config.ATTACK_SEVERITIES)} "
        f"severities): {len(inter)}"
    )
    if not inter:
        raise SystemExit("Empty intersection set — cannot compute headline curve.")

    models = {name: joblib.load(config.MODELS_DIR / f"{name}.joblib") for name in model_factories()}
    ham_X = _ham_features()
    print(f"  negative class (fixed test ham): {ham_X.shape[0]}")

    # 1) Headline: fixed intersection set, same ids at every severity.
    print("\n=== headline degradation (intersection set, fixed emails) ===")
    inter_by_sev = {s: inter for s in CURVE_SEVERITIES}
    headline_rows = _score_curve("intersection", models, ham_X, inter_by_sev, by_id, orig_text)
    headline = pd.DataFrame(headline_rows)
    out_h = config.TABLES_DIR / "degradation_intersection.csv"
    headline.to_csv(out_h, index=False)
    _print_curve(headline)
    print(f"  wrote {out_h.name}")

    # 2) Sensitivity: each severity scored on its own full-PASS set (varies).
    print("\n=== sensitivity degradation (per-severity full-PASS set) ===")
    sens_by_sev = {0.0: inter}  # 0.0 anchor uses the intersection for comparability
    sens_by_sev.update({s: per_sev[s] for s in config.ATTACK_SEVERITIES})
    sens_rows = _score_curve("per_severity_pass", models, ham_X, sens_by_sev, by_id, orig_text)
    sens = pd.DataFrame(sens_rows)
    out_s = config.TABLES_DIR / "degradation_per_severity_pass.csv"
    sens.to_csv(out_s, index=False)
    _print_curve(sens)
    print(f"  wrote {out_s.name}")

    # 3) Source/era ablation on the intersection set.
    print("\n=== source/era ablation (detection rate, intersection set) ===")
    abl_rows = _era_ablation(models, ham_X, inter, by_id, orig_text)
    abl = pd.DataFrame(abl_rows)
    out_a = config.TABLES_DIR / "degradation_era_ablation.csv"
    abl.to_csv(out_a, index=False)
    _print_ablation(abl)
    print(f"  wrote {out_a.name}")

    # 4) Spot-check export + figure.
    print("\n=== spot-check export ===")
    _export_spotcheck(recs, orig_text)
    print("\n=== figure ===")
    _plot_curve(headline)


def evaluate_cross_model(rewrites_path, out_name: str, label: str) -> pd.DataFrame:
    """Score the BASELINE detectors on a second rewriting model's rewrites, on
    that model's own retention-intersection set. Same machinery as the headline
    curve, so the result is directly comparable to degradation_intersection.csv.
    Non-API compute only."""
    config.ensure_dirs()
    recs = load_rewrites(rewrites_path)
    by_id = index_rewrites(recs)
    orig_text = _orig_text_map()
    inter = intersection_ids(by_id)
    print(f"=== cross-model degradation: {label} ===")
    print(f"  rewrites: {rewrites_path.name}  | sampled emails: {len(by_id)}")
    for s in config.ATTACK_SEVERITIES:
        n = sum(1 for _oid, d in by_id.items() if s in d and _is_valid(d[s]))
        print(f"      severity {s:.2f} full-PASS: {n}")
    print(f"  intersection (valid @ ALL severities): {len(inter)}")
    if not inter:
        raise SystemExit("Empty intersection set — cannot compute cross-model curve.")
    models = {name: joblib.load(config.MODELS_DIR / f"{name}.joblib") for name in model_factories()}
    ham_X = _ham_features()
    inter_by_sev = {s: inter for s in CURVE_SEVERITIES}
    rows = _score_curve(label, models, ham_X, inter_by_sev, by_id, orig_text)
    df = pd.DataFrame(rows)
    out = config.TABLES_DIR / out_name
    df.to_csv(out, index=False)
    _print_curve(df)
    print(f"  wrote {out}")
    return df


def _print_curve(df: pd.DataFrame) -> None:
    for name in sorted(df["model"].unique()):
        sub = df[df["model"] == name].sort_values("severity")
        print(f"  {name}:")
        for _, r in sub.iterrows():
            print(
                f"    sev {r['severity']:.2f} (n_phish={int(r['n_phish'])}): "
                f"recall={r['recall']:.3f} [{r['recall_lo']:.3f},{r['recall_hi']:.3f}]  "
                f"pr_auc={r['pr_auc']:.3f} [{r['pr_auc_lo']:.3f},{r['pr_auc_hi']:.3f}]  "
                f"roc_auc={r['roc_auc']:.3f}  f1={r['f1']:.3f}"
            )


def _print_ablation(df: pd.DataFrame) -> None:
    for era in sorted(df["era"].unique()):
        sub = df[df["era"] == era]
        n = int(sub["n_phish"].iloc[0])
        print(f"  era={era} (n_phish={n}):")
        for name in sorted(sub["model"].unique()):
            row = sub[sub["model"] == name].sort_values("severity")
            seq = "  ".join(
                f"{s:.2f}:{d:.3f}" for s, d in zip(row["severity"], row["detection_rate"])
            )
            print(f"    {name:18s} detection_rate  {seq}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PhishRewrite evaluation")
    parser.add_argument(
        "--clean", action="store_true", help="score detectors on the clean test set (severity 0.0)"
    )
    parser.add_argument(
        "--degradation",
        action="store_true",
        help="degradation curve on rewritten test set",
    )
    parser.add_argument(
        "--gemini", action="store_true", help="cross-model degradation on the Gemini rewrites"
    )
    args = parser.parse_args(argv)

    config.ensure_dirs()
    if args.degradation:
        evaluate_degradation()
        return
    if args.gemini:
        path = config.PROCESSED_DIR / "rewrites_gemini-2-5-flash.jsonl"
        evaluate_cross_model(path, "degradation_gemini.csv", "gemini_2_5_flash")
        return
    evaluate_clean()


if __name__ == "__main__":
    main()
