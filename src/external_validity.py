"""External-validity replication on the CEAS-2008 corpus.

Why this exists
---------------
The primary study pairs RECENT phishing (Nazario 2015-2024) against OLD ham
(Enron 2001-02, SpamAssassin 2003). A reviewer can object that the detector — and
therefore the rewrite-degradation finding — keys on an *era* artifact (old-ham
formatting vs modern phishing) rather than on malicious-vs-benign content.

CEAS-2008 (Champa et al., Zenodo 8339691, CC-BY 4.0) breaks that confound: BOTH
classes are 2008, so there is no era gap within the corpus to exploit. We replicate
the two headline results off-corpus:
  (1) the LLM-rewrite degradation curve, and
  (2) the URL-masked ablation (the URL-anchoring result — the paper's core).

IMPORTANT label semantics (surfaced for the paper): CEAS-2008's positive class is
generic SPAM, not phishing specifically. So this is a spam-vs-ham external check,
which is weaker than a phishing-specific replication but still tests whether the
"rewrite-to-benign evades a content detector, and the detector is URL-anchored"
mechanism is corpus-independent. This caveat is recorded in RESULTS.md.

Everything else is held identical to the primary pipeline: subject+"\n"+body text,
TEST_SIZE=0.20 stratified seeded split, same TF-IDF + handcrafted features, same 3
detectors, same URL-blind retrain, same fixed-1%-FPR-on-ham operating point, same
1000-seed bootstrap CIs, same Haiku rewrite prompts/cache/retention contract, and
the Task-C McNemar paired test on the sev0->sev1 detection drop.

  python -m src.external_validity --build      # build same-era dataset (no API)
  python -m src.external_validity --train      # fit features + 3 detectors x2 (no API)
  python -m src.external_validity --estimate   # projected calls + cost (no API)
  python -m src.external_validity --rewrite    # 200 pos x 4 sev Haiku rewrites (API)
  python -m src.external_validity --score      # degradation + url-masked -> CSV
  python -m src.external_validity --all        # build, train, rewrite, score
"""

from __future__ import annotations

import argparse
import json
import re

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import ablation, attack, config, features
from .detectors import _fit, model_factories
from .metrics import compute_metrics

CEAS_CSV = config.RAW_DIR / "ceas08" / "CEAS_08.csv"
CEAS_DIR = config.MODELS_DIR / "ceas08"
DATASET_CSV = config.PROCESSED_DIR / "ceas08_dataset.csv"
SAME_ERA_YEAR = 2008
N_REWRITE_SAMPLE = 200
REWRITE_SEVERITIES = (0.25, 0.5, 0.75, 1.0)
CURVE_SEVERITIES = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGET_FPR = ablation.TARGET_FPR  # 0.01
_HTML_RE = re.compile(r"<\s*(html|body|table|div|span|a|br|p|img|font)\b", re.I)


def _rewrites_path():
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", config.model_for()).strip("-")
    return config.PROCESSED_DIR / f"rewrites_ceas08_{slug}.jsonl"


def _model_path(name: str, masked: bool):
    suffix = "_urlmasked" if masked else ""
    return CEAS_DIR / f"{name}{suffix}.joblib"


def _vec_paths(masked: bool):
    suffix = "_urlmasked" if masked else ""
    return (CEAS_DIR / f"tfidf{suffix}.joblib", CEAS_DIR / f"scaler{suffix}.joblib")


# Dataset build (same-era 2008, stratified split)
def build_dataset(force: bool = False) -> pd.DataFrame:
    if DATASET_CSV.exists() and not force:
        return pd.read_csv(DATASET_CSV).assign(
            text=lambda d: d["text"].fillna(""), had_html=lambda d: d["had_html"].astype(bool)
        )

    raw = pd.read_csv(CEAS_CSV)
    raw["subject"] = raw["subject"].fillna("").astype(str)
    raw["body"] = raw["body"].fillna("").astype(str)
    # Same-era filter: keep only emails whose Date header parses to 2008.
    yr = pd.to_datetime(raw["date"], errors="coerce", utc=True).dt.year
    raw = raw[yr == SAME_ERA_YEAR].reset_index(drop=True)

    text = (raw["subject"] + "\n" + raw["body"]).str.strip()
    had_html = raw["body"].str.contains(_HTML_RE)
    df = pd.DataFrame(
        {
            "id": [f"ceas08_{i}" for i in range(len(raw))],
            "original_id": [f"ceas08:{i}" for i in range(len(raw))],
            "source": "ceas08",
            "text": text,
            "had_html": had_html.to_numpy(),
            "label": raw["label"].astype(int).to_numpy(),
        }
    )
    # Drop empties; stratified seeded split mirroring the primary study.
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    tr, te = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.SEED, shuffle=True
    )
    df.loc[tr.index, "split"] = "train"
    df.loc[te.index, "split"] = "test"

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_CSV, index=False)
    n1, n0 = int((df.label == 1).sum()), int((df.label == 0).sum())
    print(f"=== CEAS-2008 same-era dataset built (n={len(df)}) ===")
    print(f"  positive(spam)={n1}  negative(ham)={n0}  spam_frac={n1/len(df):.3f}")
    print(
        f"  split: train={int((df.split=='train').sum())} "
        f"test={int((df.split=='test').sum())} (stratified {config.TEST_SIZE:.0%})"
    )
    print(f"  had_html frac: {df['had_html'].mean():.3f}  -> {DATASET_CSV.name}")
    return df


# Feature build + detector training (original + URL-blind), no API
def _combine(tfidf, hc) -> sparse.csr_matrix:
    return sparse.hstack([tfidf, sparse.csr_matrix(hc)]).tocsr()


def _fit_condition(train: pd.DataFrame, masked: bool) -> None:
    text = train["text"].map(ablation.mask_urls) if masked else train["text"]
    vec = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    tfidf = vec.fit_transform(text)
    scaler = StandardScaler(with_mean=False)
    frame = pd.DataFrame({"text": text.to_numpy(), "had_html": train["had_html"].to_numpy()})
    hc = scaler.fit_transform(features.handcrafted_matrix(frame))
    x = _combine(tfidf, hc)
    y = train["label"].to_numpy()
    vec_path, scaler_path = _vec_paths(masked)
    joblib.dump(vec, vec_path)
    joblib.dump(scaler, scaler_path)
    for name, factory in model_factories().items():
        joblib.dump(_fit(name, factory(), x, y), _model_path(name, masked))
    print(f"  [{'url_masked' if masked else 'original'}] train {x.shape}; 3 models cached")


def train_all(force: bool = False) -> None:
    CEAS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        *_vec_paths(False),
        *_vec_paths(True),
        *(_model_path(n, m) for n in model_factories() for m in (False, True)),
    ]
    if all(p.exists() for p in paths) and not force:
        print("  CEAS detectors already cached.")
        return
    df = build_dataset()
    train = df[df["split"] == "train"].reset_index(drop=True)
    print("=== fit CEAS detectors (original + URL-blind) ===")
    _fit_condition(train, masked=False)
    _fit_condition(train, masked=True)


def _transform(texts, had_html, masked: bool) -> sparse.csr_matrix:
    vec_path, scaler_path = _vec_paths(masked)
    vec, scaler = joblib.load(vec_path), joblib.load(scaler_path)
    src_texts = [ablation.mask_urls(t) for t in texts] if masked else list(texts)
    frame = pd.DataFrame({"text": src_texts, "had_html": list(had_html)})
    tfidf = vec.transform(frame["text"])
    hc = scaler.transform(features.handcrafted_matrix(frame))
    return _combine(tfidf, hc)


# Rewrite the positive class (Haiku), retention contract (API)
def sample_positive(n: int = N_REWRITE_SAMPLE) -> pd.DataFrame:
    df = build_dataset()
    pool = df[(df["split"] == "train") & (df["label"] == config.LABEL_PHISHING)]
    k = min(n, len(pool))
    return pool.sample(n=k, random_state=config.SEED).reset_index(drop=True)


def _load_cache() -> dict:
    path = _rewrites_path()
    cache: dict = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                cache[attack._cache_key(rec["original_id"], float(rec["severity"]))] = rec
    return cache


def _append(rec: dict) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with _rewrites_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def estimate() -> None:
    system = attack.load_system_prompt()
    sevs = attack.load_severity_prompts()
    sevs = {s: sevs[s] for s in REWRITE_SEVERITIES}
    sample = sample_positive()
    n_emails, n_calls = len(sample), len(sample) * len(sevs)
    sys_tok = attack._approx_tokens(system)
    avg_instr = np.mean([attack._approx_tokens(t) for t in sevs.values()])
    avg_email = np.mean([attack._approx_tokens(t) for t in sample["text"].fillna("")])
    avg_out = min(config.ATTACK_MAX_TOKENS, max(64, avg_email))
    model = config.model_for()
    in_price, out_price = config.PRICE_PER_MTOK.get(model, (1.0, 5.0))
    fresh_in = n_calls * (avg_instr + avg_email)
    sys_eff = (
        sys_tok * 1.25 + (n_calls - 1) * sys_tok * 0.1
        if config.USE_PROMPT_CACHE
        else n_calls * sys_tok
    )
    total_in, total_out = fresh_in + sys_eff, n_calls * avg_out
    in_cost, out_cost = total_in / 1e6 * in_price, total_out / 1e6 * out_price
    subtotal = in_cost + out_cost
    use_batch = config.USE_BATCH and config.ATTACK_PROVIDER == "anthropic"
    total = subtotal * (0.5 if use_batch else 1.0)
    print("=== CEAS external-validity rewrite cost estimate (rough upper bound) ===")
    print(f"  provider/model : {config.ATTACK_PROVIDER} / {model}")
    print(f"  positive(spam) sampled : {n_emails} (seeded; CEAS train split)")
    print(f"  severities : {list(sevs)}")
    print(f"  projected calls : {n_emails} x {len(sevs)} = {n_calls}")
    print(f"  est. input  : {total_in/1e6:.3f} M -> ${in_cost:.2f}")
    print(f"  est. output : {total_out/1e6:.3f} M -> ${out_cost:.2f}")
    print(f"  est. subtotal : ${subtotal:.2f}")
    print(f"  ESTIMATED TOTAL : ${total:.2f}  (batch={'on' if use_batch else 'off'})")


def rewrite(sample: pd.DataFrame | None = None) -> list[dict]:
    config.require_api_key(config.ATTACK_PROVIDER)
    sample = sample_positive() if sample is None else sample
    system = attack.load_system_prompt()
    all_sevs = attack.load_severity_prompts()
    sevs = {s: all_sevs[s] for s in REWRITE_SEVERITIES}
    rewriter = attack.Rewriter(
        config.ATTACK_PROVIDER,
        config.model_for(),
        config.ATTACK_TEMPERATURE,
        config.ATTACK_MAX_TOKENS,
    )
    header = (
        f"=== CEAS rewrite: {len(sample)} spam x {len(sevs)} sev "
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
        label="CEAS rewrite",
        retention_summary=True,
    )


# Scoring: degradation + url-masked, fixed-FPR, bootstrap CIs, McNemar
def _is_valid(rec: dict) -> bool:
    return (rec.get("retained_urls") is True) and not rec.get("refused")


def _intersection(by_id: dict) -> list[str]:
    return [
        oid
        for oid, sevmap in by_id.items()
        if all(s in sevmap and _is_valid(sevmap[s]) for s in REWRITE_SEVERITIES)
    ]


def _boot_cis(y, score, thr, seed=config.SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    acc = {k: [] for k in ("recall_05", "recall_fpr", "pr_auc", "roc_auc")}
    from sklearn.metrics import average_precision_score, roc_auc_score

    for _ in range(config.N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yt, ys = y[idx], score[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        pos = ys[yt == 1]
        acc["recall_05"].append(float((pos >= 0.5).mean()))
        acc["recall_fpr"].append(float((pos >= thr).mean()))
        acc["pr_auc"].append(float(average_precision_score(yt, ys)))
        acc["roc_auc"].append(float(roc_auc_score(yt, ys)))
    a = (1 - config.BOOTSTRAP_CI) / 2
    return {k: (float(np.quantile(v, a)), float(np.quantile(v, 1 - a))) for k, v in acc.items()}


def _score_block(masked: bool, inter, by_id, orig_text) -> tuple[pd.DataFrame, dict]:
    """Degradation curve for one URL condition. Returns rows + per-model
    sev0/sev1 detection vectors for the McNemar paired test."""
    condition = "url_masked" if masked else "original"
    df = build_dataset()
    test = df[df["split"] == "test"]
    ham = test[test["label"] == config.LABEL_HAM]
    ham_X = _transform(ham["text"].tolist(), ham["had_html"].to_numpy(), masked)
    models = {n: joblib.load(_model_path(n, masked)) for n in model_factories()}
    ham_scores = {n: clf.predict_proba(ham_X)[:, 1] for n, clf in models.items()}
    thr = {n: float(np.quantile(s, 1 - TARGET_FPR)) for n, s in ham_scores.items()}
    n_ham = ham_X.shape[0]

    det = {n: {} for n in models}  # det[model][severity] -> phish detection vector @0.5
    rows = []
    for sev in CURVE_SEVERITIES:
        texts, html = [], []
        for oid in inter:
            if sev == 0.0:
                texts.append(orig_text[oid]["text"])
                html.append(orig_text[oid]["had_html"])
            else:
                rec = by_id[oid][sev]
                texts.append(rec["rewrite_text"])
                html.append(bool(rec.get("had_html", False)))
        phish_X = _transform(texts, html, masked)
        X = sparse.vstack([ham_X, phish_X]).tocsr()
        y = np.concatenate([np.zeros(n_ham, int), np.ones(len(inter), int)])
        for name, clf in models.items():
            s = clf.predict_proba(X)[:, 1]
            pscore = s[n_ham:]
            det[name][sev] = (pscore >= 0.5).astype(int)
            m = compute_metrics(y, s)
            cis = _boot_cis(y, s, thr[name])
            rows.append(
                {
                    "corpus": "ceas08",
                    "condition": condition,
                    "model": name,
                    "severity": sev,
                    "n_phish": len(inter),
                    "recall_05": m["recall"],
                    "recall_05_lo": cis["recall_05"][0],
                    "recall_05_hi": cis["recall_05"][1],
                    "recall_fpr": float((pscore >= thr[name]).mean()),
                    "recall_fpr_lo": cis["recall_fpr"][0],
                    "recall_fpr_hi": cis["recall_fpr"][1],
                    "pr_auc": m["pr_auc"],
                    "pr_auc_lo": cis["pr_auc"][0],
                    "pr_auc_hi": cis["pr_auc"][1],
                    "roc_auc": m["roc_auc"],
                    "roc_auc_lo": cis["roc_auc"][0],
                    "roc_auc_hi": cis["roc_auc"][1],
                    "fpr_threshold": thr[name],
                }
            )
    return pd.DataFrame(rows), det


def _mcnemar_rows(det: dict, condition: str) -> list[dict]:
    """Task-C paired test on sev0.0 vs sev1.0 detection per model."""
    out = []
    for name, sevmap in det.items():
        d0, d1 = sevmap[0.0], sevmap[1.0]
        b = int(((d0 == 1) & (d1 == 0)).sum())
        c = int(((d0 == 0) & (d1 == 1)).sum())
        nd = b + c
        p = 1.0 if nd == 0 else float(binomtest(min(b, c), nd, 0.5, "two-sided").pvalue)
        out.append(
            {
                "corpus": "ceas08",
                "condition": condition,
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
    return out


def score() -> None:
    train_all()
    recs = list(_load_cache().values())
    if not recs:
        raise SystemExit("No CEAS rewrites cached. Run --rewrite first (needs API key).")
    by_id: dict = {}
    for r in recs:
        by_id.setdefault(r["original_id"], {})[float(r["severity"])] = r
    inter = _intersection(by_id)
    df = build_dataset()
    orig_text = {
        row["original_id"]: {"text": row["text"], "had_html": bool(row["had_html"])}
        for _, row in df.iterrows()
        if row["original_id"] in set(by_id)
    }
    print(f"=== CEAS scoring: intersection valid at all {REWRITE_SEVERITIES} -> n={len(inter)} ===")

    frames, mcn = [], []
    for masked in (False, True):
        d, det = _score_block(masked, inter, by_id, orig_text)
        frames.append(d)
        mcn += _mcnemar_rows(det, "url_masked" if masked else "original")
    out = pd.concat(frames, ignore_index=True)
    path = config.TABLES_DIR / "external_validity.csv"
    out.to_csv(path, index=False)
    mpath = config.TABLES_DIR / "external_validity_significance.csv"
    pd.DataFrame(mcn).to_csv(mpath, index=False)
    _print_summary(out, pd.DataFrame(mcn))
    print(f"\n  wrote {path.name} and {mpath.name}")


def _print_summary(out: pd.DataFrame, mcn: pd.DataFrame) -> None:
    fpr = int(TARGET_FPR * 100)
    for cond in ["original", "url_masked"]:
        print(f"\n  === {cond} (recall@0.5 | det@{fpr}%FPR) ===")
        for name in sorted(out["model"].unique()):
            s = out[(out.condition == cond) & (out.model == name)].set_index("severity")
            print(
                f"    {name}:  "
                + "  ".join(
                    f"s{sev:.2f}={s.loc[sev,'recall_05']:.3f}/{s.loc[sev,'recall_fpr']:.3f}"
                    for sev in CURVE_SEVERITIES
                )
            )
            d05 = s.loc[1.0, "recall_05"] - s.loc[0.0, "recall_05"]
            dfp = s.loc[1.0, "recall_fpr"] - s.loc[0.0, "recall_fpr"]
            print(f"        drop 0->1.0  recall@0.5 {d05:+.3f}  det@FPR {dfp:+.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(description="CEAS-2008 external-validity replication")
    p.add_argument("--build", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--estimate", action="store_true")
    p.add_argument("--rewrite", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--all", action="store_true")
    a = p.parse_args(argv)
    config.ensure_dirs()
    if a.build or a.all:
        build_dataset(force=a.build and not a.all)
    if a.train or a.all:
        train_all()
    if a.estimate:
        estimate()
    if a.rewrite or a.all:
        rewrite()
    if a.score or a.all:
        score()
    if not any([a.build, a.train, a.estimate, a.rewrite, a.score, a.all]):
        p.print_help()


if __name__ == "__main__":
    main()
