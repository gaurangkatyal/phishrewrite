"""DistilBERT transformer baseline on the CEAS-2008 corpus (no API).

Reviewer question this answers
------------------------------
On the Nazario corpus the primary transformer run (src.transformer_detector)
found DistilBERT resisted LLM rewriting and did not appear URL-anchored — the
opposite of the classical bag-of-words detectors. A skeptic can object that this
is a Nazario-specific artifact (the transformer simply memorised the Nazario
phishing style). CEAS-2008
breaks that confound: it is a *same-era* (both classes 2008), *different* corpus.
If the transformer's robustness + non-URL-anchoring reproduce here, the finding is
corpus-independent rather than a single-dataset quirk.

Held identical to the primary transformer run so numbers are directly comparable:
  * SAME frugal fine-tune config (max_len 256 head-truncation, per-device batch 8
    x grad-accum 2 = eff 16, gradient checkpointing, save_steps 200 + resume,
    num_workers 0, MPS empty_cache + PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0,
    seed 42, class-weighted CE, 3 epochs, lr 2e-5) — reused verbatim from
    transformer_detector.finetune(frame=...).
  * SAME metrics: recall@0.5, det@1%-FPR-on-clean-ham, PR-AUC, ROC-AUC, with
    1000-seed bootstrap CIs and McNemar exact paired test (sev0 vs sev1).
  * SAME url-masked condition = INFERENCE-TIME masking of inputs (ablation.mask_urls),
    matching the primary transformer run (not the classical URL-blind retrain).

CEAS specifics (mirrors src.external_validity):
  * train on the CEAS-2008 train split (stratified 0.20 seeded test holdout),
  * score on the EXISTING cached CEAS Haiku rewrites (rewrites_ceas08_*.jsonl) —
    NO new API calls,
  * intersection = positives valid (URLs retained, not refused) at all of
    {0.25,0.5,0.75,1.0}; curve severities {0,0.25,0.5,0.75,1.0},
  * Gemini CEAS rewrites are scored too *iff* the cache file is present; it is
    absent here, so that block is skipped gracefully.

Label note (carried from external_validity): CEAS-2008's positive class is generic
SPAM, not phishing specifically — this is a spam-vs-ham external check.

  python -m src.transformer_ceas --train   # fine-tune DistilBERT on CEAS train (no API)
  python -m src.transformer_ceas --score   # score cached CEAS rewrites -> CSV (no API)
  python -m src.transformer_ceas --all     # train then score
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from . import ablation, config
from . import external_validity as ev
from . import transformer_detector as td

CEAS_BASE_DIR = config.MODELS_DIR / "distilbert_ceas"
REWRITE_SEVERITIES = ev.REWRITE_SEVERITIES  # (0.25, 0.5, 0.75, 1.0)
CURVE_SEVERITIES = ev.CURVE_SEVERITIES  # (0.0, 0.25, 0.5, 0.75, 1.0)
TARGET_FPR = ev.TARGET_FPR  # 0.01


# --------------------------------------------------------------------------- #
# Training — CEAS train split, frugal config reused from transformer_detector
# --------------------------------------------------------------------------- #
def _ceas_train_frame() -> pd.DataFrame:
    df = ev.build_dataset()
    train = df[df["split"] == "train"][["text", "label"]].reset_index(drop=True)
    train["text"] = train["text"].fillna("")
    n1, n0 = int((train.label == 1).sum()), int((train.label == 0).sum())
    print(f"  CEAS train frame: {len(train)} rows  (spam={n1}, ham={n0})")
    return train


def finetune() -> None:
    td.finetune(CEAS_BASE_DIR, augment=False, frame=_ceas_train_frame())


# --------------------------------------------------------------------------- #
# Rewrite caches: Haiku always; Gemini only if its CEAS cache file exists
# --------------------------------------------------------------------------- #
def _gemini_ceas_path():
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", td.GEMINI_REWRITES.stem.replace("rewrites_", "")).strip(
        "-"
    )
    return config.PROCESSED_DIR / f"rewrites_ceas08_{slug}.jsonl"


def _load_cache_at(path) -> dict:
    """Load a CEAS rewrites JSONL into {cache_key: rec}, mirroring ev._load_cache
    but for an arbitrary path (so we can reuse it for the Gemini cache)."""
    import json

    from . import attack

    cache: dict = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                cache[attack._cache_key(rec["original_id"], float(rec["severity"]))] = rec
    return cache


def _by_id(recs: list[dict]) -> dict:
    by_id: dict = {}
    for r in recs:
        by_id.setdefault(r["original_id"], {})[float(r["severity"])] = r
    return by_id


# --------------------------------------------------------------------------- #
# Scoring — predict_proba mirroring ev._score_block, transformer-style rows
# --------------------------------------------------------------------------- #
def _maybe_mask(texts, masked: bool):
    return [ablation.mask_urls(t) for t in texts] if masked else list(texts)


def _ham_test_texts():
    df = ev.build_dataset()
    test = df[df["split"] == "test"]
    ham = test[test["label"] == config.LABEL_HAM]["text"].fillna("").tolist()
    return ham


def _phish_curve_texts(inter, by_id, orig_text, sev):
    texts = []
    for oid in inter:
        if sev == 0.0:
            texts.append(orig_text[oid]["text"])
        else:
            texts.append(by_id[oid][sev]["rewrite_text"])
    return texts


def _clean_row():
    df = ev.build_dataset()
    test = df[df["split"] == "test"]
    ham = test[test["label"] == config.LABEL_HAM]["text"].fillna("").tolist()
    phish = test[test["label"] == config.LABEL_PHISHING]["text"].fillna("").tolist()
    ham_score = td.predict_proba(ham, CEAS_BASE_DIR)
    phish_score = td.predict_proba(phish, CEAS_BASE_DIR)
    thr = float(np.quantile(ham_score, 1.0 - TARGET_FPR))
    row = td._row(
        "clean", "na", "baseline", "original", 0.0, len(phish), ham_score, phish_score, thr
    )
    return {"corpus": "ceas08", **row}


def _degradation_block(generator, inter, by_id, orig_text, masked: bool):
    """One (generator, URL-condition) degradation curve + per-severity detection
    vectors @0.5 (for the McNemar paired test)."""
    condition = "url_masked" if masked else "original"
    ham_text = _maybe_mask(_ham_test_texts(), masked)
    ham_score = td.predict_proba(ham_text, CEAS_BASE_DIR)
    thr = float(np.quantile(ham_score, 1.0 - TARGET_FPR))
    rows, det = [], {}
    for sev in CURVE_SEVERITIES:
        ptexts = _maybe_mask(_phish_curve_texts(inter, by_id, orig_text, sev), masked)
        pscore = td.predict_proba(ptexts, CEAS_BASE_DIR)
        det[sev] = (pscore >= 0.5).astype(int)
        row = td._row(
            "degradation", generator, "baseline", condition, sev, len(inter), ham_score, pscore, thr
        )
        rows.append({"corpus": "ceas08", **row})
    return rows, det


def _mcnemar_rows(det: dict, generator: str, condition: str):
    d0, d1 = det[0.0], det[1.0]
    b = int(((d0 == 1) & (d1 == 0)).sum())
    c = int(((d0 == 0) & (d1 == 1)).sum())
    nd = b + c
    p = 1.0 if nd == 0 else float(binomtest(min(b, c), nd, 0.5, "two-sided").pvalue)
    return {
        "corpus": "ceas08",
        "generator": generator,
        "condition": condition,
        "model": "distilbert",
        "n": len(d0),
        "det_sev0": float(d0.mean()),
        "det_sev1": float(d1.mean()),
        "effect_drop": float(d0.mean() - d1.mean()),
        "mcnemar_b_lost": b,
        "mcnemar_c_gained": c,
        "mcnemar_exact_p": p,
    }


def score() -> None:
    config.ensure_dirs()
    df = ev.build_dataset()

    # ---- assemble rewrite caches (Haiku always; Gemini only if present) ----
    haiku = _by_id(list(ev._load_cache().values()))
    if not haiku:
        raise SystemExit(
            "No CEAS Haiku rewrites cached " f"({ev._rewrites_path().name}); cannot score."
        )
    generators = [("haiku", haiku)]
    gem_path = _gemini_ceas_path()
    gem = _by_id(list(_load_cache_at(gem_path).values()))
    if gem:
        generators.append(("gemini", gem))
        print(f"  Gemini CEAS cache present ({gem_path.name}) — scoring it too.")
    else:
        print(f"  Gemini CEAS cache absent ({gem_path.name}) — Haiku only (expected).")

    rows, mcn = [_clean_row()], []
    for generator, by_id in generators:
        inter = ev._intersection(by_id)
        orig_text = {
            oid: {"text": r["text"], "had_html": bool(r["had_html"])}
            for _, r in df.iterrows()
            if (oid := r["original_id"]) in set(by_id)
        }
        print(
            f"  [{generator}] intersection valid at all " f"{REWRITE_SEVERITIES} -> n={len(inter)}"
        )
        for masked in (False, True):
            r, det = _degradation_block(generator, inter, by_id, orig_text, masked)
            rows += r
            mcn.append(_mcnemar_rows(det, generator, "url_masked" if masked else "original"))

    out = pd.DataFrame(rows)
    path = config.TABLES_DIR / "transformer_ceas_degradation.csv"
    out.to_csv(path, index=False)
    mdf = pd.DataFrame(mcn)
    mpath = config.TABLES_DIR / "transformer_ceas_significance.csv"
    mdf.to_csv(mpath, index=False)
    _print_summary(out, mdf)
    print(f"\n  wrote {path.name} ({len(out)} rows) and {mpath.name} ({len(mdf)} rows)")


def _print_summary(out: pd.DataFrame, mcn: pd.DataFrame) -> None:
    fpr = int(TARGET_FPR * 100)
    deg = out[out["set"] == "degradation"]
    for gen in sorted(deg["generator"].unique()):
        for cond in ["original", "url_masked"]:
            s = deg[(deg.generator == gen) & (deg.condition == cond)].set_index("severity")
            if s.empty:
                continue
            print(f"\n  === {gen}/{cond} (recall@0.5 | det@{fpr}%FPR) ===")
            print(
                "    "
                + "  ".join(
                    f"s{sev:.2f}={s.loc[sev,'recall_05']:.3f}/{s.loc[sev,'recall_fpr']:.3f}"
                    for sev in CURVE_SEVERITIES
                )
            )
            d05 = s.loc[1.0, "recall_05"] - s.loc[0.0, "recall_05"]
            dfp = s.loc[1.0, "recall_fpr"] - s.loc[0.0, "recall_fpr"]
            print(f"        drop 0->1.0  recall@0.5 {d05:+.3f}  det@FPR {dfp:+.3f}")
    if not mcn.empty:
        print("\n  === McNemar exact (sev0 vs sev1) ===")
        for _, r in mcn.iterrows():
            print(
                f"    {r['generator']}/{r['condition']}: drop {r['effect_drop']*100:+.1f}pts "
                f"(b={r['mcnemar_b_lost']} c={r['mcnemar_c_gained']}) p={r['mcnemar_exact_p']:.2e}"
            )


# --------------------------------------------------------------------------- #
# Run report — write a standalone verdict file under logs/ (run-state, gitignored).
# RESULTS.md is maintained by hand and is not written here.
# --------------------------------------------------------------------------- #
ROBUST_EPS = 0.03  # |sev0->sev1 recall drop| below this == "robust" (matches primary)


def _verdict_md() -> str:
    deg = pd.read_csv(config.TABLES_DIR / "transformer_ceas_degradation.csv")
    mcn = pd.read_csv(config.TABLES_DIR / "transformer_ceas_significance.csv")
    clean = deg[deg["set"] == "clean"]
    hb = deg[(deg["set"] == "degradation") & (deg["generator"] == "haiku")]

    def curve(cond):
        s = hb[hb.condition == cond].set_index("severity")
        return s, (s.loc[1.0, "recall_05"] - s.loc[0.0, "recall_05"])

    s_o, drop_o = curve("original")
    s_m, drop_m = curve("url_masked")
    text_robust = abs(drop_o) < ROBUST_EPS
    url_robust = abs(drop_m) < ROBUST_EPS

    def row(s):
        return " | ".join(f"{s.loc[sev,'recall_05']*100:.1f}%" for sev in CURVE_SEVERITIES)

    mo = mcn[(mcn.generator == "haiku") & (mcn.condition == "original")].iloc[0]
    mm = mcn[(mcn.generator == "haiku") & (mcn.condition == "url_masked")].iloc[0]
    clean_recall = float(clean.iloc[0]["recall_05"]) if len(clean) else float("nan")
    n = int(hb["n_phish"].iloc[0])

    v_text = (
        f"**ROBUST** to rewriting (recall@0.5 moves {drop_o*100:+.1f} pts "
        f"sev0->sev1, McNemar p={mo['mcnemar_exact_p']:.2g}) — the same robustness "
        f"the Nazario transformer showed, reproduced on a same-era different corpus."
        if text_robust
        else f"**degrades** under rewriting ({drop_o*100:+.1f} pts, p={mo['mcnemar_exact_p']:.2g}) "
        f"— unlike the Nazario transformer, so the robustness was corpus-specific."
    )
    v_url = (
        f"**NOT URL-anchored** (url-masked recall@0.5 moves {drop_m*100:+.1f} pts "
        f"sev0->sev1, p={mm['mcnemar_exact_p']:.2g}; the semantic signal survives "
        f"masking) — reproducing the Nazario non-anchoring."
        if url_robust
        else f"**URL-anchored** here (url-masked drop {drop_m*100:+.1f} pts, "
        f"p={mm['mcnemar_exact_p']:.2g}) — masking removes the signal, unlike Nazario."
    )
    repro = (
        "Both the robustness and the non-URL-anchoring REPRODUCE on CEAS-2008, "
        "ruling out a Nazario-specific artifact: the transformer's behaviour is "
        "corpus-independent."
        if (text_robust and url_robust)
        else "The CEAS result diverges from Nazario (see above), so the transformer "
        "behaviour is at least partly corpus-dependent."
    )

    return f"""# CEAS-2008 transformer replication — run report

DistilBERT fine-tuned on the **CEAS-2008 train split** (same frugal config, seed 42,
class-weighted CE, 3 epochs) and scored on the **existing cached CEAS Haiku rewrites**
(no new API calls), through the identical pipeline as the primary transformer run:
clean baseline, degradation at all severities, URL-masked (inference-time masking),
recall@0.5 / det@1%FPR / PR-AUC with 1000-seed bootstrap CIs and McNemar exact
(sev0 vs sev1). Intersection **n={n}** (positives with URLs retained at all of
{{0.25,0.5,0.75,1.0}}). CEAS positives are generic spam (spam-vs-ham external check).

Clean test recall@0.5 = {clean_recall*100:.1f}%.

**recall@0.5 by severity (Haiku, intersection n={n})**

| condition | s0.00 | s0.25 | s0.50 | s0.75 | s1.00 | McNemar p (sev0→1) |
|---|---|---|---|---|---|---|
| original | {row(s_o)} | {mo['mcnemar_exact_p']:.2g} |
| url_masked | {row(s_m)} | {mm['mcnemar_exact_p']:.2g} |

**Verdict.**
- Text-robustness: {v_text}
- URL-anchoring: {v_url}
- Reproduction: {repro}
"""


def consolidate() -> None:
    """Write the CEAS transformer verdict to a standalone run report under logs/.
    RESULTS.md is maintained by hand and is intentionally not written here."""
    report = config.ROOT / "logs" / "ceas_transformer_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_verdict_md(), encoding="utf-8")
    print(f"  wrote CEAS transformer run report -> {report}")


def main(argv=None):
    p = argparse.ArgumentParser(description="DistilBERT on CEAS-2008 (no API)")
    p.add_argument("--train", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--consolidate", action="store_true")
    p.add_argument("--all", action="store_true")
    a = p.parse_args(argv)
    config.ensure_dirs()
    if a.all or a.train:
        finetune()
    if a.all or a.score:
        score()
    if a.all or a.consolidate:
        consolidate()
    if not (a.all or a.train or a.score or a.consolidate):
        p.print_help()


if __name__ == "__main__":
    main()
