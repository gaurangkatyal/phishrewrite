"""Adversarial-training mitigation (defensive countermeasure to the attack).

Hypothesis: if a detector is TRAINED on LLM-rewritten phishing, it should resist
the same rewriting attack at test time. We test this end to end:

  1. Sample 500 PHISHING TRAINING emails (seeded, drawn from the train split, so
     zero overlap with the test emails the attack already rewrote).
  2. Rewrite each at severities 0.5 and 1.0 (= 1,000 Haiku calls) using the SAME
     system/severity prompts, prompt cache, and URL-retention machinery as the
     attack. Cached to data/processed/rewrites_train.jsonl (separate file).
  3. Augment the training set with the rewrites that pass URL retention (label=1).
  4. Retrain all three detectors on the augmented set — both URL-intact and
     URL-blind (so the mitigation is measured under both conditions).
  5. Re-score against the EXISTING test rewrites (the attack's intersection set)
     and compare baseline vs adversarially-trained, under both URL conditions.

Spending is gated exactly like the attack:

    python -m src.mitigate --estimate        # projected calls + cost (free)
    python -m src.mitigate --run --yes        # 1,000 calls, then retrain + rescore
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from . import ablation, attack, config, features
from .detectors import _fit, model_factories

# Augment with rewrites at the two most aggressive (most evasive) severities.
MITIGATE_SEVERITIES: tuple[float, ...] = (0.5, 1.0)
TRAIN_PHISH_SAMPLE: int = 500

REWRITES_TRAIN_JSONL = config.PROCESSED_DIR / "rewrites_train.jsonl"

# Adversarially-trained model + transformer cache paths.
ADV_VEC = config.MODELS_DIR / "tfidf_vectorizer_adv.joblib"
ADV_SCALER = config.MODELS_DIR / "handcrafted_scaler_adv.joblib"
ADV_VEC_MASKED = config.MODELS_DIR / "tfidf_vectorizer_adv_urlmasked.joblib"
ADV_SCALER_MASKED = config.MODELS_DIR / "handcrafted_scaler_adv_urlmasked.joblib"


def _adv_model_path(name: str, masked: bool):
    return config.MODELS_DIR / f"{name}_adv{'_urlmasked' if masked else ''}.joblib"


# --------------------------------------------------------------------------- #
# Seeded train-phishing sample (disjoint from test by construction)
# --------------------------------------------------------------------------- #
def sample_phish_train(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(config.DATASET_CSV)
    pool = df[(df["split"] == "train") & (df["label"] == config.LABEL_PHISHING)]
    n = min(TRAIN_PHISH_SAMPLE, len(pool))
    sample = pool.sample(n=n, random_state=config.SEED).reset_index(drop=True)
    if limit is not None:
        sample = sample.iloc[:limit].reset_index(drop=True)
    return sample


def _severity_prompts() -> dict[float, str]:
    out: dict[float, str] = {}
    for sev in MITIGATE_SEVERITIES:
        path = config.SEVERITY_PROMPT_FILES[sev]
        out[sev] = path.read_text(encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Cost estimate (no API calls)
# --------------------------------------------------------------------------- #
def estimate() -> None:
    system = attack.load_system_prompt()
    sevs = _severity_prompts()
    sample = sample_phish_train()
    n_emails = len(sample)
    n_calls = n_emails * len(sevs)

    sys_tok = attack._approx_tokens(system)
    instr_tok = [attack._approx_tokens(t) for t in sevs.values()]
    avg_instr = sum(instr_tok) / len(instr_tok)
    email_toks = [attack._approx_tokens(t) for t in sample["text"].fillna("").astype(str)]
    avg_email = sum(email_toks) / max(1, len(email_toks))
    avg_out = min(config.ATTACK_MAX_TOKENS, max(64, avg_email))

    model = config.model_for()
    in_price, out_price = config.PRICE_PER_MTOK.get(model, (1.0, 5.0))
    fresh_in = n_calls * (avg_instr + avg_email)
    if config.USE_PROMPT_CACHE:
        sys_eff = sys_tok * 1.25 + (n_calls - 1) * sys_tok * 0.1
    else:
        sys_eff = n_calls * sys_tok
    total_in = fresh_in + sys_eff
    total_out = n_calls * avg_out
    in_cost = total_in / 1e6 * in_price
    out_cost = total_out / 1e6 * out_price
    subtotal = in_cost + out_cost
    sync_total = subtotal  # this run is synchronous (no batch discount)
    batch_total = subtotal * 0.5

    print("=== mitigation cost estimate (rough upper bound) ===")
    print(f"  provider/model : {config.ATTACK_PROVIDER} / {model}")
    print(f"  price (USD/MTok): input {in_price}, output {out_price}")
    print(
        f"  train phishing pool sampled : {n_emails} (seeded; train split, " f"0 overlap with test)"
    )
    print(f"  severities : {list(sevs)} -> augment at the two most evasive levels")
    print(f"  projected calls : {n_emails} x {len(sevs)} = {n_calls}")
    print(
        f"  avg tokens : system={sys_tok}, instruction~={avg_instr:.0f}, "
        f"email~={avg_email:.0f}, output~={avg_out:.0f}"
    )
    print(f"  prompt caching : {'on' if config.USE_PROMPT_CACHE else 'off'}")
    print(f"  est. input tokens  : {total_in/1e6:.3f} M  -> ${in_cost:.2f}")
    print(f"  est. output tokens : {total_out/1e6:.3f} M  -> ${out_cost:.2f}")
    print(f"  est. subtotal : ${subtotal:.2f}")
    print(f"  ESTIMATED TOTAL (synchronous, no batch) : ${sync_total:.2f}")
    print(f"  (for reference, with Batch API 50% off : ${batch_total:.2f})")
    print("  (heuristic char/4 token counts; actual usage recorded per call.)")


# --------------------------------------------------------------------------- #
# Rewriting (reuses attack machinery; separate cache file)
# --------------------------------------------------------------------------- #
def _load_train_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not REWRITES_TRAIN_JSONL.exists():
        return cache
    with REWRITES_TRAIN_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                cache[attack._cache_key(rec["original_id"], float(rec["severity"]))] = rec
    return cache


def _append_train(rec: dict) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with REWRITES_TRAIN_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def rewrite_train(sample: pd.DataFrame) -> list[dict]:
    system = attack.load_system_prompt()
    sevs = _severity_prompts()
    rewriter = attack.Rewriter(
        config.ATTACK_PROVIDER,
        config.model_for(),
        config.ATTACK_TEMPERATURE,
        config.ATTACK_MAX_TOKENS,
    )
    cache = _load_train_cache()
    records: list[dict] = []
    todo = [(row, sev) for _, row in sample.iterrows() for sev in sevs]
    n_total = len(todo)
    n_done = n_skip = n_err = 0
    print(
        f"=== MITIGATION rewrite: {len(sample)} train emails x {len(sevs)} "
        f"severities = {n_total} calls ==="
    )
    t0 = time.time()
    for row, sev in todo:
        key = attack._cache_key(row["original_id"], sev)
        ph = attack.prompt_hash(
            system, sevs[sev], row["text"], rewriter.model, rewriter.temperature
        )
        th = attack.text_sha(row["text"])
        cached = cache.get(key)
        if cached and cached.get("prompt_hash") == ph and cached.get("input_text_sha", th) == th:
            n_skip += 1
            records.append(cached)
            continue
        try:
            raw, usage = rewriter.rewrite(sevs[sev], row["text"])
        except Exception as e:  # noqa: BLE001
            fatal = attack.fatal_account_message(e)
            if fatal:
                raise SystemExit(
                    f"\nAborting mitigation rewrite: account-level error from "
                    f"{rewriter.provider!r} that no retry/resume can fix:\n"
                    f"  {fatal}\n"
                    f"  Progress is cached ({n_done} new this run); fix the "
                    f"account and re-run to resume from the cache."
                )
            n_err += 1
            msg = f"{type(e).__name__}: {str(e).replace(chr(10), ' ')[:200]}"
            print(
                f"  [{n_done + n_skip + n_err}/{n_total}] id="
                f"{row['original_id']} sev={sev:.2f} ERROR (skipped): {msg}"
            )
            continue
        rec = attack.make_record(row, sev, sevs[sev], system, rewriter, raw, usage)
        _append_train(rec)
        cache[key] = rec
        records.append(rec)
        n_done += 1
        print(
            f"  [{n_done + n_skip + n_err}/{n_total}] id={row['original_id']} "
            f"sev={sev:.2f} retention={attack._retention_label(rec)}"
        )
    print(
        f"\n  done: {n_done} new, {n_skip} cached, {n_err} errored/skipped, "
        f"in {time.time() - t0:.1f}s"
    )
    attack._retention_summary(records)
    return records


# --------------------------------------------------------------------------- #
# Augment + retrain
# --------------------------------------------------------------------------- #
def _passing(records: list[dict]) -> list[dict]:
    return [r for r in records if (r.get("retained_urls") is True) and not r.get("refused")]


def _build_and_fit(
    train_text: pd.Series, had_html: pd.Series, y: np.ndarray, vec_path, scaler_path, masked: bool
) -> None:
    text = train_text.map(ablation.mask_urls) if masked else train_text
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
    frame = pd.DataFrame({"text": text.to_numpy(), "had_html": had_html.to_numpy()})
    hc = scaler.fit_transform(features.handcrafted_matrix(frame))
    x = sparse.hstack([tfidf, sparse.csr_matrix(hc)]).tocsr()
    joblib.dump(vec, vec_path)
    joblib.dump(scaler, scaler_path)
    for name, factory in model_factories().items():
        clf = _fit(name, factory(), x, y)
        joblib.dump(clf, _adv_model_path(name, masked))
    print(
        f"  [{'url_masked' if masked else 'original'}] augmented train {x.shape}; "
        f"3 models cached"
    )


def augment_and_retrain(records: list[dict]) -> int:
    df = features.load_dataset()
    train = df[df["split"] == "train"].reset_index(drop=True)
    passing = _passing(records)
    aug = pd.DataFrame(
        {
            "text": [r["rewrite_text"] for r in passing],
            "had_html": [bool(r.get("had_html", False)) for r in passing],
            "label": [config.LABEL_PHISHING] * len(passing),
        }
    )
    full = pd.concat([train[["text", "had_html", "label"]], aug], ignore_index=True)
    y = full["label"].to_numpy()
    print("=== augment + retrain ===")
    print(
        f"  original train rows: {len(train)}  + passing rewrites: {len(aug)} "
        f"-> augmented: {len(full)}"
    )
    _build_and_fit(full["text"], full["had_html"], y, ADV_VEC, ADV_SCALER, masked=False)
    _build_and_fit(
        full["text"], full["had_html"], y, ADV_VEC_MASKED, ADV_SCALER_MASKED, masked=True
    )
    return len(aug)


# --------------------------------------------------------------------------- #
# Rescore baseline vs adversarial on the existing TEST rewrites
# --------------------------------------------------------------------------- #
def _adv_transform(texts, had_html):
    vec, scaler = joblib.load(ADV_VEC), joblib.load(ADV_SCALER)
    frame = pd.DataFrame({"text": list(texts), "had_html": list(had_html)})
    tfidf = vec.transform(frame["text"])
    hc = scaler.transform(features.handcrafted_matrix(frame))
    return sparse.hstack([tfidf, sparse.csr_matrix(hc)]).tocsr()


def _adv_masked_transform(texts, had_html):
    vec, scaler = joblib.load(ADV_VEC_MASKED), joblib.load(ADV_SCALER_MASKED)
    masked = [ablation.mask_urls(t) for t in texts]
    frame = pd.DataFrame({"text": masked, "had_html": list(had_html)})
    tfidf = vec.transform(frame["text"])
    hc = scaler.transform(features.handcrafted_matrix(frame))
    return sparse.hstack([tfidf, sparse.csr_matrix(hc)]).tocsr()


def _rescore_on(by_id, orig_text, inter) -> pd.DataFrame:
    """Score the 4 (URL-condition x training) cells on a fixed phishing
    intersection set. Shared by the headline (Haiku) rescore and the
    cross-generator (Gemini) validation so both are computed identically."""
    base = {n: joblib.load(config.MODELS_DIR / f"{n}.joblib") for n in model_factories()}
    base_m = {n: joblib.load(ablation._masked_model_path(n)) for n in model_factories()}
    adv = {n: joblib.load(_adv_model_path(n, False)) for n in model_factories()}
    adv_m = {n: joblib.load(_adv_model_path(n, True)) for n in model_factories()}

    conditions = [
        ("original", "baseline", base, features.transform_texts),
        ("original", "adv_trained", adv, _adv_transform),
        ("url_masked", "baseline", base_m, ablation.masked_transform),
        ("url_masked", "adv_trained", adv_m, _adv_masked_transform),
    ]
    frames = []
    for url_cond, train_cond, models, tf in conditions:
        d = ablation.score_condition(models, tf, inter, by_id, orig_text)
        d["url_condition"] = url_cond
        d["training"] = train_cond
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def rescore() -> None:
    test_recs = ablation.load_rewrites()
    by_id = ablation.index_rewrites(test_recs)
    orig_text = ablation._orig_text_map()
    inter = ablation.intersection_ids(by_id)
    print(f"=== rescore on test intersection (n_phish={len(inter)}) ===")

    out = _rescore_on(by_id, orig_text, inter)
    path = config.TABLES_DIR / "mitigation_rescore.csv"
    out.to_csv(path, index=False)
    _print_mitigation(out)
    print(f"\n  wrote {path.name}")


def rescore_cross(rewrites_path, out_name: str) -> None:
    """Cross-generator validation: score the HAIKU-trained adversarial detectors
    (and the baselines) against rewrites produced by a DIFFERENT generator. Tests
    whether the adversarial-training recovery generalizes across rewriting models
    or is generator-specific. Pure compute (no API calls)."""
    test_recs = ablation.load_rewrites(rewrites_path)
    by_id = ablation.index_rewrites(test_recs)
    orig_text = ablation._orig_text_map()
    inter = ablation.intersection_ids(by_id)
    print(
        f"=== cross-generator rescore on {rewrites_path.name} "
        f"intersection (n_phish={len(inter)}) ==="
    )

    out = _rescore_on(by_id, orig_text, inter)
    path = config.TABLES_DIR / out_name
    out.to_csv(path, index=False)
    _print_mitigation(out)
    print(f"\n  wrote {path.name}")


def _print_mitigation(out: pd.DataFrame) -> None:
    fpr = int(ablation.TARGET_FPR * 100)
    for url_cond in ["original", "url_masked"]:
        print(f"\n  === URL condition: {url_cond}  (detection@{fpr}%FPR) ===")
        for name in sorted(out["model"].unique()):
            b = out[
                (out.url_condition == url_cond) & (out.training == "baseline") & (out.model == name)
            ].set_index("severity")
            a = out[
                (out.url_condition == url_cond)
                & (out.training == "adv_trained")
                & (out.model == name)
            ].set_index("severity")
            print(f"    {name}:   {'sev':>4}  {'baseline':>9} {'adv':>7}  {'Δ':>7}")
            for s in ablation.CURVE_SEVERITIES:
                bv, av = b.loc[s, "recall_fpr"], a.loc[s, "recall_fpr"]
                print(f"               {s:>4.2f}  {bv:>9.3f} {av:>7.3f}  {av - bv:>+7.3f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_full() -> None:
    # Pre-flight: fail fast if the provider's API key is missing/empty, before
    # sampling or any rewrite call — so a blank key never aborts mid-spend.
    config.require_api_key(config.ATTACK_PROVIDER)
    sample = sample_phish_train()
    records = rewrite_train(sample)
    augment_and_retrain(records)
    rescore()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PhishRewrite adversarial-training mitigation")
    parser.add_argument(
        "--estimate", action="store_true", help="projected calls + cost (free, no API calls)"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="rewrite 1,000 train phish, augment, retrain, rescore (gated)",
    )
    parser.add_argument("--yes", action="store_true", help="confirm spending on --run")
    parser.add_argument(
        "--cross-gemini",
        action="store_true",
        help="cross-generator validation: score Haiku-trained adv "
        "detectors against the Gemini rewrites (free, no API calls)",
    )
    args = parser.parse_args(argv)

    config.ensure_dirs()
    if args.cross_gemini:
        gem = config.PROCESSED_DIR / "rewrites_gemini-2-5-flash.jsonl"
        rescore_cross(gem, "mitigation_cross_gemini.csv")
        return
    if args.estimate:
        estimate()
        return
    if args.run:
        if not args.yes:
            print(
                "Refusing to spend without --yes. Review the estimate first:\n"
                "    python -m src.mitigate --estimate\n"
                "then re-run:\n"
                "    python -m src.mitigate --run --yes"
            )
            sys.exit(2)
        run_full()
        return
    parser.print_help()


if __name__ == "__main__":
    main()
