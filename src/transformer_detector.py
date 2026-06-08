"""DistilBERT transformer baseline (defensive-research, no API).

Answers the reviewer question: does the LLM-rewrite degradation hold for a modern
neural detector, and is that detector also URL-anchored?

Everything is held identical to the classical pipeline so numbers are directly
comparable: SAME train split (22,560), SAME `text` field (subject+body), SAME
labels, SAME strict-270 Haiku intersection, SAME Gemini intersection (308), SAME
fixed-1%-FPR-on-clean-ham operating point, SAME 1000-seed bootstrap CIs, SAME
url-masking transform (`ablation.mask_urls`). For the transformer the url-masked
condition is INFERENCE-TIME masking of the inputs (there is no url_count feature
to zero), which differs from the classical URL-blind *retrain*; this is noted in
RESULTS.md.

Memory footprint (MPS OOM mitigation): inputs are head-truncated to the FIRST 256
tokens (max_len 256, truncation_side='right'); per-device batch 8 x grad-accum 2
(effective batch 16); gradient checkpointing on; step-based checkpoints every 200
steps with resume_from_checkpoint so a mid-epoch kill resumes rather than
restarts. 256-token truncation is acceptable because the phishing/spam signal is
concentrated in the subject + opening lines. Train applies identically to the
adversarial re-fine-tune so it does not OOM either.

  python -m src.transformer_detector --train         # fine-tune baseline
  python -m src.transformer_detector --train-adv      # fine-tune adv (augmented)
  python -m src.transformer_detector --score          # score all sets -> CSVs
  python -m src.transformer_detector --all            # train, train-adv, score
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Secondary OOM mitigation: disable the MPS allocator's high-watermark cap so it
# can release/recycle memory under pressure instead of refusing to free. Must be
# set before torch initialises the MPS backend, hence before `import torch`.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from . import ablation, config, features, mitigate
from .evaluate import (
    CURVE_SEVERITIES,
    _era_of,
    _orig_text_map,
    index_rewrites,
    intersection_ids,
    load_rewrites,
)

MODEL_NAME = "distilbert-base-uncased"
# Memory-frugal config (MPS OOM mitigation, 2026-06-07). max_len is the biggest
# lever: 512 -> 256 head-truncation. Inputs are truncated to the FIRST 256 tokens
# (truncation_side='right'), which is fine because the phishing/spam signal is
# concentrated early (subject + opening lines); noted in README/RESULTS.
MAX_LEN = 256
BATCH_TRAIN = 8  # was 16; halved to cut peak activation memory
GRAD_ACCUM = 2  # effective batch = 8 * 2 = 16 (unchanged vs original)
BATCH_INFER = 32
SAVE_STEPS = 200  # mid-epoch checkpoints so an OOM kill resumes, not restarts
EMPTY_CACHE_EVERY = 50  # torch.mps.empty_cache() cadence (steps)
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
VAL_FRAC = 0.10

BASE_DIR = config.MODELS_DIR / "distilbert"
ADV_DIR = config.MODELS_DIR / "distilbert_adv"
GEMINI_REWRITES = config.PROCESSED_DIR / "rewrites_gemini-2-5-flash.jsonl"


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _train_frame(augment: bool) -> pd.DataFrame:
    df = features.load_dataset()
    train = df[df["split"] == "train"][["text", "label"]].reset_index(drop=True)
    if not augment:
        return train
    recs = mitigate._load_train_cache()
    recs = list(recs.values()) if isinstance(recs, dict) else recs
    passing = mitigate._passing(recs)
    aug = pd.DataFrame(
        {
            "text": [r["rewrite_text"] for r in passing],
            "label": [config.LABEL_PHISHING] * len(passing),
        }
    )
    print(f"  augment: +{len(aug)} passing train rewrites -> {len(train) + len(aug)} rows")
    return pd.concat([train, aug], ignore_index=True)


def finetune(
    out_dir: Path, augment: bool, resume: bool = True, frame: pd.DataFrame | None = None
) -> None:
    """Fine-tune DistilBERT. If `frame` (cols text,label) is given it is used as
    the training pool verbatim (e.g. a different corpus's train split); otherwise
    the primary dataset is used via `_train_frame(augment)`. Everything else
    (frugal config, seed, class-weighted loss, splits) is identical."""
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    dev = _device()
    print(f"=== fine-tune {'ADV' if augment else 'baseline'} on {dev} -> {out_dir.name} ===")

    full = (
        _train_frame(augment) if frame is None else frame[["text", "label"]].reset_index(drop=True)
    )
    tr, va = train_test_split(
        full, test_size=VAL_FRAC, stratify=full["label"], random_state=config.SEED, shuffle=True
    )
    print(f"  train {len(tr)}  val {len(va)}  (stratified {VAL_FRAC:.0%} holdout)")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)  # truncation_side='right' = keep head

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LEN)

    ds_tr = (
        Dataset.from_pandas(tr[["text", "label"]], preserve_index=False)
        .map(tok_fn, batched=True)
        .rename_column("label", "labels")
    )
    ds_va = (
        Dataset.from_pandas(va[["text", "label"]], preserve_index=False)
        .map(tok_fn, batched=True)
        .rename_column("label", "labels")
    )

    classes = np.array([0, 1])
    cw = compute_class_weight("balanced", classes=classes, y=tr["label"].to_numpy())
    class_weights = torch.tensor(cw, dtype=torch.float32, device=dev)
    print(f"  class weights (balanced): {cw.round(4).tolist()}")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        return {
            "pr_auc": float(average_precision_score(labels, probs)),
            "roc_auc": float(roc_auc_score(labels, probs)),
        }

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=class_weights)
            return (loss, outputs) if return_outputs else loss

    from transformers import TrainerCallback

    class MpsEmptyCacheCallback(TrainerCallback):
        """Periodically release cached MPS memory to avoid swap/OOM kills."""

        def on_step_end(self, args, state, control, **kwargs):
            if dev == "mps" and state.global_step % EMPTY_CACHE_EVERY == 0:
                torch.mps.empty_cache()

    ckpt_dir = out_dir / "checkpoints"
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_INFER,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        save_total_limit=2,
        seed=config.SEED,
        data_seed=config.SEED,
        logging_steps=100,
        report_to="none",
        use_cpu=(dev == "cpu"),
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    from transformers import DataCollatorWithPadding

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[MpsEmptyCacheCallback()],
    )

    t0 = time.time()
    resume_ck = resume and ckpt_dir.exists() and any(ckpt_dir.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume_ck)
    wall = time.time() - t0
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(
        f"  best val pr_auc={trainer.state.best_metric:.4f}  "
        f"wall-clock={wall/60:.1f} min  saved -> {out_dir}"
    )


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
_CACHE: dict = {}


def _load(out_dir: Path):
    key = str(out_dir)
    if key not in _CACHE:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        dev = _device()
        tok = AutoTokenizer.from_pretrained(str(out_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(out_dir)).to(dev).eval()
        _CACHE[key] = (model, tok, dev)
    return _CACHE[key]


@torch.no_grad()
def predict_proba(texts: list[str], out_dir: Path) -> np.ndarray:
    model, tok, dev = _load(out_dir)
    out = np.empty(len(texts), dtype=float)
    for i in range(0, len(texts), BATCH_INFER):
        chunk = [t if isinstance(t, str) else "" for t in texts[i : i + BATCH_INFER]]
        enc = tok(chunk, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(
            dev
        )
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        out[i : i + len(chunk)] = probs
    return out


# --------------------------------------------------------------------------- #
# Scoring helpers (mirror evaluate/_score_curve + ablation fixed-FPR)
# --------------------------------------------------------------------------- #
def _maybe_mask(texts: list[str], masked: bool) -> list[str]:
    return [ablation.mask_urls(t) for t in texts] if masked else list(texts)


def _phish_texts(oids, by_id, orig_text, severity) -> list[str]:
    texts, _ = ablation._phish_texts(oids, by_id, severity, orig_text)
    return texts


def _ham_texts(masked: bool) -> list[str]:
    df = features.load_dataset()
    test = df[df["split"] == "test"]
    ham = test[test["label"] == config.LABEL_HAM]["text"].tolist()
    return _maybe_mask(ham, masked)


def _boot_cis(y, score, thr, seed=config.SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    acc = {k: [] for k in ("recall_05", "recall_fpr", "pr_auc", "roc_auc")}
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


def _row(set_label, generator, training, condition, severity, n_phish, ham_score, phish_score, thr):
    y = np.concatenate([np.zeros(len(ham_score), int), np.ones(len(phish_score), int)])
    score = np.concatenate([ham_score, phish_score])
    rec05 = float((phish_score >= 0.5).mean())
    recfpr = float((phish_score >= thr).mean())
    pr = float(average_precision_score(y, score))
    roc = float(roc_auc_score(y, score))
    cis = _boot_cis(y, score, thr)
    return {
        "set": set_label,
        "generator": generator,
        "training": training,
        "condition": condition,
        "model": "distilbert",
        "severity": severity,
        "n_phish": n_phish,
        "recall_05": rec05,
        "recall_05_lo": cis["recall_05"][0],
        "recall_05_hi": cis["recall_05"][1],
        "recall_fpr": recfpr,
        "recall_fpr_lo": cis["recall_fpr"][0],
        "recall_fpr_hi": cis["recall_fpr"][1],
        "pr_auc": pr,
        "pr_auc_lo": cis["pr_auc"][0],
        "pr_auc_hi": cis["pr_auc"][1],
        "roc_auc": roc,
        "roc_auc_lo": cis["roc_auc"][0],
        "roc_auc_hi": cis["roc_auc"][1],
        "fpr_threshold": float(thr),
    }


def _degradation_block(out_dir, generator, training, inter, by_id, orig_text, masked: bool):
    """One (model, generator, URL-condition) degradation curve."""
    condition = "url_masked" if masked else "original"
    ham_text = _ham_texts(masked)
    ham_score = predict_proba(ham_text, out_dir)
    thr = float(np.quantile(ham_score, 1.0 - ablation.TARGET_FPR))
    rows = []
    for sev in CURVE_SEVERITIES:
        ptexts = _maybe_mask(_phish_texts(inter, by_id, orig_text, sev), masked)
        pscore = predict_proba(ptexts, out_dir)
        rows.append(
            _row(
                "degradation",
                generator,
                training,
                condition,
                sev,
                len(inter),
                ham_score,
                pscore,
                thr,
            )
        )
    return rows


def _clean_row(out_dir):
    df = features.load_dataset()
    test = df[df["split"] == "test"]
    ham_score = predict_proba(test[test["label"] == config.LABEL_HAM]["text"].tolist(), out_dir)
    phish = test[test["label"] == config.LABEL_PHISHING]["text"].tolist()
    phish_score = predict_proba(phish, out_dir)
    thr = float(np.quantile(ham_score, 1.0 - ablation.TARGET_FPR))
    return _row("clean", "na", "baseline", "original", 0.0, len(phish), ham_score, phish_score, thr)


def _era_rows(out_dir, inter, by_id, orig_text):
    buckets: dict[str, list[str]] = {}
    for oid in inter:
        buckets.setdefault(_era_of(oid), []).append(oid)
    rows = []
    for era, oids in sorted(buckets.items()):
        for sev in CURVE_SEVERITIES:
            ptexts = _phish_texts(oids, by_id, orig_text, sev)
            pscore = predict_proba(ptexts, out_dir)
            rows.append(
                {
                    "era": era,
                    "model": "distilbert",
                    "severity": sev,
                    "n_phish": len(oids),
                    "detection_rate": float((pscore >= 0.5).mean()),
                }
            )
    return rows


def score_all() -> None:
    config.ensure_dirs()
    orig_text = _orig_text_map()

    haiku = index_rewrites(load_rewrites())
    inter_h = intersection_ids(haiku)
    gem = index_rewrites(load_rewrites(GEMINI_REWRITES))
    inter_g = intersection_ids(gem)
    print(f"  haiku intersection n={len(inter_h)}  gemini intersection n={len(inter_g)}")

    rows = [_clean_row(BASE_DIR)]
    # Baseline detector: Haiku + Gemini, both URL conditions.
    for masked in (False, True):
        rows += _degradation_block(BASE_DIR, "haiku", "baseline", inter_h, haiku, orig_text, masked)
        rows += _degradation_block(BASE_DIR, "gemini", "baseline", inter_g, gem, orig_text, masked)
    # Adversarially-trained detector (mitigation + cross-generator).
    for masked in (False, True):
        rows += _degradation_block(ADV_DIR, "haiku", "adv", inter_h, haiku, orig_text, masked)
        rows += _degradation_block(ADV_DIR, "gemini", "adv", inter_g, gem, orig_text, masked)

    out = pd.DataFrame(rows)
    path = config.TABLES_DIR / "transformer_degradation.csv"
    out.to_csv(path, index=False)
    print(f"  wrote {path.name}  ({len(out)} rows)")

    era = pd.DataFrame(_era_rows(BASE_DIR, inter_h, haiku, orig_text))
    epath = config.TABLES_DIR / "transformer_era.csv"
    era.to_csv(epath, index=False)
    print(f"  wrote {epath.name}  ({len(era)} rows)")


def main(argv=None):
    p = argparse.ArgumentParser(description="DistilBERT transformer baseline")
    p.add_argument("--train", action="store_true")
    p.add_argument("--train-adv", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--all", action="store_true")
    a = p.parse_args(argv)
    config.ensure_dirs()
    if a.all or a.train:
        finetune(BASE_DIR, augment=False)
    if a.all or a.train_adv:
        finetune(ADV_DIR, augment=True)
    if a.all or a.score:
        score_all()
    if not (a.all or a.train or a.train_adv or a.score):
        p.print_help()


if __name__ == "__main__":
    main()
