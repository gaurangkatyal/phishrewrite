# PhishRewrite

A benchmark for measuring how phishing-email detectors degrade when phishing text
is rewritten by a large language model.

> Defensive research only. PhishRewrite quantifies a weakness in content-based
> abuse detectors so that builders can address it. See [`ETHICS.md`](ETHICS.md)
> before using this project. Do not use it to attack anyone.

## Overview

Content-based phishing detectors rely heavily on surface text. PhishRewrite tests
how robust they are: we train standard detectors on public phishing-vs-ham email,
then rewrite held-out phishing emails with an LLM at increasing severity so they
read as benign while preserving the malicious ask and its URLs, and measure the
change in detection accuracy.

The main result is a **degradation curve** — detection rate, PR-AUC, and
ROC-AUC as a function of rewrite severity — reported for classical bag-of-words
detectors and for a DistilBERT baseline, with bootstrap confidence intervals and
paired significance tests. Headline numbers, intervals, and figure/table
references are in [`RESULTS.md`](RESULTS.md).

## Datasets

| Source | Role | Label |
|---|---|---|
| SpamAssassin (`easy_ham`, `easy_ham_2`, `hard_ham` — ham only) | legitimate email | 0 |
| Nazario phishing corpus | phishing (the attack target) | 1 |
| Enron (sampled ham) | additional legitimate email | 0 |
| CEAS-2008 | external-validity check (spam vs ham) | 1 / 0 |

SpamAssassin `spam` is excluded: generic spam is not phishing and would add label
noise. Provenance and terms are documented in [`DATA_LICENSE`](DATA_LICENSE).

The merged primary dataset is kept at its natural ~1:4.2 phishing:ham ratio
(5,390 phishing, 22,811 ham; 28,201 total; no subsampling). Imbalance is handled
inside the detectors — `class_weight="balanced"` for LogisticRegression and
RandomForest, equivalent per-sample weights for GradientBoosting — so PR-AUC is
reported alongside ROC-AUC and recall as the primary metric.

## Repository layout

```
src/config.py            single source of truth: seed, paths, severities, attack config
src/data_prep.py         load, clean, and unify the corpora into one labeled table
src/features.py          TF-IDF + handcrafted features
src/detectors.py         train and cross-validate LR, RF, GB
src/attack.py            LLM rewriting attack across severity levels
src/evaluate.py          clean baseline, degradation curve, era ablation
src/ablation.py          URL-blind retrain and URL-masked evaluation
src/mitigate.py          adversarial re-training on passing rewrites
src/mitigate_reverse.py  reverse-direction mitigation check
src/significance.py      paired significance tests
src/transformer_detector.py  DistilBERT baseline + degradation
src/transformer_ceas.py  DistilBERT replication on CEAS-2008
src/external_validity.py classical detectors on CEAS-2008
src/release.py           package the public, defanged benchmark
src/prompts/             severity rewrite prompts (part of the benchmark)
results/tables/          CSV outputs
results/figures/         plots
scripts/                 run tooling (supervisors, finalizer); not part of the method
tests/                   smoke tests
data/, results/models/   downloads and trained artifacts (gitignored)
```

## Install

Requires Python >= 3.10 (3.12 is the tested interpreter).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY (and GEMINI_API_KEY for the cross-model run)
```

## Reproduce

```bash
python -m src.data_prep                  # download, clean, unify, split   [downloads data]
python -m src.features                   # build feature matrices
python -m src.detectors                  # train + 5-fold cross-validation
python -m src.evaluate --clean           # clean test baseline (severity 0.0)
python -m src.attack                     # LLM rewrites of held-out phishing  [API cost]
python -m src.evaluate --degradation     # degradation curve + era ablation
python -m src.evaluate --gemini          # cross-model curve (Gemini rewrites)
python -m src.ablation                   # URL-blind / URL-masked ablations
python -m src.mitigate                   # adversarial re-training
python -m src.significance               # paired significance tests
python -m src.transformer_detector --all # DistilBERT baseline + degradation
python -m src.external_validity --all     # CEAS-2008 external check (classical)
python -m src.transformer_ceas --all      # CEAS-2008 replication (DistilBERT)
python -m src.release                    # package the public benchmark
```

Steps that download data or spend on the API print a projected cost and pause for
confirmation before running. The attack caches rewrites on disk and is
idempotent, so a re-run resumes rather than re-spends.

## Reproducibility

- A single seed (`src/config.py: SEED = 42`) governs the train/test split,
  cross-validation, phishing sampling, bootstrap resampling, and every model's
  `random_state`.
- Dependencies are pinned in `requirements.txt`.
- The TF-IDF vectorizer is fit on the training split only.
- Handcrafted features (including `url_count`) are computed pre-defang on verbatim
  URLs. The released benchmark is defanged (`http://` → `hxxp://`), and the
  defang is losslessly reversible via `release/refang.py`, so recomputing features
  on the released text reproduces the original values.

## Limitations

See [`LIMITATIONS.md`](LIMITATIONS.md). In brief: the strict all-URLs retention
checker is conservative, so the headline attack-success set is a lower bound (a
primary-URL variant recovers more emails); ROC-AUC is insensitive under class
imbalance, so recall@0.5, PR-AUC, and detection@1%-FPR are the load-bearing
metrics; the corpus is English-only; and the URL-frozen contract is a deliberate
lower bound, with the URL-masked curve giving the text-only bound.

## Release

`python -m src.release` packages the public benchmark under `release/` and stages
a Zenodo-ready archive in `dist/`:

- `release/data/emails.csv` (28,201 rows) and `release/data/rewrites.csv`
  (Haiku + Gemini) with all URLs defanged. The transform is scheme-only and
  reversible via `release/refang.py`.
- `release/README.md` is a Hugging Face dataset card with `configs:` frontmatter,
  so `load_dataset("./release", "emails")` and `"rewrites"` work directly.
- `PROVENANCE.md`, `LICENSE`, `ETHICS.md`, `LIMITATIONS.md`, and the severity
  `prompts/` travel with the data.

Note: `release/` (including `release/refang.py` and `PROVENANCE.md`) is generated
by `python -m src.release` and is not committed to this repo — the packaged
benchmark is hosted on Zenodo. A fresh clone will not contain these files until
that command is run.

The attack is provider-abstracted (`ATTACK_PROVIDER`, `ATTACK_MODEL`). The
headline run uses Claude Haiku; setting the provider to `gemini` reproduces
`results/tables/degradation_gemini.csv`. Caches are model-slugged, so providers
never collide.

## Citation

```bibtex
@misc{phishrewrite,
  author    = {Katyal, Gaurang},
  title     = {PhishRewrite: a benchmark for LLM-driven evasion of phishing detectors},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0},
  doi       = {10.5281/zenodo.21018700},
  url       = {https://doi.org/10.5281/zenodo.21018700}
}
```

## Ethics and license

- Defensive use only; see [`ETHICS.md`](ETHICS.md).
- Code: MIT ([`LICENSE`](LICENSE)). Data and derived benchmark:
  [`DATA_LICENSE`](DATA_LICENSE).
