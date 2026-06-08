# PhishRewrite — Results manifest

Consolidated headline numbers with 95% bootstrap confidence intervals
(`N_BOOTSTRAP = 1000`, percentile method; single seed `SEED = 42`). Every figure
in this file is reproduced by a CSV under `results/tables/`; pointers are given
per section and collected at the end.

**One-line finding:** content-based phishing detectors degrade under LLM rewriting
that preserves the malicious ask + URLs verbatim. The drop is largest for the
linear model and deepens further once the frozen URL signal is removed
(`url_masked`), confirming the URL-intact curve is a lower bound. Adversarial
training recovers — and exceeds — clean detection, **and the recovery generalizes
across rewriting models**: detectors trained only on Haiku rewrites also recover
detection on held-out Gemini rewrites (§8b).

---

## 1. Dataset and split

| | phishing (1) | ham (0) | total |
|---|---:|---:|---:|
| nazario | 5,390 | — | 5,390 |
| enron | — | 18,700 | 18,700 |
| spamassassin (ham only) | — | 4,111 | 4,111 |
| **total** | **5,390** | **22,811** | **28,201** |

Natural imbalance ~1 : 4.23; no resampling. Split: 22,560 train / 5,641 test
(stratified, seeded). Source: `results/tables/class_balance.csv`.

---

## 2. Clean baseline (no attack)

5-fold CV on train (`cv_baseline.csv`) and held-out clean test (`test_clean.csv`):

| model | test ROC-AUC | test PR-AUC | test F1 | test precision | test recall |
|---|---:|---:|---:|---:|---:|
| logreg | 0.999 | 0.997 | 0.975 | 0.965 | 0.985 |
| random_forest | 1.000 | 0.998 | 0.977 | 0.996 | 0.958 |
| gradient_boosting | 0.999 | 0.996 | 0.973 | 0.977 | 0.970 |

All three detectors are strong on clean data — the attack operates on an
honest, well-calibrated baseline.

---

## 3. Retention funnel (label-validity contract)

500 held-out phishing emails rewritten at each of 4 severities (Haiku headline
run). A rewrite is a valid adversarial example only if it preserves the malicious
ask **and** its URL(s). Two definitions (`retention_definitions.csv`):

- **strict (all-URLs):** every original URL must survive.
- **primary-URL:** only the ask-bearing URL(s) must survive (incidental
  footer/tracking/brand-nav URLs may drop).

| severity | strict PASS | strict FAIL | primary PASS | primary FAIL | manual | refused |
|---|---:|---:|---:|---:|---:|---:|
| 0.25 | 362 | 31 | 372 | 11 | 104/114 | 3 |
| 0.50 | 328 | 65 | 363 | 20 | 104/114 | 3 |
| 0.75 | 314 | 80 | 359 | 25 | 103/113 | 3 |
| 1.00 | 303 | 89 | 351 | 31 | 80/90 | 28 |

**Intersection valid at ALL severities (the scored degradation set):**

| definition | intersection size |
|---|---:|
| strict (headline) | **270** |
| primary-URL | **331** |
| **delta** | **+61** |

The +61 emails are strict-FAILs whose dropped URLs were purely incidental
(brand/regulator footers, unsubscribe links, trackers) while the credential
lander survived. The headline set remains the **strict 270** — the primary
variant is reported as a sensitivity bound (the strict set is a *lower bound* on
attack success; see `LIMITATIONS.md` §1). Manual column shows strict/primary
counts (no-URL anchors flagged for spot-check, not auto-scored).

---

## 4. Headline degradation — strict intersection, n = 270 (Claude Haiku)

Detection vs. severity, **URLs intact** (`degradation_intersection.csv`,
`url_ablation_degradation.csv` condition=`original`). recall@0.5 = phishing
detection rate; det@1%FPR = detection at a ham-set 1%-FPR threshold (fixed across
severities). CIs are 95% bootstrap.

### recall @ 0.5 (with 95% CI)

| model | sev 0.0 | sev 0.5 | sev 1.0 | Δ(0→1) |
|---|---|---|---|---:|
| logreg | 0.989 [0.976, 1.000] | 0.911 [0.875, 0.943] | 0.867 [0.827, 0.904] | **−0.122** |
| random_forest | 0.974 [0.955, 0.992] | 0.963 [0.937, 0.983] | 0.922 [0.887, 0.953] | −0.052 |
| gradient_boosting | 0.981 [0.965, 0.996] | 0.981 [0.964, 0.996] | 0.970 [0.948, 0.989] | −0.011 |

### detection @ 1% FPR

| model | sev 0.0 | sev 0.5 | sev 1.0 | Δ(0→1) |
|---|---:|---:|---:|---:|
| logreg | 0.989 | 0.922 | 0.881 | **−0.107** |
| random_forest | 0.993 | 0.989 | 0.993 | 0.000 |
| gradient_boosting | 0.989 | 0.989 | 0.985 | −0.004 |

### PR-AUC (with 95% CI)

| model | sev 0.0 | sev 1.0 | Δ(0→1) |
|---|---|---|---:|
| logreg | 0.992 [0.986, 0.997] | 0.903 [0.862, 0.944] | **−0.089** |
| random_forest | 0.995 [0.990, 0.999] | 0.992 [0.983, 0.998] | −0.003 |
| gradient_boosting | 0.993 [0.986, 0.997] | 0.988 [0.980, 0.994] | −0.005 |

### Why not ROC-AUC

Over the same sweep logreg ROC-AUC moves only **0.999 → 0.996** while its
recall@0.5 falls **0.989 → 0.867** — ROC-AUC is insensitive under this imbalance
and is reported for completeness only (`LIMITATIONS.md` §2). The linear detector
is the most fragile; the tree ensembles lean on the (frozen) URL feature and
barely move — which §5 tests directly.

---

## 5. URL-masked degradation — the realistic text-only bound, n = 270

Retrain/score with URLs masked (`url_ablation_degradation.csv`
condition=`url_masked`). Removing the frozen URL signal roughly doubles the
linear-model collapse and finally moves the trees.

### recall @ 0.5

| model | sev 0.0 | sev 1.0 | Δ(0→1) | (vs URL-intact Δ) |
|---|---:|---:|---:|---:|
| logreg | 0.985 | 0.763 | **−0.222** | (−0.122) |
| random_forest | 0.948 | 0.856 | −0.093 | (−0.052) |
| gradient_boosting | 0.978 | 0.926 | −0.052 | (−0.011) |

### detection @ 1% FPR

| model | sev 0.0 | sev 1.0 | Δ(0→1) | (vs URL-intact Δ) |
|---|---:|---:|---:|---:|
| logreg | 0.985 | 0.767 | **−0.219** | (−0.107) |
| random_forest | 0.981 | 0.981 | 0.000 | (0.000) |
| gradient_boosting | 0.978 | 0.937 | −0.041 | (−0.004) |

The URL-intact headline understates a URL-mutating attacker; the URL-masked curve
is the honest text-only bound (`LIMITATIONS.md` §4).

---

## 6. Era ablation (detection rate, `degradation_era_ablation.csv`)

Strict-set split by corpus era — legacy `*.mbox` (eBay/PayPal era, n=128) vs
modern by-year (n=142). recall@0.5:

| era | model | sev 0.0 | sev 1.0 | Δ |
|---|---|---:|---:|---:|
| legacy_mbox | logreg | 0.984 | 0.938 | −0.047 |
| modern_year | logreg | 0.993 | 0.803 | **−0.190** |
| legacy_mbox | gradient_boosting | 0.977 | 0.984 | +0.008 |
| modern_year | gradient_boosting | 0.986 | 0.958 | −0.028 |

The attack degrades **modern** phishing more, but read this as indicative only:
the strict checker over-fails the brand-footer-heavy legacy bucket, so this split
is on a non-random subsample (`LIMITATIONS.md` §1a; primary-URL recovers legacy
128→182 vs modern 142→149).

---

## 7. Cross-model confirmation — Gemini 2.5 Flash, n = 308 (`degradation_gemini.csv`)

Independent rewriting model, separate billing, scored against the same baseline
detectors. The effect is not Haiku-specific:

| model | recall@0.5 sev 0.0 | sev 1.0 | Δ(0→1) |
|---|---:|---:|---:|
| logreg | 0.990 | 0.948 | **−0.042** |
| random_forest | 0.981 | 0.968 | −0.013 |
| gradient_boosting | 0.987 | 0.984 | −0.003 |

Same qualitative shape (linear model most fragile, trees URL-anchored). Two
models is confirmation, not an exhaustive survey (`LIMITATIONS.md` §5).

---

## 8. Mitigation — adversarial training, Haiku-on-Haiku (`mitigation_rescore.csv`)

Augment train with passing severity≥0.5 Haiku rewrites (+614 rows → 23,174 train),
retrain, rescore the same n=270 Haiku set. Baseline vs `adv_trained`:

### recall @ 0.5, URLs intact

| model | sev 1.0 baseline | sev 1.0 adv_trained |
|---|---:|---:|
| logreg | 0.867 | **0.989** |
| random_forest | 0.922 | **1.000** |
| gradient_boosting | 0.970 | 0.985 |

### recall @ 0.5, URL-masked (the hard case)

| model | sev 1.0 baseline | sev 1.0 adv_trained |
|---|---:|---:|
| logreg | 0.763 | **0.989** |
| random_forest | 0.856 | 0.989 |
| gradient_boosting | 0.926 | 0.985 |

Adversarial training closes the gap entirely — even URL-masked sev-1.0 logreg
recovers from 0.763 to 0.989, at/above its own clean baseline (0.985). The
degradation is a fixable training-distribution gap, not an inherent ceiling.

---

## 8b. Cross-generator validation — Haiku-trained detectors vs Gemini rewrites (`mitigation_cross_gemini.csv`)

**The generalization test.** §8 trains and evaluates on the same generator
(Haiku), so its recovery could be generator-specific overfitting. Here the
*identical* Haiku-trained adversarial detectors are scored against the held-out
**Gemini** rewrites (n=308 Gemini intersection, same fixed-FPR / intersection-set
methodology). If recovery is real it should transfer to a generator the detector
never trained on.

**Result: it transfers.** Every cell shows the adversarial model ≥ baseline (one
−0.003 noise exception, RF url_masked sev 0.25 det@1%FPR), and the recovery is
largest exactly where the attack bit hardest — the URL-masked linear model.

### URLs intact

| model | sev | recall@0.5 base→adv (Δ) | det@1%FPR base→adv (Δ) | PR-AUC base→adv (Δ) |
|---|---:|---|---|---|
| logreg | 0.00 | 0.990→0.990 (+0.000) | 0.990→0.990 (+0.000) | 0.992→0.993 (+0.001) |
|  | 0.25 | 0.919→0.925 (+0.006) | 0.922→0.935 (+0.013) | 0.941→0.959 (+0.018) |
|  | 0.50 | 0.935→0.981 (+0.046) | 0.938→0.984 (+0.046) | 0.954→0.978 (+0.024) |
|  | 0.75 | 0.938→0.987 (+0.049) | 0.951→0.997 (+0.046) | 0.951→0.981 (+0.030) |
|  | 1.00 | 0.948→0.994 (+0.046) | 0.955→0.994 (+0.039) | 0.952→0.986 (+0.034) |
| random_forest | 0.00 | 0.981→0.981 (+0.000) | 0.994→0.994 (+0.000) | 0.997→0.996 (−0.001) |
|  | 0.25 | 0.964→0.968 (+0.004) | 0.994→0.994 (+0.000) | 0.995→0.995 (+0.000) |
|  | 0.50 | 0.961→0.984 (+0.023) | 0.994→1.000 (+0.006) | 0.997→0.999 (+0.002) |
|  | 0.75 | 0.968→0.990 (+0.022) | 1.000→1.000 (+0.000) | 0.998→0.999 (+0.001) |
|  | 1.00 | 0.968→0.994 (+0.026) | 0.997→1.000 (+0.003) | 0.998→0.999 (+0.001) |
| gradient_boosting | 0.00 | 0.987→0.990 (+0.003) | 0.990→0.994 (+0.004) | 0.994→0.996 (+0.002) |
|  | 0.25 | 0.958→0.974 (+0.016) | 0.971→0.977 (+0.006) | 0.988→0.988 (+0.000) |
|  | 0.50 | 0.977→0.994 (+0.017) | 0.984→0.997 (+0.013) | 0.993→0.994 (+0.001) |
|  | 0.75 | 0.987→1.000 (+0.013) | 0.997→1.000 (+0.003) | 0.997→0.997 (+0.000) |
|  | 1.00 | 0.984→0.994 (+0.010) | 0.990→0.997 (+0.007) | 0.996→0.996 (+0.000) |

### URL-masked (the hard, text-only case)

| model | sev | recall@0.5 base→adv (Δ) | det@1%FPR base→adv (Δ) | PR-AUC base→adv (Δ) |
|---|---:|---|---|---|
| logreg | 0.00 | 0.987→0.990 (+0.003) | 0.987→0.990 (+0.003) | 0.991→0.992 (+0.001) |
|  | 0.25 | 0.880→0.906 (+0.026) | 0.886→0.909 (+0.023) | 0.922→0.950 (+0.028) |
|  | 0.50 | 0.919→0.968 (+0.049) | 0.922→0.968 (+0.046) | 0.946→0.978 (+0.032) |
|  | 0.75 | 0.916→0.974 (+0.058) | 0.919→0.981 (+0.062) | 0.938→0.983 (+0.045) |
|  | 1.00 | 0.886→0.990 (+0.104) | 0.890→0.994 (+0.104) | 0.931→0.987 (+0.056) |
| random_forest | 0.00 | 0.958→0.964 (+0.006) | 0.987→0.987 (+0.000) | 0.994→0.994 (+0.000) |
|  | 0.25 | 0.932→0.938 (+0.006) | 0.977→0.974 (−0.003) | 0.989→0.990 (+0.001) |
|  | 0.50 | 0.942→0.974 (+0.032) | 0.987→0.994 (+0.007) | 0.994→0.997 (+0.003) |
|  | 0.75 | 0.955→0.987 (+0.032) | 0.994→1.000 (+0.006) | 0.995→0.998 (+0.003) |
|  | 1.00 | 0.922→0.961 (+0.039) | 0.977→0.990 (+0.013) | 0.988→0.996 (+0.008) |
| gradient_boosting | 0.00 | 0.984→0.987 (+0.003) | 0.984→0.987 (+0.003) | 0.991→0.992 (+0.001) |
|  | 0.25 | 0.935→0.938 (+0.003) | 0.938→0.945 (+0.007) | 0.971→0.977 (+0.006) |
|  | 0.50 | 0.955→0.974 (+0.019) | 0.961→0.981 (+0.020) | 0.984→0.991 (+0.007) |
|  | 0.75 | 0.968→0.990 (+0.022) | 0.974→0.990 (+0.016) | 0.988→0.996 (+0.008) |
|  | 1.00 | 0.938→0.981 (+0.043) | 0.942→0.984 (+0.042) | 0.975→0.993 (+0.018) |

**Interpretation (honest read):** adversarial training **generalizes across
rewriting models** — this is a headline-grade positive finding, not Haiku-specific
overfitting. The clearest evidence is the worst-case cell: URL-masked, sev-1.0
logreg detection@1%FPR rises **0.890 → 0.994 (+0.104)** against Gemini, reaching
the same near-ceiling the Haiku-on-Haiku run hit (0.993). Two honest caveats: (i)
the *baseline* degradation under Gemini is milder than under Haiku (e.g. URL-masked
sev-1.0 logreg det@1%FPR baseline 0.890 vs Haiku 0.767), so there is less to
recover; but the adversarial model still closes essentially all of it. (ii) This is
one cross-generator pair (Haiku→Gemini); it is strong evidence of transfer, not a
proof of universality.

---

## 9. Figure & table index

| artifact | path |
|---|---|
| degradation curve (figure) | `results/figures/degradation_curve.png` |
| class balance | `results/tables/class_balance.csv` |
| clean test baseline | `results/tables/test_clean.csv` |
| 5-fold CV baseline | `results/tables/cv_baseline.csv` |
| retention (both definitions) | `results/tables/retention_definitions.csv` |
| per-severity pass set | `results/tables/degradation_per_severity_pass.csv` |
| **headline intersection (n=270)** | `results/tables/degradation_intersection.csv` |
| URL-intact vs URL-masked | `results/tables/url_ablation_degradation.csv` |
| era ablation | `results/tables/degradation_era_ablation.csv` |
| Gemini cross-model (n=308) | `results/tables/degradation_gemini.csv` |
| adversarial-training mitigation (Haiku) | `results/tables/mitigation_rescore.csv` |
| cross-generator validation (Gemini) | `results/tables/mitigation_cross_gemini.csv` |
| manual spot-check pairs | `results/tables/spotcheck_pairs.csv` |

Caveats for every number above: `LIMITATIONS.md`. Public benchmark + loader:
`release/` (`python -m src.release`). Tests: `pytest tests/` (25 passing).

## 10. Transformer detector, external validity, significance, and reverse mitigation

Every number in this section is read from the CSV named in each subheading, so the
prose matches the tables.

### A. Transformer detector (DistilBERT) — text-fragility & URL-anchoring

Fine-tuned `distilbert-base-uncased` on the SAME train split, SAME `text`
field, SAME labels; scored on the SAME strict-270 Haiku intersection and
Gemini-308 set, SAME fixed-1%-FPR operating point and 1000-seed CIs.
Inputs head-truncated to the first 256 tokens (MPS memory; signal is
concentrated early).

**recall @ 0.5 — Haiku, URLs intact (baseline detector)**

| condition | s0.00 | s0.25 | s0.50 | s0.75 | s1.00 |
|---|---|---|---|---|---|
| DistilBERT (orig) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| DistilBERT (url-masked) | 99.6% | 99.6% | 100.0% | 100.0% | 99.6% |

**detection @ 1% FPR — Haiku (baseline detector)**

| condition | s0.00 | s0.25 | s0.50 | s0.75 | s1.00 |
|---|---|---|---|---|---|
| DistilBERT (orig) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| DistilBERT (url-masked) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

**Cross-model (Gemini-308) and adversarial-trained, recall @ 0.5**

| curve | s0.00 | s0.25 | s0.50 | s0.75 | s1.00 |
|---|---|---|---|---|---|
| Gemini, baseline (orig) | 99.7% | 99.7% | 99.7% | 99.7% | 99.7% |
| Haiku, adv-trained (orig) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Haiku, adv-trained (url-masked) | 99.3% | 99.3% | 100.0% | 100.0% | 99.6% |

**Verdict — text fragility.** On this phishing corpus, DistilBERT recall@0.5 on the same 270 emails holds at 100.0% → 100.0% across severities (drop only 0.0 pts; McNemar p=1.0, see `transformer_significance.csv`), where the classical logreg drops 12.2 pts. On the Nazario phishing data, then, the transformer resisted rewriting while the bag-of-words detectors did not. We do **not** generalize this to a claim that transformers are robust to LLM rewriting, nor that fragility is specific to lexical detectors: on the same-era CEAS-2008 spam corpus the transformer degrades significantly under the same attack (§11). The transformer's resistance is therefore **corpus-dependent**, and the most we can say is that on Nazario phishing it held where the lexical detectors broke.

**Verdict — URL-anchoring.** On Nazario, under inference-time URL masking recall stays 99.6% at sev 1.0 (drop 0.0 pts): here the transformer is not URL-anchored and relies on textual/semantic signal that survives URL removal, unlike the classical logreg whose URL-masked drop is 22.2 pts (≈ doubling vs intact). This too is corpus-dependent — on CEAS-2008 the transformer's URL-masked recall drops 42.6 pts (§11), i.e. it *is* URL-anchored there.
**Era ablation:** detection_rate is a flat 1.00 across legacy_mbox, modern_year at all severities (`transformer_era.csv`) — no era confound for the transformer.


_(URL-masked here is INFERENCE-TIME input masking, not a URL-blind retrain — documented methodological difference vs. the classical pipeline.)_

### B. External validity — CEAS-2008, n = 61 (`external_validity.csv`)

Independent, **same-era (both classes 2008)** corpus, so the era confound is removed. NOTE: CEAS labels are **spam, not phishing specifically** (21,639 spam / 17,308 ham measured in-file) — a deliberate generality check, surfaced here as a caveat. Same-era stratified split, 3 classical detectors retrained, degradation + URL-masked ablation replicated; 200 positives × 4 severities rewritten with Claude Haiku.

**recall @ 0.5 by severity**

| model (condition) | s0.00 | s0.25 | s0.50 | s0.75 | s1.00 |
|---|---|---|---|---|---|
| logreg (original) | 100.0% | 100.0% | 95.1% | 73.8% | 77.0% |
| random_forest (original) | 100.0% | 98.4% | 70.5% | 27.9% | 44.3% |
| gradient_boosting (original) | 100.0% | 100.0% | 85.2% | 63.9% | 73.8% |
| logreg (url_masked) | 100.0% | 100.0% | 86.9% | 63.9% | 52.5% |
| random_forest (url_masked) | 100.0% | 98.4% | 63.9% | 13.1% | 19.7% |
| gradient_boosting (url_masked) | 100.0% | 96.7% | 88.5% | 36.1% | 63.9% |

**McNemar (original condition, det@0.5, sev0→sev1.0):** logreg 23.0pts (p=1.2e-04); random_forest 55.7pts (p=1.2e-10); gradient_boosting 26.2pts (p=3.1e-05).

**Read.** Both the degradation curve AND the URL-masking amplification reproduce on this independent corpus: every original-condition drop is McNemar-significant, and URL masking enlarges the sev-1.0 drop further (for logreg the drop roughly doubles, 0.23→0.48). The finding is not a quirk of the primary dataset or of phishing-vs-spam labeling.

### C. Paired significance — McNemar exact (`significance_paired.csv`)

Paired per-email test (b=lost, c=gained; binomtest two-sided) on the strict-270 set, sev 0.0 vs 1.0.

| condition | model | drop (det@0.5) | 95% CI | McNemar exact p |
|---|---|---|---|---|
| original | logreg | 12.2 pts | [8.1, 16.3] | 2.10e-09 |
| original | random_forest | 5.2 pts | [1.9, 8.5] | 4.34e-03 |
| original | gradient_boosting | 1.1 pts | [-1.5, 3.7] | 5.81e-01 (NS) |
| url_masked | logreg | 22.2 pts | [17.0, 27.4] | 2.73e-17 |
| url_masked | random_forest | 9.3 pts | [5.2, 13.3] | 1.12e-04 |
| url_masked | gradient_boosting | 5.2 pts | [1.5, 8.5] | 9.36e-03 |

**Honest note:** the only non-significant cell is gradient_boosting / URLs-intact (drop 1.1 pts, p≈0.58) — GB barely degrades when URLs are present, so its tiny drop is within noise. Every other cell, and all URL-masked cells, are highly significant.

### D. Reverse mitigation — train on Gemini, test on Haiku (`mitigation_cross_haiku.csv`)

Augment the train set with 1,000 **Gemini** TRAIN-phish rewrites (severities 0.5/1.0), retrain the 3 classical detectors, and re-test against the canonical **Haiku** strict-270 rewrites under both URL conditions. Tests generator-agnosticism of the adversarial-training fix.

**detection @ 1% FPR by severity (baseline vs Gemini-augmented)**

| cell | s0.00 | s0.25 | s0.50 | s0.75 | s1.00 |
|---|---|---|---|---|---|
| logreg original base | 98.9% | 93.3% | 92.2% | 88.1% | 88.1% |
| logreg original adv(gem) | 98.9% | 96.3% | 98.9% | 99.6% | 98.9% |
| logreg url_masked base | 98.5% | 91.1% | 87.4% | 81.5% | 76.7% |
| logreg url_masked adv(gem) | 98.9% | 95.2% | 98.9% | 97.8% | 98.1% |
| random_forest url_masked adv(gem) | 98.1% | 97.8% | 98.9% | 98.9% | 99.6% |
| gradient_boosting url_masked adv(gem) | 98.5% | 95.9% | 98.1% | 96.7% | 97.0% |

**McNemar:** 5/6 baseline cells show a significant sev0→sev1.0 drop; after Gemini-augmented training only 0/6 remain significant.

**Read.** Training on rewrites from one LLM family (Gemini) neutralizes evasion crafted by another (Haiku): every baseline degradation collapses to non-significant after augmentation. The mitigation is **generator-agnostic**.

## 11. CEAS-2008 transformer replication

DistilBERT fine-tuned on the **CEAS-2008 train split** (same frugal config, seed 42,
class-weighted CE, 3 epochs) and scored on the **existing cached CEAS Haiku rewrites**
(no new API calls), through the identical pipeline as the primary transformer run:
clean baseline, degradation at all severities, URL-masked (inference-time masking),
recall@0.5 / det@1%FPR / PR-AUC with 1000-seed bootstrap CIs and McNemar exact
(sev0 vs sev1). Intersection **n=61** (positives with URLs retained at all of
{0.25,0.5,0.75,1.0}). CEAS positives are generic spam (spam-vs-ham external check).

Clean test recall@0.5 = 99.7%.

**recall@0.5 by severity (Haiku, intersection n=61)**

| condition | s0.00 | s0.25 | s0.50 | s0.75 | s1.00 | McNemar p (sev0→1) |
|---|---|---|---|---|---|---|
| original | 100.0% | 100.0% | 95.1% | 90.2% | 85.2% | 0.0039 |
| url_masked | 100.0% | 95.1% | 80.3% | 88.5% | 57.4% | 3e-08 |

**Verdict.**
- Text-robustness: **degrades** under rewriting (-14.8 pts, p=0.0039) — unlike the Nazario transformer, so the robustness was corpus-specific.
- URL-anchoring: **URL-anchored** here (url-masked drop -42.6 pts, p=3e-08) — masking removes the signal, unlike Nazario.
- Reproduction: The CEAS result diverges from Nazario (see above), so the transformer behaviour is at least partly corpus-dependent.
