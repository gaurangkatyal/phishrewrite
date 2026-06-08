# PhishRewrite — Implementation Plan

Design document recording the methodology decisions. The implemented pipeline
follows this plan; results are in [`RESULTS.md`](RESULTS.md). Heavy/gated stages
prompt before running: the dataset downloads and the API rewrites.

## Confirmed decisions
- **Corpus:** named research datasets only — SpamAssassin (**ham only**), Nazario (phishing), Enron (additional ham). Not the pre-combined Kaggle set. Phishing = positive class (label 1).
- **Schema:** `id, text (subject+body), subject, body, label (1=phish,0=ham), source (spamassassin|nazario|enron), split (train|test), original_id, had_html`. HTML stripped to plain text in cleaning.
- **Rewrites:** phishing TEST emails × 4 non-zero severities; ham never attacked. Cap via `MAX_PHISH_SAMPLE` (default 500 → 2,000 calls). Report projected call count before running.
- **Attack model:** default Claude Haiku 4.5 + batch + prompt caching; provider/model abstracted in config (OpenAI/local swappable). First run = Haiku only.
- **License:** MIT for code; separate `DATA_LICENSE` for redistributed processed data.
- **Benchmark release (decided):** the rewrites ARE the benchmark — release them **defanged** via **Zenodo** (mirrors Paper 1). Ethics framing supports it (derived from public corpora, no operational uplift). Drives `DATA_LICENSE` + `ETHICS.md`.

---

## Methodology decisions

### Per-source label mapping
SpamAssassin contains both ham and spam; generic spam ≠ phishing. Mapping is explicit and enforced in code:

| Source | Subsets used | Label | Notes |
|---|---|---|---|
| SpamAssassin | `easy_ham`, `easy_ham_2`, `hard_ham` | **0 (ham)** | **`spam`/`spam_2` are EXCLUDED entirely** — not relabeled as phishing. Avoids label noise. |
| Nazario | all phishing mboxes | **1 (phishing)** | sole phishing source |
| Enron | sampled ham | **0 (ham)** | additional ham diversity |

A unit test asserts no SpamAssassin-spam message reaches the processed table and that `source`↔`label` invariants hold.

### Severity design: full replacement, aggressiveness in the prompt; intersection set for the headline
Severity does **not** mean "fraction of phish replaced." At severity *s*, **every** sampled phishing test email is replaced by **its own severity-*s* rewrite**; ham is untouched. The attacked test set at level *s* = {sampled phish → their s-rewrites} ∪ {clean ham}. s=0.0 = clean anchor. The headline curve is metric vs. rewrite *aggressiveness*, with sample composition held constant so only rewrite intensity varies.

**Composition-confound fix (resolves the severity-design × retention interaction):** retention-drop rates differ by severity (highest at 1.0), so naively dropping failures per-level would change N per severity and re-introduce a composition confound. Therefore the **headline degradation curve is computed on the intersection set** — the phishing emails whose rewrites pass the label-validity retention check at **all four** non-zero severities (plus their clean originals for s=0.0). Same N at every point; only intensity varies. Per-severity retention rates are reported **separately** as a finding in their own right ("at full-rewrite severity, X% of attacks fail to preserve the ask").

### Severity as named, cumulative prompt instructions
Each level maps to concrete instructions; levels are additive. Exact prompt text is committed to the repo (`src/prompts/`) as part of the benchmark:
- **0.25 — light lexical paraphrase:** synonym swaps, minor wording; structure/tone unchanged.
- **0.50 — + sentence restructuring:** reorder/merge/split sentences; reword openings.
- **0.75 — + tone/register shift:** make polite/professional, remove urgency and pressure cues.
- **1.00 — full rewrite preserving only the ask:** rewrite freely; keep solely the call-to-action and URLs.
All levels carry the label-validity invariant. A `severity_prompts.md` documenting the verbatim prompts ships in the repo.

### Label-validity safeguards (the #1 reviewer question)
To ensure a high-severity rewrite is still phishing (not a defanged-into-benign email):
- **Prompt invariant:** every prompt explicitly instructs the model to **preserve the malicious call-to-action and all URLs verbatim** (credential request, link-click, payment, etc.). The "softening" applies to tone/urgency, never to the ask.
- **Automated post-check** per rewrite: verify the CTA survives (URL set preserved verbatim; ask-intent heuristic — link/credential/payment cue present). **Report retention rate per severity** → `results/tables/rewrite_retention.csv`; **drop failures** and log them. The **headline curve uses the intersection set** (emails passing retention at all four severities); per-severity retention is reported separately.
- **Manual spot-check:** export ~30 random rewrites (stratified by severity) to `results/spotcheck_sample.csv` to eyeball before finalizing the headline numbers.

### URL handling, defined and feature-safe
- **During the experiment:** URLs are **preserved verbatim** through rewriting (enforced by the label-validity check). The handcrafted `url_count` feature is therefore computed on identical URLs across clean vs. rewritten — degradation cannot be a trivial artifact of dropped URLs.
- **For public release:** URLs are **defanged (`hxxp://`)** consistently in **both** originals and rewrites, so the released benchmark carries no live phishing links while keeping URL features comparable. Defang is a release-time transform only; internal experiment uses verbatim URLs.

### Per-source / era ablation (mandatory)
Enron ham is 2001–02; Nazario phishing is later. To preempt the "detector learned era/source, not phishing-ness" rebuttal:
- Report degradation broken down per phishing source/era.
- Run a source-artifact probe (e.g., can a classifier separate ham sources by era? how much does that overlap with the phishing signal?) and a sensitivity check (e.g., restricting ham to reduce era skew).
- Output → `results/tables/ablation_source_era.csv` + figure. Cheap; not optional.

### Bootstrap CIs on test-set metrics
CV mean±std only covers train. Add bootstrap resampling of the test set (e.g., 1,000 resamples, seeded) → 95% CIs per metric **per severity**, written alongside the degradation table.

---

## Build order

### Scaffolding (no network)
Tree + `LICENSE` (MIT), `DATA_LICENSE`, `ETHICS.md`, `.gitignore` (ignores `data/`, `.env`, `__pycache__`), `.env.example` (`ANTHROPIC_API_KEY=`), pinned `requirements.txt`
(`scikit-learn, pandas, numpy, scipy, matplotlib, beautifulsoup4, lxml, anthropic, python-dotenv, tqdm, pyyaml, pytest`).

### `src/config.py`
`SEED=42` (split, CV, sampling, bootstrap, model `random_state`); paths, dataset URLs, `TEST_SIZE=0.2`, `CV_FOLDS=5`, `SEVERITIES=[0.0,0.25,0.5,0.75,1.0]`, `N_BOOTSTRAP=1000`; attack block (`PROVIDER`, `MODEL="claude-haiku-4-5"`, `USE_BATCH`, `USE_CACHE`, `MAX_PHISH_SAMPLE=500`, `MAX_TOKENS`, temperature) with provider abstraction; release/defang flags.

### `src/data_prep.py` (heavy: downloads, gated)
Download to `data/raw/` (untouched): SpamAssassin, Nazario `.mbox`, Enron (~1.4 GB; need ham only — download once then sample N ham, default 20k; report raw size before pulling). Apply the **per-source label mapping**. Clean (HTML→text, set `had_html`, normalize whitespace, drop empties + exact/near-dup bodies) → unified schema → `data/processed/dataset.csv`. Stratified 80/20 split (seeded). Report class balance per source + overall → `results/tables/class_balance.csv`; wait on ham-subsampling decision.

### `src/features.py`
TF-IDF (fit on train only, transform test — no leakage) + handcrafted (length, `url_count`, link count, urgency-word count, `had_html`, digit ratio, ALL-CAPS ratio, exclamations). Persist fitted vectorizer + feature names.

### `src/detectors.py`
LogisticRegression, RandomForest, GradientBoosting (`random_state=SEED`). 5-fold stratified CV on train → mean ± std for ROC-AUC, F1, precision, recall → `results/tables/cv_baseline.csv`. Refit on full train; persist models.

### `src/evaluate.py` (clean baseline)
Score persisted models on clean test set → `results/tables/test_clean.csv` (s=0.0 anchor).

### `src/attack.py` (heavy: API cost, gated)
Phishing rows where `split==test`; sample up to `MAX_PHISH_SAMPLE` (seeded). Print projected calls (sample×4) + estimated cost, then stop for approval. Apply the **severity prompts** with the **preservation invariant**; static instructions in cached prefix; batched submission. Run the **automated retention post-check** (drop+log failures, write retention table). Cache each rewrite to `data/processed/rewrites.jsonl` keyed by `(original_id, severity)` — idempotent, resumable. Emit the **spot-check sample**.

### `src/evaluate.py` (degradation, finalize)
Per severity, build the attacked test set per the **severity design** (all sampled phish → their s-rewrites + clean ham), re-extract with the saved vectorizer, re-score detectors. Outputs: `results/tables/degradation.csv` (+ **bootstrap CIs**), `results/figures/degradation_curve.png`, calibration plot, and the **mandatory** `results/tables/ablation_source_era.csv` + figure.

### Release export + `tests/` + `README.md`
- **Release export:** defang URLs in originals + rewrites (**URL handling**), package the benchmark for **Zenodo**; document schema + provenance.
- Smoke tests: label-mapping invariants, schema/dtypes, no train/test leakage, feature shapes, retention logic, attack I/O on a mocked 2-email stub (no network).
- README: exact run steps, seed/version notes, provider-swap instructions, prominent `ETHICS.md` link, Zenodo/citation pointer. **Defang reproducibility note:** features were computed pre-defang on verbatim URLs; the released (defanged) data yields different `url_count` under naive regexes — so we ship the computed feature matrices and a re-fang note so reproducers don't report a false discrepancy.

## Ethics & Safety
Defensive-research artifact; repo carries a prominent intended-use statement (`ETHICS.md`, linked from README header):
- **Purpose-limited generation.** Rewrites exist only to measure detector robustness (the degradation curve) — a measurement instrument, not deployable evasive content.
- **Defensive framing / release.** Released (defanged, via Zenodo) for defensive research and detector hardening.
- **Intended-use statement.** Permitted: academic study, detector evaluation/hardening, reproducibility. Prohibited: sending email, real phishing/social-engineering, deployment against people or live systems. Explicit "do not use to attack" notice.
- **No operational uplift.** Derived from public corpora; URLs defanged in the release; no payloads, real targets, or live infrastructure.
- **Ethical data handling.** Public named research datasets only; no live scraping, no email sending, no PII. `DATA_LICENSE` documents upstream terms.
- **Responsible release posture.** README leads with defensive motivation; results emphasize the gap to fix.

## Reproducibility
One seed used everywhere, pinned deps, vectorizer fit on train only, on-disk rewrite cache, committed prompts, all outputs as committed CSVs.

## Decisions surfaced, not auto-made
1. Final class balance + whether to subsample ham (data preparation).
2. Projected attack call count + cost (the rewrite attack).
3. Manual spot-check sign-off on ~30 rewrites before headline numbers are finalized.
