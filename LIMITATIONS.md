# Limitations

These are the known threats to validity for the PhishRewrite benchmark. They are
stated up front so results are read with the right caveats. None of them changes
the qualitative headline (content-based detectors degrade under LLM rewriting);
several mean the reported degradation is, if anything, an **under**-estimate.

## 1. The retention checker is conservative (biased toward FAIL)

The label-validity contract requires a rewrite to preserve the malicious ask
*and its URLs*. The headline checker (`src/attack.py: check_retention`) enforces
the **strict** form: a rewrite PASSes only if **every** URL in the original
survives (`len(missing_urls) == 0`). Dropping even one URL marks it FAIL.

Spot-checking shows many FAILs drop only **incidental** URLs — email-client /
antivirus signature links, social-media and unsubscribe/mailing-list footers,
tracking/redirect infrastructure (DoubleClick, SendGrid, `click*.ebay`
trackers, Yahoo redirects), corpus artifacts (`monkey.org`, the Nazario host
itself), and brand/regulator footer links (paypal.com/privacy, ncua.gov
navigation, ebay help/policy pages) — **while preserving the actual
credential-harvest ("primary") URL**. Such rewrites are still valid phishing for
the benchmark, but the strict checker discards them.

We quantified this with a **primary-URL retention** variant
(`src/attack.py: check_retention_primary` / `primary_urls`), which identifies the
ask-bearing URL(s) per email (raw-IP hosts, credential/ask path keywords,
lookalike vs. known-brand hosts) and requires only those to survive. On the
2,000-record Haiku set the headline retention **intersection grows 270 → 331
(+61)** and per-severity FAILs roughly a third as large (e.g. severity 1.0:
89 → 31). See `results/tables/retention_definitions.csv`.

**Consequence:** the strict headline degradation set is a **lower bound on attack
success** — it discards genuinely-valid adversarial rewrites, so reported
detector robustness is optimistic, not pessimistic.

### 1a. Selection effect on the era ablation

The strict checker does not discard emails uniformly across eras. Legacy
Nazario `*.mbox` phishing (the eBay/PayPal era) carries many brand-footer URLs,
so it is over-failed by the all-URLs rule. The primary-URL definition recovers
the legacy bucket far more than the modern one (**legacy 128 → 182, +54** vs.
**modern 142 → 149, +7**). The era ablation is therefore computed on a
non-random subsample that under-represents successful legacy-era evasions; read
the era split as indicative, not as an unbiased per-era success rate.

## 2. ROC-AUC is insensitive under this class imbalance

The test set is ~1:4.4 phishing:ham. Under imbalance, ROC-AUC is dominated by
the (large, **severity-invariant**) ham class and barely moves even as detection
collapses. Concretely, logistic regression's recall@0.5 falls **0.989 → 0.867**
across severity while its ROC-AUC stays **0.999 → 0.996** — essentially flat.

ROC-AUC would make the attack look ineffective. The honest metrics, and the ones
we report, are:

- **recall @ 0.5** = the phishing detection rate (fraction of phish flagged);
- **PR-AUC / average precision**, which is sensitive to the positive class under
  imbalance;
- **detection @ 1% FPR** (`src/ablation.py`), an operating point set on the ham
  score distribution (which does not change with severity), so the threshold is
  fixed and the detection drop is directly comparable across severities.

ROC-AUC is reported for completeness only; it should not be used to judge the
attack.

## 3. English-only, single-language corpus

All sources (Nazario phishing; SpamAssassin and Enron ham) are predominantly
English. The TF-IDF features, the urgency lexicon, and the LLM rewrites are all
English. Nothing here measures non-English phishing or cross-lingual rewriting,
and the absolute numbers should not be assumed to transfer to other languages.

## 4. The URL-frozen contract is a deliberate lower bound on evasion

By construction the attack **freezes all URLs verbatim** and rewrites only the
surrounding natural language. This isolates the *text* signal and keeps the label
valid (the malicious destination is unchanged), but it hands the defender a
strong, unperturbed feature: the URL itself. A real adversary would also mutate
URLs (new domains, redirectors, shorteners, homoglyphs). The URL-ablation
(`src/ablation.py`, URL-blind retrain) shows that once that frozen signal is
removed, degradation roughly doubles (e.g. logreg detection@1%FPR drop deepens
from −0.107 to −0.219). So the headline (URL-intact) curve **understates** what a
URL-mutating attacker would achieve; the URL-masked curve is the more realistic
text-only bound.

## 5. Other scope notes

- **Single phishing source.** Phishing positives come only from the Nazario
  corpus; ham comes from SpamAssassin (ham subsets only) and Enron. Source-linked
  artifacts can correlate with the label; the era ablation is the partial check.
- **One rewriting model per headline.** The headline attack uses Claude Haiku;
  a Gemini 2.5 Flash run (`results/tables/degradation_gemini.csv`) confirms the
  effect is not model-specific, but two models is not an exhaustive survey.
- **Defang on release.** The public release defangs URLs (`http:` → `hxxp:`);
  features were computed pre-defang on verbatim URLs. Recomputing features from
  the released text with a naive URL regex will not reproduce `hc_url_count`; use
  the shipped feature matrices or re-fang first (see `README.md`).
