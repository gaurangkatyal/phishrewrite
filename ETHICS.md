# Ethics & Intended Use — PhishRewrite

**PhishRewrite is a defensive security research artifact.** It measures how
content-based phishing-email detectors degrade when phishing text is rewritten by
a large language model. Its purpose is to expose and quantify a known weakness so
that detector builders can close it.

## Intended use

**Permitted:**
- Academic study of detector robustness and LLM-driven evasion.
- Evaluation, benchmarking, and hardening of phishing/abuse detectors.
- Reproduction and extension of the published results.

**Prohibited:**
- Sending email of any kind, or conducting real phishing / social-engineering.
- Deploying the rewrites (or derived models) against people, mailboxes, or live
  systems.
- Any use intended to evade detection in the wild or to cause harm.

> **Do not use this project to attack anyone.** The rewrites are a measurement
> instrument, not a product, and not a deployable attack toolkit.

## Why generating the rewrites is justified

- **Purpose-limited.** Rewrites are produced only to compute the detector
  degradation curve. They are tied to already-public phishing corpora.
- **No operational uplift.** We add no working links, payloads, real targets, or
  live infrastructure. URLs in the public release are **defanged** (`hxxp://`).
  The contribution is the evaluation method and findings, not usable attacks.
- **Label validity is enforced.** Rewrites must preserve the malicious
  call-to-action; rewrites that fail this check are dropped and reported. This
  keeps the study honest (we measure evasion, not emails that stopped being
  attacks) — it does not increase real-world harm.

## Data ethics

- Public, named research datasets only (SpamAssassin, Nazario, Enron). See
  `DATA_LICENSE`.
- No live email scraping, no email sending, no collection of personal data.
- Defanged release; takedown process available (see `DATA_LICENSE`).

## Responsible release

The released benchmark (defanged) is published for defensive research via Zenodo.
The README leads with the defensive motivation and frames results as a gap for
detector builders to close.
