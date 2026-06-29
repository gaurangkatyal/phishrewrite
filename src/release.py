"""Public release packaging.

Builds a defanged, documented, Hugging-Face-loadable benchmark export under
`release/` and stages a Zenodo-ready archive under `dist/`. The internal
experiment keeps verbatim URLs in `data/processed`; ONLY the release is defanged.

Defang is **scheme-only and losslessly reversible** (`http://`->`hxxp://`,
`https://`->`hxxps://`, leading `www.`->`www[.]`), so re-fanging the released
text and recomputing features reproduces the internal `hc_url_count` exactly —
no need to ship feature matrices. `refang()` is the inverse; a standalone
`release/refang.py` ships with the data.

Usage:
    python -m src.release            # build release/ + dist/ archive
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd

from . import attack, config, features

RELEASE_DIR: Path = config.ROOT / "release"
DIST_DIR: Path = config.ROOT / "dist"
VERSION: str = "1.0"
ARCHIVE_STEM: str = f"phishrewrite-benchmark-v{VERSION}"

# Defanged-URL regex (inverse of features.URL_RE), for refang().
DEFANGED_URL_RE = re.compile(r"hxxps?://\S+|www\[\.\]\S+", re.IGNORECASE)


# Lossless scheme-only defang / refang
def _defang_token(u: str) -> str:
    u = re.sub(r"^https://", "hxxps://", u, flags=re.IGNORECASE)
    u = re.sub(r"^http://", "hxxp://", u, flags=re.IGNORECASE)
    if u[:4].lower() == "www.":
        u = "www[.]" + u[4:]
    return u


def _refang_token(u: str) -> str:
    u = re.sub(r"^hxxps://", "https://", u, flags=re.IGNORECASE)
    u = re.sub(r"^hxxp://", "http://", u, flags=re.IGNORECASE)
    if u[:6].lower() == "www[.]":
        u = "www." + u[6:]
    return u


def defang(text: str) -> str:
    """Defang every URL in `text`. Reversible via refang().

    The URL_RE token pass handles the leading scheme and a bare `www.` host. The
    trailing scheme sweep also neutralises schemes nested in redirector query
    strings (e.g. google `/url?q=http://...`) that the per-token, leading-anchored
    rewrite would otherwise miss — so no live `http(s)://` ever survives.
    """
    text = features.URL_RE.sub(lambda m: _defang_token(m.group(0)), text or "")
    return text.replace("https://", "hxxps://").replace("http://", "hxxp://")


def refang(text: str) -> str:
    """Inverse of defang(): restore verbatim, clickable URLs."""
    text = DEFANGED_URL_RE.sub(lambda m: _refang_token(m.group(0)), text or "")
    return text.replace("hxxps://", "https://").replace("hxxp://", "http://")


# Build
def _build_emails() -> pd.DataFrame:
    """Defanged labeled email table (verbatim text columns -> defanged)."""
    df = pd.read_csv(config.DATASET_CSV)
    for col in ("text", "subject", "body"):
        df[col] = df[col].fillna("").astype(str).map(defang)
    cols = ["id", "label", "source", "split", "original_id", "had_html", "subject", "body", "text"]
    return df[cols]


def _build_rewrites() -> pd.DataFrame:
    """Defanged rewrites from every available model file, with BOTH retention
    definitions (strict all-URL + primary-URL) recomputed and recorded."""
    orig_text = {r["original_id"]: r["text"] for _, r in pd.read_csv(config.DATASET_CSV).iterrows()}
    files = [config.REWRITES_JSONL, config.PROCESSED_DIR / "rewrites_gemini-2-5-flash.jsonl"]
    rows: list[dict] = []
    for path in files:
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            ot = orig_text.get(r["original_id"], "")
            prim = attack.check_retention_primary(ot, r.get("rewrite_text") or "")
            rows.append(
                {
                    "original_id": r["original_id"],
                    "source": r["source"],
                    "severity": r["severity"],
                    "provider": r["provider"],
                    "model": r["model"],
                    "temperature": r["temperature"],
                    "refused": bool(r.get("refused", False)),
                    "n_orig_urls": r.get("n_orig_urls"),
                    "retained_urls_strict": r.get("retained_urls"),
                    "retained_urls_primary": prim["retained_primary"],
                    "n_primary_urls": prim["n_primary_urls"],
                    "rewrite_subject": defang(r.get("rewrite_subject") or ""),
                    "rewrite_body": defang(r.get("rewrite_body") or ""),
                    "rewrite_text": defang(r.get("rewrite_text") or ""),
                }
            )
    return pd.DataFrame(rows)


_CARD = """---
license: other
license_name: phishrewrite-data-license
license_link: LICENSE
language:
- en
task_categories:
- text-classification
tags:
- phishing
- email
- adversarial-robustness
- security
pretty_name: PhishRewrite
configs:
- config_name: emails
  data_files: data/emails.csv
- config_name: rewrites
  data_files: data/rewrites.csv
---

# PhishRewrite benchmark v{version}

A benchmark measuring how content-based phishing-email detectors degrade when
phishing text is rewritten by a large language model to read benign **while
preserving the malicious ask and every URL**. Defensive research only — read
`ETHICS` before use.

> **All URLs in this release are defanged** (`http://`->`hxxp://`,
> `https://`->`hxxps://`, `www.`->`www[.]`). The transform is scheme-only and
> **losslessly reversible**: run `refang.py` (shipped here) to restore verbatim
> URLs before recomputing URL features. Do not visit any link.

## Configs

| config | rows | description |
|---|---|---|
| `emails` | {n_emails} | labeled phishing/ham emails with a fixed train/test split |
| `rewrites` | {n_rewrites} | LLM rewrites of held-out **test** phishing at 4 severities |

```python
from datasets import load_dataset
emails   = load_dataset("phishrewrite", "emails")     # or data_files=...
rewrites = load_dataset("phishrewrite", "rewrites")
```

## `emails` schema

| column | type | meaning |
|---|---|---|
| `id` | string | stable unique id within the table |
| `label` | int | 1 = phishing, 0 = ham |
| `source` | string | `nazario` (phishing) / `spamassassin`, `enron` (ham) |
| `split` | string | `train` or `test` (stratified, seed={seed}) |
| `original_id` | string | source-local id (traces back to the raw message) |
| `had_html` | bool | did the raw body contain HTML before stripping? |
| `subject` | string | parsed subject (defanged) |
| `body` | string | HTML-stripped plain-text body (defanged) |
| `text` | string | `subject` + `body` — the model/feature input (defanged) |

## `rewrites` schema

| column | type | meaning |
|---|---|---|
| `original_id` | string | the test phishing email this rewrites (join to `emails`) |
| `source` | string | always `nazario` |
| `severity` | float | rewrite aggressiveness: 0.25 / 0.5 / 0.75 / 1.0 |
| `provider`, `model` | string | rewriting model (claude-haiku-4-5 or gemini-2.5-flash) |
| `temperature` | float | sampling temperature |
| `refused` | bool | model returned meta-text instead of an email |
| `n_orig_urls` | int | URLs in the original email |
| `retained_urls_strict` | bool/null | strict: ALL original URLs preserved (null = no URL) |
| `retained_urls_primary` | bool/null | relaxed: ask-bearing URL(s) kept (null = no anchor) |
| `n_primary_urls` | int | number of ask-bearing URLs identified in the original |
| `rewrite_subject`, `rewrite_body`, `rewrite_text` | string | the rewrite (defanged) |

The two retention columns are the strict and primary label-validity definitions
documented in `LIMITATIONS`. The headline degradation set uses the strict
definition.

## Provenance

See `PROVENANCE.md`. Sources: the Nazario phishing corpus (phishing), the
SpamAssassin public corpus (ham subsets only), and the Enron email corpus
(sampled ham). SpamAssassin spam is excluded by design. Seed = {seed}.

## Citation

```bibtex
@misc{{phishrewrite{version_nodot},
  title        = {{PhishRewrite: Measuring Phishing-Detector Degradation under LLM Rewriting}},
  author       = {{Katyal, Gaurang}},
  year         = {{2026}},
  publisher    = {{Zenodo}},
  doi          = {{10.5281/zenodo.21018700}},
  url          = {{https://doi.org/10.5281/zenodo.21018700}},
  note         = {{Version {version}. Defensive-research benchmark.}}
}}
```

Please also cite the underlying corpora (Nazario; SpamAssassin; Klimt & Yang,
the Enron corpus) — see `PROVENANCE.md` for details.

## License

Code: MIT. Data / derived benchmark: see `LICENSE` (`DATA_LICENSE` in the source
repo). Defensive use only.
"""


_PROVENANCE = """# Provenance & datasheet — PhishRewrite v{version}

## Composition

- **{n_phish} phishing** emails (label 1), source: **Nazario phishing corpus**.
- **{n_ham} ham** emails (label 0), sources: **SpamAssassin** (`easy_ham`,
  `easy_ham_2`, `hard_ham` — ham subsets ONLY; spam subsets excluded by design)
  and **Enron** (a seeded sample for legitimate-email diversity).
- Natural class ratio ~1:4.2 phishing:ham; left unbalanced (handled in-model).
- Train/test split: stratified, seed={seed}, test_size={test_size}.

## Rewrites

- LLM rewrites of held-out **test** phishing only (ham is never attacked).
- Severities 0.25 / 0.5 / 0.75 / 1.0 (cumulative aggressiveness); the system +
  per-severity prompts ship in `prompts/`.
- Label-validity contract: preserve the malicious ask and reproduce every URL verbatim.
- Models: Claude Haiku 4.5 (headline) and Gemini 2.5 Flash (cross-model check).
- `{n_rewrites}` rewrite rows total.

## Transform applied for release

- URLs defanged scheme-only and **losslessly**: `http://`->`hxxp://`,
  `https://`->`hxxps://`, `www.`->`www[.]`. Reverse with `refang.py`.
- No other modification to the text. Features in the source repo were computed
  pre-defang on verbatim URLs; re-fang before recomputing URL-based features.

## Source terms

- Nazario corpus: research use; cite the corpus. URLs defanged on release.
- SpamAssassin public corpus: distributed for anti-spam research.
- Enron corpus (Klimt & Yang, 2004): public.

See `LICENSE` for the combined data-license terms and restrictions.

## Ethics

Defensive-use-only. This benchmark exists to quantify and help close a weakness
in content-based detectors. Do not use it to build or improve phishing attacks.
"""


_REFANG_SCRIPT = '''"""Restore verbatim URLs in the PhishRewrite release (inverse of the
scheme-only defang applied for distribution).

    python refang.py emails.csv emails_refanged.csv

Defang here is lossless, so re-fanging then recomputing URL features reproduces
the original feature values. Re-fanged links are live — do not visit them.
"""
import re
import sys
import pandas as pd

DEFANGED_URL_RE = re.compile(r"hxxps?://\\S+|www\\[\\.\\]\\S+", re.IGNORECASE)


def _refang_token(u: str) -> str:
    u = re.sub(r"^hxxps://", "https://", u, flags=re.IGNORECASE)
    u = re.sub(r"^hxxp://", "http://", u, flags=re.IGNORECASE)
    if u[:6].lower() == "www[.]":
        u = "www." + u[6:]
    return u


def refang(text: str) -> str:
    text = DEFANGED_URL_RE.sub(lambda m: _refang_token(m.group(0)), str(text or ""))
    return text.replace("hxxps://", "https://").replace("hxxp://", "http://")


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    df = pd.read_csv(src)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(refang)
    df.to_csv(dst, index=False)
    print(f"refanged {src} -> {dst}")
'''


def build() -> Path:
    config.ensure_dirs()
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    (RELEASE_DIR / "data").mkdir(parents=True)
    (RELEASE_DIR / "prompts").mkdir(parents=True)

    emails = _build_emails()
    rewrites = _build_rewrites()
    emails.to_csv(RELEASE_DIR / "data" / "emails.csv", index=False)
    rewrites.to_csv(RELEASE_DIR / "data" / "rewrites.csv", index=False)

    n_phish = int((emails["label"] == config.LABEL_PHISHING).sum())
    n_ham = int((emails["label"] == config.LABEL_HAM).sum())
    fmt = dict(
        version=VERSION,
        version_nodot=VERSION.replace(".", ""),
        seed=config.SEED,
        test_size=config.TEST_SIZE,
        n_emails=len(emails),
        n_rewrites=len(rewrites),
        n_phish=n_phish,
        n_ham=n_ham,
    )
    (RELEASE_DIR / "README.md").write_text(_CARD.format(**fmt), encoding="utf-8")
    (RELEASE_DIR / "PROVENANCE.md").write_text(_PROVENANCE.format(**fmt), encoding="utf-8")
    (RELEASE_DIR / "refang.py").write_text(_REFANG_SCRIPT, encoding="utf-8")

    # Ship license, ethics, limitations, and the prompts (part of the benchmark).
    shutil.copy(config.ROOT / "DATA_LICENSE", RELEASE_DIR / "LICENSE")
    for name in ("ETHICS.md", "LIMITATIONS.md"):
        if (config.ROOT / name).exists():
            shutil.copy(config.ROOT / name, RELEASE_DIR / name)
    for p in sorted(config.PROMPTS_DIR.glob("*.txt")):
        shutil.copy(p, RELEASE_DIR / "prompts" / p.name)

    # Stage a Zenodo-ready archive (do not upload).
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(str(DIST_DIR / ARCHIVE_STEM), "zip", root_dir=RELEASE_DIR)
    print(f"  emails.csv   : {len(emails)} rows")
    print(f"  rewrites.csv : {len(rewrites)} rows " f"({rewrites['model'].nunique()} models)")
    print(f"  release dir  : {RELEASE_DIR}")
    print(f"  archive      : {archive}")
    return Path(archive)


if __name__ == "__main__":
    build()
