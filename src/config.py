"""Central configuration for PhishRewrite — the single source of truth.

Every other module imports paths, the seed, dataset definitions, the severity
grid, and attack settings from here. Nothing else should hard-code these values.

Secrets are never stored here; API keys come from the environment (.env).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import dotenv_values, load_dotenv

    load_dotenv()  # populate os.environ from a local .env if present
    # load_dotenv() won't overwrite a variable already in the environment, and a
    # shell often exports an EMPTY placeholder (e.g. ANTHROPIC_API_KEY="") which
    # would then shadow the real key in .env. Backfill any missing-or-empty var
    # from .env so the file wins over an empty placeholder, while a genuinely set
    # shell value still takes precedence.
    for _k, _v in dotenv_values().items():
        if _v and not os.environ.get(_k):
            os.environ[_k] = _v
except ModuleNotFoundError:  # allow importing config before deps are installed

    def load_dotenv(*_args, **_kwargs) -> bool:  # type: ignore[misc]
        return False


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
# One seed governs the train/test split, cross-validation shuffling, phishing
# sampling for the attack, bootstrap resampling, and every model's random_state.
SEED: int = 42

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = ROOT / "results"
TABLES_DIR: Path = RESULTS_DIR / "tables"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
MODELS_DIR: Path = RESULTS_DIR / "models"
PROMPTS_DIR: Path = ROOT / "src" / "prompts"

# Canonical processed artifacts.
DATASET_CSV: Path = PROCESSED_DIR / "dataset.csv"
REWRITES_JSONL: Path = PROCESSED_DIR / "rewrites.jsonl"
VECTORIZER_PATH: Path = MODELS_DIR / "tfidf_vectorizer.joblib"


def ensure_dirs() -> None:
    """Create all output directories. Safe to call repeatedly."""
    for d in (RAW_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Unified schema
# --------------------------------------------------------------------------- #
# Order is authoritative; data_prep writes exactly these columns.
SCHEMA_COLUMNS: tuple[str, ...] = (
    "id",  # stable unique id within the processed table
    "text",  # subject + body concatenated (the model/feature input)
    "subject",  # parsed subject (may be empty)
    "body",  # parsed, HTML-stripped plain-text body
    "label",  # 1 = phishing (positive), 0 = ham
    "source",  # one of SOURCES keys
    "split",  # "train" | "test"
    "original_id",  # source-local id, traces a row back to its raw message
    "had_html",  # bool: did the raw body contain HTML before stripping?
)

LABEL_PHISHING: int = 1
LABEL_HAM: int = 0

# --------------------------------------------------------------------------- #
# Datasets (public, named research corpora)
# --------------------------------------------------------------------------- #
# IMPORTANT label policy:
#   - SpamAssassin contributes HAM ONLY (easy_ham, easy_ham_2, hard_ham).
#     The spam subsets are EXCLUDED: generic spam != phishing and mislabeling it
#     as phishing would inject label noise. data_prep enforces this; a unit test
#     asserts no SpamAssassin-spam message reaches the processed table.
#   - Nazario is the sole phishing (positive) source.
#   - Enron contributes a sampled HAM subset for legitimate-email diversity.


@dataclass(frozen=True)
class SourceSpec:
    name: str
    label: int
    base_url: str
    # Specific archive files to fetch (resolved/confirmed at download time).
    files: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


SOURCES: dict[str, SourceSpec] = {
    "spamassassin": SourceSpec(
        name="spamassassin",
        label=LABEL_HAM,
        base_url="https://spamassassin.apache.org/old/publiccorpus/",
        files=(
            "20030228_easy_ham.tar.bz2",
            "20030228_easy_ham_2.tar.bz2",
            "20030228_hard_ham.tar.bz2",
        ),  # spam_2 / spam intentionally omitted
        note="HAM ONLY; spam subsets excluded by design.",
    ),
    "nazario": SourceSpec(
        name="nazario",
        label=LABEL_PHISHING,
        base_url="http://monkey.org/~jose/phishing/",
        files=(),  # mbox filenames are confirmed before download
        note="Sole phishing source; cite the Nazario corpus. URLs defanged on release.",
    ),
    "enron": SourceSpec(
        name="enron",
        label=LABEL_HAM,
        base_url="https://www.cs.cmu.edu/~enron/",
        files=("enron_mail_20150507.tar.gz",),
        note="Large corpus (~1.7GB uncompressed); we sample HAM only (ENRON_HAM_SAMPLE).",
    ),
}

# Enron is large; we only need ham diversity. Sample this many ham messages
# (seeded) rather than ingesting all ~500k. Set to None to use all (not advised).
ENRON_HAM_SAMPLE: int = 20_000

# --------------------------------------------------------------------------- #
# Split / cross-validation / bootstrap
# --------------------------------------------------------------------------- #
TEST_SIZE: float = 0.20  # stratified, seeded
CV_FOLDS: int = 5  # stratified k-fold on the training set
N_BOOTSTRAP: int = 1_000  # bootstrap resamples for test-set metric CIs
BOOTSTRAP_CI: float = 0.95  # confidence level for reported intervals

# --------------------------------------------------------------------------- #
# Attack: severities
# --------------------------------------------------------------------------- #
# Severity encodes rewrite AGGRESSIVENESS (in the prompt), not a fraction of
# emails. At severity s, every sampled phishing TEST email is replaced by its own
# severity-s rewrite; ham is never attacked. 0.0 is the clean anchor.
SEVERITIES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
ATTACK_SEVERITIES: tuple[float, ...] = tuple(s for s in SEVERITIES if s > 0.0)

# Each non-zero severity maps to a committed prompt file.
# Levels are cumulative: 0.25 light lexical paraphrase; 0.5 + sentence
# restructuring; 0.75 + tone/register shift (remove urgency); 1.0 full rewrite
# preserving only the malicious ask + URLs.
SEVERITY_PROMPT_FILES: dict[float, Path] = {
    0.25: PROMPTS_DIR / "severity_025.txt",
    0.50: PROMPTS_DIR / "severity_050.txt",
    0.75: PROMPTS_DIR / "severity_075.txt",
    1.00: PROMPTS_DIR / "severity_100.txt",
}

# Cap on how many held-out phishing emails get rewritten. Calls ~= sample * 4.
# data_prep/attack report the projected call count and pause before spending.
MAX_PHISH_SAMPLE: int = 500

# --------------------------------------------------------------------------- #
# Attack: provider abstraction
# --------------------------------------------------------------------------- #
# The provider/model is configurable so results can later be shown to hold across
# rewriting models. First run uses Anthropic Haiku only.
ATTACK_PROVIDER: str = os.environ.get(
    "ATTACK_PROVIDER", "anthropic"
)  # anthropic|gemini|openai|local
ATTACK_MODEL: str = os.environ.get("ATTACK_MODEL", "claude-haiku-4-5")

# Default model per provider, so a cross-model run can be selected with just
# ATTACK_PROVIDER=gemini; an explicit ATTACK_MODEL=... still wins (see model_for).
DEFAULT_MODEL_FOR_PROVIDER: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}
ATTACK_TEMPERATURE: float = 0.7
ATTACK_MAX_TOKENS: int = 1_024

# Optional client-side request pacing (requests/minute) to stay comfortably under
# a provider's RPM cap on long synchronous runs. 0 disables pacing. Sequential
# single-threaded calls rarely approach this on their own; it's a safety valve
# that complements (does not replace) the transient-error backoff in attack.py.
# Override per run via ATTACK_RPM=... (e.g. the paid-tier Gemini Flash run).
ATTACK_RPM: int = int(os.environ.get("ATTACK_RPM", "0") or "0")
USE_BATCH: bool = True  # Anthropic Batch API (50% cost reduction)
USE_PROMPT_CACHE: bool = True  # cache the static instruction prefix (~90% off cached input)

# Rough price table (USD per 1M tokens) for the pre-spend cost estimate only.
# input, output. Kept here so the estimate is transparent and editable.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (5.0, 25.0),
    # Google Gemini Flash (text), for the cross-model rewriting check.
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
}


def api_key_for(provider: str | None = None) -> str | None:
    """Return the API key for the given provider from the environment."""
    provider = provider or ATTACK_PROVIDER
    return {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "gemini": os.environ.get("GEMINI_API_KEY"),
        "openai": os.environ.get("OPENAI_API_KEY"),
        "local": None,
    }.get(provider)


_ENV_VAR_FOR_PROVIDER: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def require_api_key(provider: str | None = None) -> str:
    """Pre-flight assertion for spend gates: return the API key for `provider`,
    or raise SystemExit with a clear message if it is missing or empty.

    Call this at the top of every code path that spends money on API calls, so a
    blank/unset key fails fast and loudly BEFORE any request is issued — never
    mid-run after partial spend. The "local" provider needs no key.
    """
    provider = provider or ATTACK_PROVIDER
    if provider == "local":
        return ""
    key = api_key_for(provider)
    if not key or not key.strip():
        env_name = _ENV_VAR_FOR_PROVIDER.get(provider, f"{provider.upper()}_API_KEY")
        raise SystemExit(
            f"\nPre-flight check FAILED: {env_name} is missing or empty "
            f"(provider={provider!r}).\n"
            f"  Add a non-empty {env_name} to your .env (gitignored) and retry.\n"
            f"  No API calls were made; nothing was spent."
        )
    return key


def model_for(provider: str | None = None) -> str:
    """Resolve the model name: an explicit ATTACK_MODEL env override always wins;
    otherwise fall back to the provider's default. Lets a Gemini run be selected
    with just ATTACK_PROVIDER=gemini."""
    provider = provider or ATTACK_PROVIDER
    if os.environ.get("ATTACK_MODEL"):
        return os.environ["ATTACK_MODEL"]
    return DEFAULT_MODEL_FOR_PROVIDER.get(provider, ATTACK_MODEL)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
TFIDF_MAX_FEATURES: int = 20_000
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
TFIDF_MIN_DF: int = 2

# Small urgency lexicon for a handcrafted feature.
URGENCY_WORDS: tuple[str, ...] = (
    "urgent",
    "immediately",
    "verify",
    "suspend",
    "suspended",
    "expire",
    "expires",
    "act now",
    "account",
    "confirm",
    "password",
    "click here",
    "limited time",
    "warning",
    "alert",
    "important",
    "required",
    "update",
)
