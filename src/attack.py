"""The LLM rewriting attack.

Each held-out PHISHING test email is rewritten at every non-zero severity so it
reads as more benign while preserving the malicious ask and ALL URLs verbatim
(the label-validity contract). Rewrites are cached to data/processed/
rewrites.jsonl, keyed by (original_id, severity) and guarded by a prompt hash, so
the attack is idempotent and resumable. The degradation evaluation
(evaluate --degradation) scores detectors on these rewrites.

Spending is gated. Inspect everything for free first, then spend in two stages:

    python -m src.attack --show-prompts   # the verbatim severity prompts (free)
    python -m src.attack --estimate       # projected call count + cost (free)
    python -m src.attack --pilot          # 5 emails x 4 severities = 20 calls
    python -m src.attack --run --yes      # the remaining full sample (gated)

--run refuses to spend without --yes, so the full ~2,000-call run can only happen
after the pilot has been reviewed.

Provider/model are configurable (config.ATTACK_PROVIDER / ATTACK_MODEL); the first
run uses Anthropic Haiku 4.5. The static system prompt is sent with prompt caching
so its tokens are billed at the cached rate across the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from . import config

PILOT_EMAILS = 5  # 5 emails x 4 severities = 20 calls

# Retry policy for transient provider errors (rate limits / 5xx / overloaded).
_MAX_RETRIES = 6
_TRANSIENT_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_TRANSIENT_HINTS = (
    "overloaded",
    "unavailable",
    "high demand",
    "rate limit",
    "ratelimit",
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "try again",
)


def _is_transient(e: Exception) -> bool:
    """True for errors worth retrying (vs. a 4xx auth/validation that won't fix
    itself). Works across Anthropic and Gemini SDK exception shapes."""
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    if isinstance(code, int) and code in _TRANSIENT_CODES:
        return True
    blob = f"{type(e).__name__} {e}".lower()
    return any(h in blob for h in _TRANSIENT_HINTS)


# Account-level failures that no retry or resume can fix until the user acts
# (exhausted credits, invalid/revoked key). Kept deliberately narrow so it does
# NOT match Gemini's transient 429 quota message (which also mentions "billing"):
# those stay in the transient->retry->skip path above.
_FATAL_ACCOUNT_HINTS = (
    "credit balance is too low",
    "credit balance too low",
    "purchase credits",
    "upgrade or purchase",
    "invalid api key",
    "invalid x-api-key",
    "authentication_error",
    "authentication error",
    "permission_error",
    "permission denied",
)


def fatal_account_message(e: Exception) -> str | None:
    """If `e` is an account-level error a run can't recover from (billing/credits
    or auth), return a short one-line message; otherwise None. Used to abort a
    spend loop cleanly instead of dumping a traceback or skipping every call."""
    blob = f"{type(e).__name__} {e}".lower()
    if any(h in blob for h in _FATAL_ACCOUNT_HINTS):
        return str(e).replace("\n", " ")[:300]
    return None


# Tokens that the rewrite begins with, per the committed output contract.
SUBJECT_PREFIX = "Subject:"

# Retention-specific URL regex. Unlike the (greedy) handcrafted-feature regex,
# this stops at quotes/brackets/angle-brackets so it never captures HTML residue
# (e.g. a mashed  ...webscr/">https://...</a>  token) and over-count URLs.
RETENTION_URL_RE = re.compile(r"""https?://[^\s<>"')\]]+|www\.[^\s<>"')\]]+""", re.I)

# Phrases that signal the model broke character / refused instead of returning an
# email (seen on garbage word-salad inputs with no recoverable ask).
_REFUSAL_MARKERS = (
    "i appreciate you testing",
    "i can't help",
    "i cannot help",
    "i'm unable",
    "i am unable",
    "doesn't contain",
    "does not contain",
    "could you provide",
    "please provide the",
    "as an ai",
    "i'm not able",
    "i am not able",
    "appears to be corrupted",
    "appears to be incomplete",
    "i need a complete",
    "no identifiable",
)


# Prompts
def load_system_prompt() -> str:
    return (config.PROMPTS_DIR / "system.txt").read_text(encoding="utf-8")


def load_severity_prompts() -> dict[float, str]:
    """{severity: instruction text} for every non-zero severity, in order."""
    out: dict[float, str] = {}
    for sev in config.ATTACK_SEVERITIES:
        path = config.SEVERITY_PROMPT_FILES[sev]
        if not path.exists():
            raise FileNotFoundError(f"missing severity prompt: {path}")
        out[sev] = path.read_text(encoding="utf-8")
    return out


def build_user_text(instruction: str, email_text: str) -> str:
    """The user-turn content: severity instruction + the email to rewrite."""
    return (
        f"{instruction.strip()}\n\n"
        "Rewrite the following phishing email. Output only the rewritten email "
        "in the required format.\n\n"
        "----- BEGIN EMAIL -----\n"
        f"{email_text}\n"
        "----- END EMAIL -----"
    )


def prompt_hash(
    system: str, instruction: str, email_text: str, model: str, temperature: float
) -> str:
    """Stable hash of everything that determines a rewrite. Guards the cache:
    if any prompt/text/model/temp changes, the old cached rewrite is ignored."""
    h = hashlib.sha256()
    for part in (system, instruction, email_text, model, f"{temperature:.4f}"):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def text_sha(email_text: str) -> str:
    """Hash of the input email's processed text alone. Stored alongside
    prompt_hash so a change to a message's processed text (e.g. after a data_prep
    fix) invalidates that message's cached rewrites even if the prompts didn't
    change — making the invalidation intent explicit and robust to future
    refactors of prompt_hash."""
    return hashlib.sha256(email_text.encode("utf-8")).hexdigest()


# Sampling the phishing test pool (seeded, stable)
def sample_phish_test(limit: int | None = None) -> pd.DataFrame:
    """The seeded sample of held-out phishing emails to attack.

    Always drawn from the same seeded ordering so --pilot and --run operate on a
    consistent set (the pilot's 5 are the first 5 of the full sample)."""
    df = pd.read_csv(config.DATASET_CSV)
    phish_test = df[(df["split"] == "test") & (df["label"] == config.LABEL_PHISHING)]
    n = min(config.MAX_PHISH_SAMPLE, len(phish_test))
    sample = phish_test.sample(n=n, random_state=config.SEED).reset_index(drop=True)
    if limit is not None:
        sample = sample.iloc[:limit].reset_index(drop=True)
    return sample


# Rewrite parsing + URL-retention check
def parse_rewrite(raw: str) -> tuple[str, str, str]:
    """Split a model rewrite into (subject, body, full_text).

    full_text mirrors how training rows were built (subject + body) so it can be
    featurized by the same vectorizer and scored fairly."""
    text = raw.strip()
    subject, body = "", text
    if text.lower().startswith(SUBJECT_PREFIX.lower()):
        first, _, rest = text.partition("\n")
        subject = first[len(SUBJECT_PREFIX) :].strip()
        body = rest.strip()
    full = f"{subject}\n{body}".strip()
    return subject, body, full


def extract_urls(text: str) -> list[str]:
    """URLs in `text`, using the strict (non-HTML-greedy) retention regex."""
    return RETENTION_URL_RE.findall(text or "")


def _norm_url(u: str) -> str:
    """Normalize for comparison: lowercase, drop trailing punctuation/slash so a
    cosmetic trailing '/' or sentence punctuation doesn't cause a false miss."""
    return u.rstrip("/.,;:!?)]}>\"'").lower()


def check_retention(original_text: str, rewrite_full: str) -> dict:
    """Automated retention check: every URL in the original must appear in the rewrite.
    Emails with no URL can't be auto-verified this way -> flagged for the manual
    spot-check (retained=None) rather than silently passed/failed."""
    orig = list(dict.fromkeys(_norm_url(u) for u in extract_urls(original_text)))
    rw = [_norm_url(u) for u in extract_urls(rewrite_full)]
    rw_set = set(rw)
    # A URL is retained if it matches a rewrite URL exactly or up to a path-suffix
    # difference (substring either direction handles added/dropped trailing bits).
    missing = [u for u in orig if u not in rw_set and not any(u in r or r in u for r in rw)]
    if not orig:
        retained: bool | None = None  # no URL anchor -> manual review
    else:
        retained = len(missing) == 0
    return {
        "n_orig_urls": len(orig),
        "missing_urls": missing,
        "retained_urls": retained,
    }


def looks_like_refusal(rewrite_full: str) -> bool:
    """True if the model returned meta-text / a refusal instead of an email."""
    low = rewrite_full.lower()
    return any(m in low for m in _REFUSAL_MARKERS)


# Primary-URL retention (a less conservative label-validity variant)
# The strict check (check_retention) requires EVERY original URL to survive. That
# over-counts failures: rewrites that drop only incidental links (social/footer/
# tracking/unsubscribe/asset) while keeping the credential-harvest ("primary")
# URL are still valid phishing. This variant identifies the ask-bearing URL(s)
# per email and requires only those to be retained. Deterministic, no API.

_HOST_RE = re.compile(r"(?:https?://)?(?:www\.)?([^/\s:]+)", re.I)
_IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

# Hosts/paths that are essentially never the ask: they are footer/chrome.
_INCIDENTAL_HOSTS = (
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "linkedin.com",
    "youtube.com",
    "pinterest.com",
    "plus.google.com",
    "t.me",
    "whatsapp.com",
    "doubleclick.net",
    "sendgrid.net",
    "rds.yahoo.com",
    "list-manage.com",
    "mailchimp.com",
    "exct.net",
    "sparkpostmail.com",
    "mcsv.net",
    "w3.org",
    "verisign.com",
    "validator.w3.org",
    "sourceforge.net",
    "apache.org",
    "adobe.com",
    "monkey.org",  # monkey.org = Nazario corpus host
)
_INCIDENTAL_PATH_HINTS = (
    "unsubscribe",
    "optout",
    "opt-out",
    "listinfo",
    "mailman",
    "/list/",
    "preferences",
    "manage-subscription",
    "email-preferences",
    "remove?",
)
_ASSET_RE = re.compile(r"\.(?:png|jpe?g|gif|css|js|ico|svg|woff2?)(?:\?|$)", re.I)

# Credential / ask path keywords that strongly mark a link as the payload.
_ASK_PATH_HINTS = (
    "login",
    "signin",
    "sign-in",
    "verify",
    "secure",
    "account",
    "update",
    "confirm",
    "webscr",
    "cmd=",
    "banking",
    "password",
    "auth",
    "session",
    "validate",
    "unlock",
    "resolve",
    "billing",
    "payment",
    "wp-admin",
    "/r/",
    "redirect",
    "logon",
    "authenticate",
    "ebayisapi",
)

# Well-known legitimate brand/registrar hosts. A bare link to one of these
# (no ask-path) is usually a decoy/footer rather than the lander, so it scores
# lower than an unknown host.
_KNOWN_BRANDS = (
    "paypal.com",
    "ebay.com",
    "ebay.co.uk",
    "amazon.com",
    "apple.com",
    "microsoft.com",
    "google.com",
    "yahoo.com",
    "ncua.gov",
    "usaa.com",
    "wellsfargo.com",
    "chase.com",
    "bankofamerica.com",
    "citi.com",
    "hsbc.com",
    "barclays.co.uk",
    "natwest.com",
    "halifax.co.uk",
    "scotiabank.com",
    "creditunions.com",
    "irs.gov",
    "fdic.gov",
)


def _host_of(u: str) -> str:
    m = _HOST_RE.match(u.lower())
    return m.group(1) if m else u.lower()


def _is_incidental_url(u: str) -> bool:
    """True for footer/chrome links that are essentially never the ask."""
    h = _host_of(u)
    low = u.lower()
    if any(h == d or h.endswith("." + d) for d in _INCIDENTAL_HOSTS):
        return True
    if h.startswith("click") and "ebay" in h:  # ebay click-tracking redirector
        return True
    if _ASSET_RE.search(low):
        return True
    if any(k in low for k in _INCIDENTAL_PATH_HINTS):
        return True
    return False


def _is_known_brand(h: str) -> bool:
    return any(h == d or h.endswith("." + d) for d in _KNOWN_BRANDS)


def _url_strength(u: str) -> int:
    """Ask-likelihood score for a non-incidental URL. Higher = more likely the
    credential-harvest target. -1 marks an incidental (never-primary) link."""
    if _is_incidental_url(u):
        return -1
    h = _host_of(u)
    score = 0
    if _IP_HOST_RE.match(h):
        score += 3  # raw IP host: classic phishing lander
    if any(k in u.lower() for k in _ASK_PATH_HINTS):
        score += 2  # credential/ask path
    if not _is_known_brand(h):
        score += 1  # unknown host more likely the lander
    return score


def primary_urls(text: str) -> list[str]:
    """The ask-bearing URL(s) in `text`, normalized. Relative within the email:
    incidental links are removed; among the rest the highest-strength tier wins.
    If the only candidates are bare known-brand links (no ask-path, no IP), they
    are ambiguous and ALL kept as primary (conservative — must be retained).
    Returns [] when the email has no identifiable primary anchor."""
    urls = list(dict.fromkeys(_norm_url(u) for u in extract_urls(text)))
    scored = [(u, _url_strength(u)) for u in urls]
    cand = [(u, s) for u, s in scored if s >= 0]  # drop incidental
    if not cand:
        return []
    top = max(s for _u, s in cand)
    if top == 0:  # all bare known-brand: ambiguous
        return [u for u, _s in cand]
    return [u for u, s in cand if s == top]


def check_retention_primary(original_text: str, rewrite_full: str) -> dict:
    """Primary-URL variant of check_retention: PASS iff every PRIMARY (ask-bearing)
    URL survives; incidental URLs may be dropped. Emails with no identifiable
    primary anchor return retained_primary=None (manual review, like no-URL)."""
    prim = primary_urls(original_text)
    rw = [_norm_url(u) for u in extract_urls(rewrite_full)]
    rw_set = set(rw)
    missing = [u for u in prim if u not in rw_set and not any(u in r or r in u for r in rw)]
    retained: bool | None = None if not prim else (len(missing) == 0)
    return {
        "n_primary_urls": len(prim),
        "primary_urls": prim,
        "missing_primary_urls": missing,
        "retained_primary": retained,
    }


# Provider abstraction
class Rewriter:
    """Thin provider wrapper. Anthropic (Haiku) and Gemini (Flash) are implemented
    so the benchmark can show the attack holds across rewriting models; openai/
    local remain stubs. The same system prompt + output contract is used for every
    provider so rewrites are directly comparable."""

    def __init__(self, provider: str, model: str, temperature: float, max_tokens: int):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._system = load_system_prompt()

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.provider == "anthropic":
            self._client = self._init_anthropic()
        elif self.provider == "gemini":
            self._client = self._init_gemini()
        else:
            raise NotImplementedError(
                f"provider '{self.provider}' not implemented; use anthropic or gemini"
            )

    def _init_anthropic(self):
        key = config.api_key_for("anthropic")
        if not key:
            raise SystemExit(
                "ANTHROPIC_API_KEY not set. Add it to .env (see .env.example) "
                "before running --pilot or --run."
            )
        try:
            import anthropic
        except ModuleNotFoundError as e:  # pragma: no cover
            raise SystemExit(
                "anthropic package not installed; pip install -r requirements.txt"
            ) from e
        return anthropic.Anthropic(api_key=key)

    def _init_gemini(self):
        key = config.api_key_for("gemini")
        if not key:
            raise SystemExit(
                "GEMINI_API_KEY not set. Add it to .env (see .env.example) "
                "before running --pilot or --run with ATTACK_PROVIDER=gemini."
            )
        try:
            from google import genai
        except ModuleNotFoundError as e:  # pragma: no cover
            raise SystemExit("google-genai not installed; pip install -r requirements.txt") from e
        return genai.Client(api_key=key)

    def _system_blocks(self) -> list[dict]:
        """Static system prompt with a cache breakpoint so it is billed at the
        cached rate after the first call (USE_PROMPT_CACHE). Anthropic-specific."""
        block: dict = {"type": "text", "text": self._system}
        if config.USE_PROMPT_CACHE:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def rewrite(self, instruction: str, email_text: str) -> tuple[str, dict]:
        """One synchronous rewrite. Returns (raw_text, usage_dict). Transient
        provider errors (429/5xx/overloaded/high-demand) are retried with
        exponential backoff so a momentary spike doesn't abort a long run."""
        self._ensure_client()
        user_text = build_user_text(instruction, email_text)
        fn = self._rewrite_gemini if self.provider == "gemini" else self._rewrite_anthropic
        delay = 2.0
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return fn(user_text)
            except Exception as e:  # noqa: BLE001 — narrow via _is_transient
                if attempt == _MAX_RETRIES or not _is_transient(e):
                    raise
                msg = str(e).replace("\n", " ")[:160]
                print(
                    f"    transient error ({type(e).__name__}: {msg}); retry "
                    f"{attempt}/{_MAX_RETRIES - 1} in {delay:.0f}s"
                )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")  # pragma: no cover

    def _rewrite_anthropic(self, user_text: str) -> tuple[str, dict]:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self._system_blocks(),
            messages=[{"role": "user", "content": user_text}],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        u = resp.usage
        usage = {
            "input_tokens": getattr(u, "input_tokens", 0),
            "output_tokens": getattr(u, "output_tokens", 0),
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        return raw, usage

    def _rewrite_gemini(self, user_text: str) -> tuple[str, dict]:
        from google.genai import types

        # Gemini 2.5 Flash enables "thinking" by default, which is billed against
        # max_output_tokens — left on, it consumed the whole budget and truncated
        # the rewrite (dropping URLs). We disable it so the full budget produces
        # the email; the rewrite task needs no chain-of-thought.
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=self._system,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = resp.text or ""
        u = getattr(resp, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(u, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(u, "candidates_token_count", 0) or 0,
            "thinking_tokens": getattr(u, "thoughts_token_count", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cached_content_token_count", 0) or 0,
            "cache_creation_input_tokens": 0,
        }
        return raw, usage


# Rewrite cache (JSONL, resumable)
def _cache_key(original_id, severity: float) -> str:
    return f"{original_id}|{severity:.2f}"


def rewrites_path() -> "object":
    """Where this provider/model's rewrites live. The canonical Haiku run keeps
    data/processed/rewrites.jsonl (backward-compatible, used by evaluate); any
    other provider/model writes to a model-slugged sibling file so a cross-model
    run never collides with — or corrupts — the headline Haiku dataset."""
    model = config.model_for()
    default = config.DEFAULT_MODEL_FOR_PROVIDER.get("anthropic")
    if config.ATTACK_PROVIDER == "anthropic" and model == default:
        return config.REWRITES_JSONL
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-")
    return config.PROCESSED_DIR / f"rewrites_{slug}.jsonl"


def load_rewrite_cache() -> dict[str, dict]:
    """Existing rewrites keyed by (original_id, severity). Last write wins."""
    path = rewrites_path()
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cache[_cache_key(rec["original_id"], float(rec["severity"]))] = rec
    return cache


def append_rewrite(rec: dict) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with rewrites_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_record(row, severity, instruction, system, rewriter, raw, usage) -> dict:
    subject, body, full = parse_rewrite(raw)
    retention = check_retention(row["text"], full)
    refused = looks_like_refusal(full)
    return {
        "id": row["id"],
        "original_id": row["original_id"],
        "source": row["source"],
        "severity": round(float(severity), 2),
        "provider": rewriter.provider,
        "model": rewriter.model,
        "temperature": rewriter.temperature,
        "prompt_hash": prompt_hash(
            system, instruction, row["text"], rewriter.model, rewriter.temperature
        ),
        "input_text_sha": text_sha(row["text"]),
        "rewrite_subject": subject,
        "rewrite_body": body,
        "rewrite_text": full,
        "had_html": False,  # rewrites are plain text
        "refused": refused,
        **retention,
        "usage": usage,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


# Token / cost estimate (no API calls; char/4 heuristic)
def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate(limit: int | None = None) -> None:
    system = load_system_prompt()
    severities = load_severity_prompts()
    sample = sample_phish_test(limit)
    n_emails = len(sample)
    n_calls = n_emails * len(severities)

    sys_tok = _approx_tokens(system)
    instr_tok = {s: _approx_tokens(t) for s, t in severities.items()}
    avg_instr = sum(instr_tok.values()) / len(instr_tok)
    email_toks = [_approx_tokens(t) for t in sample["text"].fillna("").astype(str)]
    avg_email = sum(email_toks) / max(1, len(email_toks))
    # Output assumed comparable to the email length, capped at max_tokens.
    avg_out = min(config.ATTACK_MAX_TOKENS, max(64, avg_email))

    model = config.model_for()
    price = config.PRICE_PER_MTOK.get(model, (1.0, 5.0))
    in_price, out_price = price  # USD per 1M tokens

    # Per call: system tokens (cached after first), severity instruction + email
    # as fresh input, plus output.
    fresh_in_per_call = avg_instr + avg_email
    # System tokens: ~1 cache-write + (n_calls-1) cache reads (~0.1x) if caching.
    if config.USE_PROMPT_CACHE:
        sys_in_tokens_effective = sys_tok * 1.25 + (n_calls - 1) * sys_tok * 0.1
    else:
        sys_in_tokens_effective = n_calls * sys_tok

    total_fresh_in = n_calls * fresh_in_per_call
    total_in_tokens = total_fresh_in + sys_in_tokens_effective
    total_out_tokens = n_calls * avg_out

    in_cost = total_in_tokens / 1e6 * in_price
    out_cost = total_out_tokens / 1e6 * out_price
    subtotal = in_cost + out_cost
    # The 50%-off Batch API path is Anthropic-only; other providers run synchronous.
    use_batch = config.USE_BATCH and config.ATTACK_PROVIDER == "anthropic"
    batch_factor = 0.5 if use_batch else 1.0
    total = subtotal * batch_factor

    print("=== attack cost estimate (rough upper bound) ===")
    print(f"  provider/model : {config.ATTACK_PROVIDER} / {model}")
    print(f"  price (USD/MTok): input {in_price}, output {out_price}")
    print(
        f"  phishing test pool sampled : {n_emails} "
        f"(cap MAX_PHISH_SAMPLE={config.MAX_PHISH_SAMPLE})"
    )
    print(f"  severities : {len(severities)} -> {list(severities)}")
    print(f"  projected calls : {n_emails} x {len(severities)} = {n_calls}")
    print(
        f"  avg tokens : system={sys_tok}, instruction~={avg_instr:.0f}, "
        f"email~={avg_email:.0f}, output~={avg_out:.0f}"
    )
    print(
        f"  prompt caching : {'on' if config.USE_PROMPT_CACHE else 'off'}; "
        f"batch (50% off) : {'on' if use_batch else 'off'}"
    )
    print(f"  est. input tokens  : {total_in_tokens/1e6:.3f} M  -> ${in_cost:.2f}")
    print(f"  est. output tokens : {total_out_tokens/1e6:.3f} M  -> ${out_cost:.2f}")
    print(f"  est. subtotal : ${subtotal:.2f}")
    if use_batch:
        print(f"  est. with batch discount (x0.5) : ${total:.2f}")
    print(f"  ESTIMATED TOTAL : ${total:.2f}")
    print("  (heuristic char/4 token counts; actual usage is recorded per call.)")


# Show prompts
def show_prompts() -> None:
    system = load_system_prompt()
    severities = load_severity_prompts()
    print("=" * 72)
    print("SYSTEM PROMPT (static, prompt-cached across the run)")
    print("=" * 72)
    print(system.rstrip())
    for sev, text in severities.items():
        print()
        print("=" * 72)
        print(f"SEVERITY {sev:.2f} INSTRUCTION  ({config.SEVERITY_PROMPT_FILES[sev].name})")
        print("=" * 72)
        print(text.rstrip())


# Synchronous rewrite loop (the attack + mitigation passes all share this)
def _retention_label(rec: dict) -> str:
    if rec.get("refused"):
        return "REFUSED (model returned meta-text, not an email)"
    r = rec["retained_urls"]
    if r is None:
        return "manual (no URL)"
    return "PASS" if r else f"FAIL (missing {len(rec['missing_urls'])}/{rec['n_orig_urls']})"


def run_rewrite_loop(
    sample: pd.DataFrame,
    sevs: dict[float, str],
    *,
    rewriter: "Rewriter",
    system: str,
    cache: dict,
    append_fn,
    header: str,
    label: str,
    rpm: int = 0,
    show_each: bool = False,
    retention_summary: bool = False,
) -> list[dict]:
    """Drive one rewrite pass over sample x severities, reusing cache where possible.

    Shared by the main attack (run_sync) and the mitigation / external-validity
    passes. For each (email, severity) it reuses a cache hit whose prompt config and
    input text are both unchanged, otherwise it calls the model and appends the new
    record via append_fn. Per-call errors are logged and skipped so a later re-run
    retries them; only fatal account-level errors abort. Returns cached + new records.
    """
    todo = [(row, sev) for _, row in sample.iterrows() for sev in sevs]
    n_total = len(todo)
    records: list[dict] = []
    n_done = n_skipped = n_errored = 0
    # Client-side request pacing (RPM cap), applied only to real API calls.
    min_interval = 60.0 / rpm if rpm > 0 else 0.0
    last_call_at = 0.0
    print(header)
    if min_interval:
        print(f"  pacing: <= {rpm} req/min (min {min_interval:.3f}s between calls)")
    t0 = time.time()
    for row, sev in todo:
        key = _cache_key(row["original_id"], sev)
        ph = prompt_hash(system, sevs[sev], row["text"], rewriter.model, rewriter.temperature)
        th = text_sha(row["text"])
        cached = cache.get(key)
        # Reuse only if BOTH the prompt config and the input text are unchanged.
        # input_text_sha is the explicit text guard; the missing-field fallback keeps
        # legacy caches (pre-field) valid since prompt_hash already encodes the text.
        if cached and cached.get("prompt_hash") == ph and cached.get("input_text_sha", th) == th:
            n_skipped += 1
            records.append(cached)
            continue
        if min_interval:
            wait = min_interval - (time.time() - last_call_at)
            if wait > 0:
                time.sleep(wait)
        last_call_at = time.time()
        try:
            raw, usage = rewriter.rewrite(sevs[sev], row["text"])
        except Exception as e:  # noqa: BLE001 — one bad call must not abort the run
            fatal = fatal_account_message(e)
            if fatal:
                raise SystemExit(
                    f"\nAborting {label}: account-level error from "
                    f"{rewriter.provider!r} that no retry/resume can fix:\n"
                    f"  {fatal}\n"
                    f"  Progress is cached ({n_done} new this run); fix the "
                    f"account and re-run to resume from the cache."
                )
            n_errored += 1
            msg = f"{type(e).__name__}: {str(e).replace(chr(10), ' ')[:200]}"
            print(
                f"  [{n_done + n_skipped + n_errored}/{n_total}] id="
                f"{row['original_id']} sev={sev:.2f} ERROR (skipped): {msg}"
            )
            continue  # not cached/recorded; a later re-run will retry this pair
        rec = make_record(row, sev, sevs[sev], system, rewriter, raw, usage)
        append_fn(rec)
        cache[key] = rec
        records.append(rec)
        n_done += 1
        if show_each:
            _print_rewrite(row, rec)
        else:
            print(
                f"  [{n_done + n_skipped}/{n_total}] id={row['original_id']} "
                f"sev={sev:.2f} retention={_retention_label(rec)}"
            )
    dt = time.time() - t0
    print(f"\n  done: {n_done} new, {n_skipped} cached, {n_errored} errored/skipped, in {dt:.1f}s")
    if retention_summary:
        _retention_summary(records)
    return records


def run_sync(sample: pd.DataFrame, *, show_each: bool, label: str) -> list[dict]:
    # TODO: this is the synchronous path (RPM-paced, one request at a time). The
    # cost estimate already assumes the 50%-off Batch API (config.USE_BATCH), but
    # the actual batch-submit path isn't wired up yet — runs are sync regardless.
    # Fail fast if the provider's API key is missing/empty before any call is
    # issued; this guards every spend path (pilot + full run).
    config.require_api_key(config.ATTACK_PROVIDER)
    system = load_system_prompt()
    severities = load_severity_prompts()
    rewriter = Rewriter(
        config.ATTACK_PROVIDER,
        config.model_for(),
        config.ATTACK_TEMPERATURE,
        config.ATTACK_MAX_TOKENS,
    )
    header = (
        f"=== {label}: {len(sample)} emails x {len(severities)} severities "
        f"= {len(sample) * len(severities)} calls ==="
    )
    return run_rewrite_loop(
        sample,
        severities,
        rewriter=rewriter,
        system=system,
        cache=load_rewrite_cache(),
        append_fn=append_rewrite,
        header=header,
        label=label,
        rpm=config.ATTACK_RPM,
        show_each=show_each,
    )


def _print_rewrite(row, rec: dict) -> None:
    print("\n" + "-" * 72)
    print(
        f"original_id={rec['original_id']}  severity={rec['severity']:.2f}  "
        f"source={rec['source']}"
    )
    print(f"orig URLs ({rec['n_orig_urls']}): {extract_urls(row['text'])}")
    print(
        f"retention: {_retention_label(rec)}"
        + (f"  missing={rec['missing_urls']}" if rec["missing_urls"] else "")
    )
    orig = str(row["text"])
    print("--- ORIGINAL (truncated 600 chars) ---")
    print(orig[:600] + ("..." if len(orig) > 600 else ""))
    print("--- REWRITE ---")
    print(f"Subject: {rec['rewrite_subject']}")
    print(rec["rewrite_body"])


def _retention_summary(records: list[dict]) -> None:
    by_sev: dict[float, list[dict]] = {}
    for r in records:
        by_sev.setdefault(r["severity"], []).append(r)
    print("\n=== retention summary (per severity) ===")
    print(
        f"  {'severity':>8}  {'pass':>5}  {'fail':>5}  {'manual':>6}  "
        f"{'refused':>7}  {'total':>5}"
    )
    for sev in sorted(by_sev):
        rs = by_sev[sev]
        nref = sum(1 for r in rs if r.get("refused"))
        npass = sum(1 for r in rs if not r.get("refused") and r["retained_urls"] is True)
        nfail = sum(1 for r in rs if not r.get("refused") and r["retained_urls"] is False)
        nman = sum(1 for r in rs if not r.get("refused") and r["retained_urls"] is None)
        print(f"  {sev:>8.2f}  {npass:>5}  {nfail:>5}  {nman:>6}  {nref:>7}  {len(rs):>5}")


def run_pilot() -> None:
    sample = sample_phish_test(limit=PILOT_EMAILS)
    records = run_sync(sample, show_each=True, label="PILOT")
    _retention_summary(records)
    print("\nPilot complete. Review the rewrites and retention above.")
    print("If the prompts/retention logic look correct, approve the full run:")
    print("    python -m src.attack --run --yes")


def run_full() -> None:
    sample = sample_phish_test()
    records = run_sync(sample, show_each=False, label="FULL RUN")
    _retention_summary(records)
    print(f"\nWrote rewrites -> {rewrites_path()}")


# CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PhishRewrite attack")
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="print the verbatim system + severity prompts (free)",
    )
    parser.add_argument(
        "--estimate", action="store_true", help="projected call count + cost (free, no API calls)"
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=f"rewrite {PILOT_EMAILS} emails x all severities "
        f"({PILOT_EMAILS * len(config.ATTACK_SEVERITIES)} calls) and show results",
    )
    parser.add_argument(
        "--run", action="store_true", help="full sample run (requires --yes; resumable)"
    )
    parser.add_argument("--yes", action="store_true", help="confirm spending on the full --run")
    args = parser.parse_args(argv)

    config.ensure_dirs()

    if args.show_prompts:
        show_prompts()
        return
    if args.estimate:
        estimate()
        return
    if args.pilot:
        run_pilot()
        return
    if args.run:
        if not args.yes:
            print(
                "Refusing to spend on the full run without --yes.\n"
                "Review the pilot first:\n"
                "    python -m src.attack --pilot\n"
                "then re-run:\n"
                "    python -m src.attack --run --yes"
            )
            sys.exit(2)
        run_full()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
