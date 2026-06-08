"""Download, clean, and unify the public corpora into one labeled table.

Pipeline:  verify URLs  ->  download (gated)  ->  parse  ->  clean  ->
           unify to SCHEMA_COLUMNS  ->  dedup  ->  stratified split  ->
           class-balance report.

Label policy (enforced here, asserted in tests):
  - SpamAssassin: HAM ONLY (easy_ham / easy_ham_2 / hard_ham). Spam excluded.
  - Nazario:      phishing (positive class).
  - Enron:        sampled HAM.

CLI:
  python -m src.data_prep --verify-urls   # HEAD/ranged-GET probe, no download
  python -m src.data_prep --download      # actually fetch into data/raw (GATED)
  python -m src.data_prep --build         # parse already-downloaded raw -> processed
  python -m src.data_prep --all           # verify -> download -> build

Heavy imports (pandas, bs4, sklearn) are loaded lazily so --verify-urls runs on
the stdlib alone, before dependencies are installed.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import config

_UA = "PhishRewrite/0.1 (defensive research; +https://github.com/)"

# --------------------------------------------------------------------------- #
# Candidate URLs
# --------------------------------------------------------------------------- #
# monkey.org (Nazario) hosting is historically flaky and has moved over the
# years; SpamAssassin Apache mirror paths also drift. We therefore probe every
# canonical URL at the gate and REPORT failures instead of silently substituting
# a mirror — dataset provenance goes in the paper, so a fallback is the author's
# explicit choice, not an automatic one.

# Nazario filenames vary by year/release; we probe a candidate set and use those
# that resolve. (Empty config.SOURCES["nazario"].files signals "discover here".)
NAZARIO_CANDIDATE_FILES: tuple[str, ...] = (
    "phishing0.mbox",
    "phishing1.mbox",
    "phishing2.mbox",
    "phishing3.mbox",
    "phishing-2015",
    "phishing-2016",
    "phishing-2017",
    "phishing-2018",
    "phishing-2019",
    "phishing-2020",
    "phishing-2021",
    "phishing-2022",
    "phishing-2023",
    "phishing-2024",
)


@dataclass(frozen=True)
class UrlTarget:
    source: str
    label: int
    url: str
    required: bool  # if a required target is dead, downloading must not proceed


def candidate_targets() -> list[UrlTarget]:
    """All URLs we intend to fetch, with whether each is required."""
    targets: list[UrlTarget] = []

    sa = config.SOURCES["spamassassin"]
    for f in sa.files:  # all three ham archives are required
        targets.append(UrlTarget("spamassassin", sa.label, sa.base_url + f, required=True))

    naz = config.SOURCES["nazario"]
    # Each individual Nazario file is optional, but AT LEAST ONE must resolve
    # (handled in verify_urls' summary, not per-target).
    for f in NAZARIO_CANDIDATE_FILES:
        targets.append(UrlTarget("nazario", naz.label, naz.base_url + f, required=False))

    enr = config.SOURCES["enron"]
    for f in enr.files:
        targets.append(UrlTarget("enron", enr.label, enr.base_url + f, required=True))

    return targets


# --------------------------------------------------------------------------- #
# URL verification (HEAD, with ranged-GET fallback)
# --------------------------------------------------------------------------- #
@dataclass
class ProbeResult:
    url: str
    ok: bool
    status: int | None
    size: int | None
    detail: str = ""


def _content_length(headers) -> int | None:
    raw = headers.get("Content-Length")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def probe_url(url: str, timeout: int = 20) -> ProbeResult:
    """HEAD the URL; if HEAD is unsupported, fall back to a 1-byte ranged GET."""
    head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(head, timeout=timeout) as r:
            return ProbeResult(url, True, r.status, _content_length(r.headers))
    except urllib.error.HTTPError as e:
        if e.code in (403, 405, 501):  # HEAD not allowed -> try ranged GET
            return _probe_ranged_get(url, timeout)
        return ProbeResult(url, False, e.code, None, f"HTTP {e.code}")
    except urllib.error.URLError as e:
        return ProbeResult(url, False, None, None, f"URLError: {e.reason}")
    except Exception as e:  # noqa: BLE001 - report any probe failure verbatim
        return ProbeResult(url, False, None, None, f"{type(e).__name__}: {e}")


def _probe_ranged_get(url: str, timeout: int) -> ProbeResult:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            # Content-Range carries the full size when a range is honored.
            size = None
            cr = r.headers.get("Content-Range")
            if cr and "/" in cr:
                try:
                    size = int(cr.rsplit("/", 1)[1])
                except ValueError:
                    size = None
            return ProbeResult(url, True, r.status, size, "via ranged GET")
    except urllib.error.HTTPError as e:
        return ProbeResult(url, False, e.code, None, f"HTTP {e.code} (GET)")
    except Exception as e:  # noqa: BLE001
        return ProbeResult(url, False, None, None, f"{type(e).__name__}: {e}")


def _fmt_size(n: int | None) -> str:
    if n is None:
        return "    ?  "
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:7.1f}{unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n}B"


def verify_urls(timeout: int = 20) -> list[ProbeResult]:
    """Probe every candidate URL and print a provenance/availability report.

    Returns the probe results. Does NOT download anything.
    """
    targets = candidate_targets()
    print("Verifying dataset URLs (no download):\n")
    results: list[ProbeResult] = []
    by_source: dict[str, list[ProbeResult]] = {}
    for t in targets:
        res = probe_url(t.url, timeout=timeout)
        results.append(res)
        by_source.setdefault(t.source, []).append(res)
        flag = "ok " if res.ok else "DEAD"
        req = "req" if t.required else "opt"
        print(
            f"  [{flag}] ({req}) {_fmt_size(res.size)}  {t.url}"
            + (f"   <- {res.detail}" if res.detail and not res.ok else "")
        )

    print("\nSummary:")
    blockers: list[str] = []

    # SpamAssassin + Enron: every required target must resolve.
    for src in ("spamassassin", "enron"):
        dead_required = [r for r in by_source.get(src, []) if not r.ok]
        if dead_required:
            blockers.append(f"{src}: {len(dead_required)} required URL(s) unreachable")
        else:
            print(f"  {src}: all required URLs reachable.")

    # Nazario: at least one candidate must resolve, else surface for a manual
    # fallback decision (do NOT auto-substitute a mirror).
    naz_ok = [r for r in by_source.get("nazario", []) if r.ok]
    if naz_ok:
        print(f"  nazario: {len(naz_ok)} of {len(by_source['nazario'])} candidate files reachable.")
    else:
        blockers.append(
            "nazario: NO canonical files reachable at monkey.org — choose a "
            "documented fallback source manually (provenance goes in the paper)."
        )

    if blockers:
        print("\n  BLOCKERS (resolve before downloading):")
        for b in blockers:
            print(f"    - {b}")
    else:
        print("\n  All sources verified. Safe to proceed to --download.")
    return results


# --------------------------------------------------------------------------- #
# Download (GATED)
# --------------------------------------------------------------------------- #
def download(timeout: int = 60) -> None:
    """Fetch verified corpora into data/raw. Refuses to run if verification fails."""
    config.ensure_dirs()
    results = verify_urls(timeout=20)
    ok_by_url = {r.url: r for r in results}

    # Re-derive blockers without re-printing the whole report.
    naz_ok = any(ok_by_url[t.url].ok for t in candidate_targets() if t.source == "nazario")
    req_dead = [t for t in candidate_targets() if t.required and not ok_by_url[t.url].ok]
    if req_dead or not naz_ok:
        print("\nAborting download: unresolved URL blockers (see above).", file=sys.stderr)
        sys.exit(2)

    for t in candidate_targets():
        if not ok_by_url[t.url].ok:
            continue  # skip optional dead Nazario candidates
        dest = config.RAW_DIR / t.source / t.url.rsplit("/", 1)[1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  exists, skip: {dest}")
            continue
        print(f"  downloading {t.url} -> {dest}")
        req = urllib.request.Request(t.url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as fh:
            while chunk := r.read(1 << 16):
                fh.write(chunk)
    print("Download complete.")


# --------------------------------------------------------------------------- #
# Parse + clean (run after download)
# --------------------------------------------------------------------------- #
# Link-bearing attributes whose destinations are the actual phishing payload.
# get_text() discards attributes, so we inline these targets into the text before
# extraction; otherwise ~half of HTML phishing emails lose every URL (the href
# lives in the tag, only the anchor text survives). Schemes that are never a
# web landing page (mailto/js/in-page anchors/inline data) are skipped.
_LINK_ATTRS: tuple[tuple[str, str], ...] = (
    ("a", "href"),
    ("area", "href"),
    ("form", "action"),
)
_SKIP_URL_PREFIXES = ("mailto:", "javascript:", "tel:", "data:", "#")


def _looks_like_html(text: str) -> bool:
    """Heuristic: does this text contain HTML markup (tags)?"""
    import re

    return re.search(r"</?[a-zA-Z][^>]*>", text) is not None


def _html_to_text(html: str) -> str:
    import warnings

    from bs4 import (
        BeautifulSoup,  # lazy
        XMLParsedAsHTMLWarning,
    )

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Inline link destinations so href/action URLs survive get_text().
    for tag_name, attr in _LINK_ATTRS:
        for el in soup.find_all(tag_name):
            val = (el.get(attr) or "").strip()
            if val and not val.lower().startswith(_SKIP_URL_PREFIXES):
                el.append(f" {val} ")
    return soup.get_text(separator=" ")


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _extract_subject_body(raw_bytes: bytes) -> tuple[str, str, bool]:
    """Return (subject, plain_body, had_html) from raw RFC822 bytes."""
    import email
    from email import policy

    msg = email.message_from_bytes(raw_bytes, policy=policy.default)
    subject = _normalize_ws(str(msg.get("subject", "") or ""))

    had_html = False
    plain_parts: list[str] = []
    html_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                plain_parts.append(_part_text(part))
            elif ctype == "text/html":
                had_html = True
                html_parts.append(_part_text(part))
    else:
        text = _part_text(msg)
        if msg.get_content_type() == "text/html":
            had_html = True
            html_parts.append(text)
        else:
            plain_parts.append(text)

    if plain_parts:
        body = "\n".join(p for p in plain_parts if p)
        # Some scam samples carry raw HTML inside a text/plain part; clean it the
        # same way (strips tag residue like </a> AND inlines href URLs).
        if _looks_like_html(body):
            had_html = True
            body = _html_to_text(body)
    elif html_parts:
        had_html = True
        body = "\n".join(_html_to_text(h) for h in html_parts if h)
    else:
        body = ""

    return subject, _normalize_ws(body), had_html


def _is_low_content(subject: str, body: str) -> bool:
    """Drop messages with too little real content to rewrite or score.

    Catches empty / near-empty bodies (e.g. image-only HTML phish that left no
    text after stripping). Deliberately conservative — it counts word-like tokens
    (>=2 alphabetic chars) so it won't drop short-but-real emails. Semantic
    word-salad with no recoverable ask is left to the attack-stage refusal/no-URL
    checks rather than risking false drops here."""
    combined = f"{subject} {body}"
    alpha_tokens = [w for w in combined.split() if sum(c.isalpha() for c in w) >= 2]
    return len(alpha_tokens) < 5


def _part_text(part) -> str:
    try:
        payload = part.get_content()
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace")
        return str(payload)
    except Exception:  # noqa: BLE001 - tolerate undecodable parts
        raw = part.get_payload(decode=True)
        return raw.decode("utf-8", errors="replace") if raw else ""


def _iter_spamassassin():
    """Yield (original_id, raw_bytes) from the SpamAssassin HAM archives only."""
    import tarfile

    for archive in config.SOURCES["spamassassin"].files:
        path = config.RAW_DIR / "spamassassin" / archive
        if not path.exists():
            continue
        with tarfile.open(path, "r:bz2") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name.rsplit("/", 1)[-1]
                if name.startswith("."):  # cmds / index files
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                yield f"sa:{archive}:{member.name}", fh.read()


def _iter_nazario():
    """Yield (original_id, raw_bytes) from downloaded Nazario mbox/flat files."""
    import mailbox

    naz_dir = config.RAW_DIR / "nazario"
    if not naz_dir.exists():
        return
    for path in sorted(naz_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            box = mailbox.mbox(str(path))
            for i, message in enumerate(box):
                yield f"naz:{path.name}:{i}", message.as_bytes()
        except Exception:  # noqa: BLE001 - some files may not be valid mbox
            continue


def _iter_enron(sample_n: int | None, seed: int):
    """Reservoir-sample (original_id, raw_bytes) ham messages from the Enron tar."""
    import random
    import tarfile

    path = config.RAW_DIR / "enron" / config.SOURCES["enron"].files[0]
    if not path.exists():
        return
    rng = random.Random(seed)
    reservoir: list[tuple[str, bytes]] = []
    seen = 0
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            item = (f"enron:{member.name}", fh.read())
            seen += 1
            if sample_n is None:
                reservoir.append(item)
            elif len(reservoir) < sample_n:
                reservoir.append(item)
            else:
                j = rng.randint(0, seen - 1)
                if j < sample_n:
                    reservoir[j] = item
    yield from reservoir


def build(sample_enron: int | None = config.ENRON_HAM_SAMPLE) -> None:
    """Parse downloaded raw data into the unified, split, deduped processed table."""
    import hashlib

    import pandas as pd
    from sklearn.model_selection import train_test_split

    config.ensure_dirs()
    rows: list[dict] = []

    iterators = [
        ("spamassassin", config.LABEL_HAM, _iter_spamassassin()),
        ("nazario", config.LABEL_PHISHING, _iter_nazario()),
        ("enron", config.LABEL_HAM, _iter_enron(sample_enron, config.SEED)),
    ]

    seen_keys: set[str] = set()
    n_dupes = 0
    n_lowcontent = 0
    for source, label, it in iterators:
        for original_id, raw in it:
            subject, body, had_html = _extract_subject_body(raw)
            if _is_low_content(subject, body):
                n_lowcontent += 1
                continue
            dedup_key = hashlib.sha1(
                (subject.lower() + "\n" + body.lower()).encode("utf-8")
            ).hexdigest()
            if dedup_key in seen_keys:
                n_dupes += 1
                continue
            seen_keys.add(dedup_key)
            text = (subject + "\n" + body).strip()
            rows.append(
                {
                    "text": text,
                    "subject": subject,
                    "body": body,
                    "label": label,
                    "source": source,
                    "original_id": original_id,
                    "had_html": had_html,
                }
            )

    if not rows:
        print("No rows parsed — did you run --download first?", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.insert(0, "id", [f"{s}-{i}" for i, s in enumerate(df["source"])])

    # SAFETY: no SpamAssassin spam should ever appear as phishing.
    assert not ((df["source"] == "spamassassin") & (df["label"] == config.LABEL_PHISHING)).any()

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=df["label"],
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    df = df[list(config.SCHEMA_COLUMNS)]
    df.to_csv(config.DATASET_CSV, index=False)
    print(
        f"Wrote {len(df):,} rows ({n_dupes:,} duplicates, "
        f"{n_lowcontent:,} low-content dropped) -> {config.DATASET_CSV}"
    )

    _report_class_balance(df)


def _report_class_balance(df) -> None:

    per_source = df.groupby(["source", "label"]).size().rename("count").reset_index()
    overall = df.groupby(["split", "label"]).size().rename("count").reset_index()
    out = config.TABLES_DIR / "class_balance.csv"
    per_source.to_csv(out, index=False)

    n_phish = int((df["label"] == config.LABEL_PHISHING).sum())
    n_ham = int((df["label"] == config.LABEL_HAM).sum())
    print("\nClass balance:")
    print(f"  phishing (1): {n_phish:,}")
    print(f"  ham      (0): {n_ham:,}")
    print(f"  ratio phish:ham = 1 : {n_ham / max(n_phish, 1):.2f}")
    print("\n  per source x label:")
    print(per_source.to_string(index=False))
    print("\n  per split x label:")
    print(overall.to_string(index=False))
    print(f"\n  (written to {out})")
    if n_ham > 3 * n_phish:
        print(
            "\n  NOTE: ham heavily outnumbers phishing — consider subsampling ham. "
            "Bringing this table to the user before deciding."
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PhishRewrite data preparation")
    parser.add_argument(
        "--verify-urls",
        action="store_true",
        help="probe dataset URLs (HEAD/ranged GET); no download",
    )
    parser.add_argument(
        "--download", action="store_true", help="download verified corpora into data/raw (GATED)"
    )
    parser.add_argument(
        "--build", action="store_true", help="parse downloaded raw data into the processed table"
    )
    parser.add_argument("--all", action="store_true", help="verify -> download -> build")
    args = parser.parse_args(argv)

    if args.all:
        download()
        build()
    elif args.download:
        download()
    elif args.build:
        build()
    else:  # default and --verify-urls both just verify
        verify_urls()


if __name__ == "__main__":
    main()
