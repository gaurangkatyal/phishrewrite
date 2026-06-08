"""Retention-checker logic for BOTH label-validity definitions.

Strict (all-URLs): a rewrite PASSes only if EVERY original URL survives.
Primary-URL: only the ask-bearing URL(s) must survive; incidental footer/
tracking/brand-nav URLs may be dropped.

These tests pin the behaviour that distinguishes the two definitions (the +61
intersection delta in LIMITATIONS.md depends on exactly this logic).
"""

from __future__ import annotations

from src import attack


# --------------------------------------------------------------------------- #
# Strict (all-URLs) retention
# --------------------------------------------------------------------------- #
def test_strict_pass_when_all_urls_survive():
    r = attack.check_retention(
        "Verify at http://evil.com/login or call us.",
        "Please verify your account at http://evil.com/login.",
    )
    assert r["retained_urls"] is True
    assert r["missing_urls"] == []
    assert r["n_orig_urls"] == 1


def test_strict_fail_when_any_url_dropped():
    r = attack.check_retention(
        "Verify at http://evil.com/login and unsubscribe http://list.example.com/u",
        "Please verify your account at http://evil.com/login.",
    )
    assert r["retained_urls"] is False
    assert "http://list.example.com/u" in r["missing_urls"]


def test_strict_no_urls_is_flagged_for_manual_review():
    """No URL anchor -> can't auto-verify; retained_urls is None (manual), not a
    silent pass/fail."""
    r = attack.check_retention("no links here", "still no links")
    assert r["n_orig_urls"] == 0
    assert r["retained_urls"] is None


# --------------------------------------------------------------------------- #
# Primary-URL retention
# --------------------------------------------------------------------------- #
def test_primary_keeps_credential_lander_drops_footer():
    """The classic strict-FAIL -> primary-PASS flip: credential URL kept, only an
    incidental social-footer URL dropped."""
    original = (
        "Confirm your account at http://1.2.3.4/login.php " "Follow us http://facebook.com/brand"
    )
    primaries = attack.primary_urls(original)
    assert "http://1.2.3.4/login.php" in primaries
    assert "http://facebook.com/brand" not in primaries

    r = attack.check_retention_primary(original, "Please confirm at http://1.2.3.4/login.php")
    assert r["retained_primary"] is True
    assert r["missing_primary_urls"] == []


def test_primary_fail_when_ask_url_dropped():
    original = "Update your details at http://1.2.3.4/verify and unsub http://list.org/u"
    r = attack.check_retention_primary(original, "Update your details soon.")
    assert r["retained_primary"] is False
    assert "http://1.2.3.4/verify" in r["missing_primary_urls"]


def test_ip_host_beats_brand_in_strength():
    """A raw-IP credential host must outrank a bare known-brand host."""
    primaries = attack.primary_urls(
        "login http://1.2.3.4/secure and info http://paypal.com/privacy"
    )
    assert "http://1.2.3.4/secure" in primaries
    assert "http://paypal.com/privacy" not in primaries


def test_no_primary_anchor_returns_none_retained():
    """When the email has only incidental URLs there is no ask anchor; the
    primary checker reports retained_primary is None (not a PASS, not a FAIL)."""
    r = attack.check_retention_primary(
        "follow http://facebook.com/brand and http://twitter.com/brand",
        "follow http://facebook.com/brand",
    )
    assert r["retained_primary"] is None
    assert r["n_primary_urls"] == 0


def test_primary_is_subset_of_strict_difficulty():
    """Anything that passes strict (all URLs kept) must also pass primary."""
    original = "verify http://1.2.3.4/login and footer http://facebook.com/brand"
    full_keep = "verify http://1.2.3.4/login and footer http://facebook.com/brand"
    strict = attack.check_retention(original, full_keep)
    primary = attack.check_retention_primary(original, full_keep)
    assert strict["retained_urls"] is True
    assert primary["retained_primary"] is True
