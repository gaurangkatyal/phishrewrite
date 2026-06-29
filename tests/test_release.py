"""Defang/refang round-trip for the public release.

The release ships defanged URLs (no live links). These guard two properties:
  - defang() leaves NO live http(s):// scheme, including ones nested in
    redirector query strings (e.g. google /url?q=http://...), which the old
    leading-anchored, per-token rewrite missed;
  - refang() is an exact inverse, so re-fanging reproduces the original text
    (and therefore the original URL features).
"""

from __future__ import annotations

from src import release

_NESTED = "http://www.google.com/url?sa=X&q=http://64-60-13-140.static-ip.example/ebay.com/reg.php"
_CASES = [
    "go to http://1.2.3.4/login now",
    "verify at https://secure.example.com/v and www.evil.example/x",
    f"redirect {_NESTED} end",
    "no urls here at all",
    "",
]


def test_defang_leaves_no_live_scheme():
    for s in _CASES:
        d = release.defang(s)
        assert "http://" not in d and "https://" not in d, f"live scheme survived: {d}"


def test_refang_is_exact_inverse():
    for s in _CASES:
        assert release.refang(release.defang(s)) == s


def test_nested_redirector_url_is_fully_defanged_and_restored():
    d = release.defang(_NESTED)
    assert "http://" not in d  # both the outer and the embedded scheme are neutralised
    assert d.count("hxxp://") == 2
    assert release.refang(d) == _NESTED
