"""Attack I/O against a mocked provider stub (no network, no API key).

Verifies that Rewriter.rewrite():
  - calls the provider client and parses (raw_text, usage) out of the response;
  - retries transient errors with backoff and eventually succeeds;
  - re-raises non-transient errors immediately;
and that make_record() turns a raw rewrite into a well-formed cache record with
both the parsed fields and the retention verdict.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import attack


# Fakes that mimic the anthropic Messages response shape
class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Usage:
    def __init__(self):
        self.input_tokens = 11
        self.output_tokens = 7
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


class _Resp:
    def __init__(self, text):
        self.content = [_Block(text)]
        self.usage = _Usage()


class _FakeMessages:
    """Records the kwargs of the most recent create() call."""

    def __init__(self, text, fail_times=0, error=None):
        self.text = text
        self.fail_times = fail_times
        self.error = error or RuntimeError("boom")
        self.calls = 0
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.calls <= self.fail_times:
            raise self.error
        return _Resp(self.text)


class _FakeClient:
    def __init__(self, messages):
        self.messages = messages


def _rewriter_with(messages) -> attack.Rewriter:
    rw = attack.Rewriter("anthropic", "claude-test", temperature=0.0, max_tokens=256)
    rw._client = _FakeClient(messages)  # bypass _ensure_client / network
    rw._system = "SYSTEM"  # avoid prompt-file dependency
    return rw


class _Transient(Exception):
    """Carries a status_code so attack._is_transient() classifies it as retryable."""

    def __init__(self):
        super().__init__("429 rate limit exceeded")
        self.status_code = 429


# Tests
def test_rewrite_parses_text_and_usage():
    msgs = _FakeMessages("Subject: Hi\n\nBody here http://x.test/login")
    rw = _rewriter_with(msgs)
    raw, usage = rw.rewrite("rewrite this", "original email")
    assert "http://x.test/login" in raw
    assert usage["input_tokens"] == 11
    assert usage["output_tokens"] == 7
    assert msgs.calls == 1
    # the model + token budget are forwarded to the client
    assert msgs.last_kwargs["model"] == "claude-test"
    assert msgs.last_kwargs["max_tokens"] == 256


def test_rewrite_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr(attack.time, "sleep", lambda *_: None)  # no real backoff
    msgs = _FakeMessages("ok", fail_times=2, error=_Transient())
    rw = _rewriter_with(msgs)
    raw, _ = rw.rewrite("instr", "email")
    assert raw == "ok"
    assert msgs.calls == 3  # 2 failures + 1 success


def test_rewrite_reraises_non_transient(monkeypatch):
    monkeypatch.setattr(attack.time, "sleep", lambda *_: None)
    msgs = _FakeMessages("never", fail_times=99, error=ValueError("bad request"))
    rw = _rewriter_with(msgs)
    with pytest.raises(ValueError):
        rw.rewrite("instr", "email")
    assert msgs.calls == 1  # not retried


def test_make_record_is_well_formed():
    msgs = _FakeMessages("Subject: Verify\n\nGo to http://1.2.3.4/login now")
    rw = _rewriter_with(msgs)
    raw, usage = rw.rewrite("instr", "Go to http://1.2.3.4/login now")
    row = pd.Series(
        {
            "id": "e1",
            "original_id": "o1",
            "source": "nazario",
            "text": "Please go to http://1.2.3.4/login now",
        }
    )
    rec = attack.make_record(row, 0.5, "instr", "SYSTEM", rw, raw, usage)
    assert rec["original_id"] == "o1"
    assert rec["severity"] == 0.5
    assert rec["provider"] == "anthropic"
    assert rec["rewrite_subject"] == "Verify"
    assert "http://1.2.3.4/login" in rec["rewrite_text"]
    # retention fields are merged in
    assert rec["retained_urls"] is True
    assert rec["n_orig_urls"] == 1
    assert rec["refused"] is False


# The shared rewrite driver (attack.run_rewrite_loop), exercised against the fakes
# above so the cache/skip/error paths are covered without network or an API key.
_SAMPLE = pd.DataFrame(
    [
        {"id": "e1", "original_id": "o1", "source": "nazario", "text": "Go to http://a.test/login"},
        {"id": "e2", "original_id": "o2", "source": "nazario", "text": "Verify at http://b.test/v"},
    ]
)
_SEVS = {0.5: "instr-0.5", 1.0: "instr-1.0"}


def test_run_loop_calls_model_and_appends_new():
    msgs = _FakeMessages("Subject: Hi\n\nGo to http://a.test/login")
    rw = _rewriter_with(msgs)
    appended, cache = [], {}
    recs = attack.run_rewrite_loop(
        _SAMPLE,
        _SEVS,
        rewriter=rw,
        system="SYSTEM",
        cache=cache,
        append_fn=appended.append,
        header="=== test ===",
        label="test",
    )
    n = len(_SAMPLE) * len(_SEVS)
    assert msgs.calls == n  # every (email, severity) hit the model
    assert len(appended) == n  # ...and every new record was persisted
    assert len(recs) == n
    assert len(cache) == n  # cache populated keyed by (id, severity)


def test_run_loop_reuses_cache_without_calling_model():
    msgs = _FakeMessages("unused")
    rw = _rewriter_with(msgs)
    # Pre-seed the cache with a valid hit for every (email, severity): matching
    # prompt_hash AND input_text_sha means the loop must reuse, never call.
    cache = {}
    for _, row in _SAMPLE.iterrows():
        for sev, instr in _SEVS.items():
            key = attack._cache_key(row["original_id"], sev)
            cache[key] = {
                "original_id": row["original_id"],
                "prompt_hash": attack.prompt_hash(
                    "SYSTEM", instr, row["text"], rw.model, rw.temperature
                ),
                "input_text_sha": attack.text_sha(row["text"]),
            }
    appended = []
    recs = attack.run_rewrite_loop(
        _SAMPLE,
        _SEVS,
        rewriter=rw,
        system="SYSTEM",
        cache=cache,
        append_fn=appended.append,
        header="=== test ===",
        label="test",
    )
    assert msgs.calls == 0  # everything served from cache
    assert appended == []  # nothing new written
    assert len(recs) == len(_SAMPLE) * len(_SEVS)


def test_run_loop_skips_per_call_errors(monkeypatch):
    monkeypatch.setattr(attack.time, "sleep", lambda *_: None)
    # A non-transient, non-account error is logged and skipped, not raised.
    msgs = _FakeMessages("never", fail_times=99, error=ValueError("bad request"))
    rw = _rewriter_with(msgs)
    appended, cache = [], {}
    recs = attack.run_rewrite_loop(
        _SAMPLE,
        _SEVS,
        rewriter=rw,
        system="SYSTEM",
        cache=cache,
        append_fn=appended.append,
        header="=== test ===",
        label="test",
    )
    assert recs == []  # all failed -> none recorded
    assert appended == []
    assert cache == {}
