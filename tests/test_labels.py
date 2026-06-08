"""A1 label-mapping invariants for the unified dataset.

These guard the headline label contract: phishing comes ONLY from Nazario,
ham ONLY from SpamAssassin (ham subsets) + Enron, SpamAssassin *spam* is
excluded entirely, and labels are exactly {0, 1}.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config

pytestmark = pytest.mark.skipif(
    not config.DATASET_CSV.exists(),
    reason="unified dataset not built (run `python -m src.data_prep`)",
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(config.DATASET_CSV)


def test_labels_are_binary(df):
    assert set(df["label"].unique()) == {0, 1}


def test_phishing_is_nazario_only(df):
    phish_sources = set(df.loc[df["label"] == 1, "source"].unique())
    assert phish_sources == {"nazario"}, phish_sources


def test_ham_sources_only(df):
    ham_sources = set(df.loc[df["label"] == 0, "source"].unique())
    assert ham_sources <= {"spamassassin", "enron"}, ham_sources


def test_nazario_is_all_phishing(df):
    naz = df[df["source"] == "nazario"]
    assert (naz["label"] == 1).all()


def test_spamassassin_is_all_ham(df):
    """SpamAssassin contributes ham only; its `spam` subset must be excluded so
    generic spam never leaks in as a phishing positive."""
    sa = df[df["source"] == "spamassassin"]
    assert (sa["label"] == 0).all()


def test_imbalance_is_natural_not_resampled(df):
    """No subsampling: ham must materially outnumber phishing (~1:4)."""
    n_phish = int((df["label"] == 1).sum())
    n_ham = int((df["label"] == 0).sum())
    assert n_ham > 3 * n_phish
