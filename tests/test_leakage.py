"""No train/test leakage + feature-shape invariants.

The no-leakage contract (src/features.py): the TF-IDF vectorizer and the
handcrafted scaler are fit on TRAIN only; train and test feature matrices share
the same column space; and no email id appears in both splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, features

pytestmark = pytest.mark.skipif(
    not config.DATASET_CSV.exists(),
    reason="unified dataset not built (run `python -m src.data_prep`)",
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(config.DATASET_CSV)


def test_splits_are_disjoint_by_id(df):
    train_ids = set(df.loc[df["split"] == "train", "id"])
    test_ids = set(df.loc[df["split"] == "test", "id"])
    assert train_ids.isdisjoint(test_ids)


def test_split_values_are_only_train_test(df):
    assert set(df["split"].unique()) == {"train", "test"}


def test_ids_are_unique(df):
    assert df["id"].is_unique


_FEATURES_BUILT = all(
    p.exists()
    for p in (
        features.X_TRAIN_NPZ,
        features.X_TEST_NPZ,
        features.Y_TRAIN_NPY,
        features.Y_TEST_NPY,
        config.VECTORIZER_PATH,
    )
)

needs_features = pytest.mark.skipif(
    not _FEATURES_BUILT, reason="feature cache not built (run `python -m src.features`)"
)


@needs_features
def test_feature_matrices_same_width():
    x_train, y_train, x_test, y_test = features.load_features()
    assert x_train.shape[1] == x_test.shape[1]


@needs_features
def test_feature_rows_match_labels():
    x_train, y_train, x_test, y_test = features.load_features()
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]


@needs_features
def test_width_is_tfidf_plus_handcrafted():
    import joblib

    x_train, *_ = features.load_features()
    vec = joblib.load(config.VECTORIZER_PATH)
    expected = len(vec.get_feature_names_out()) + len(features.HANDCRAFTED_NAMES)
    assert x_train.shape[1] == expected


def test_handcrafted_matrix_shape_and_values():
    """transform_texts-independent: handcrafted matrix is well-formed and the
    URL counter (computed pre-defang on verbatim URLs) works."""
    frame = pd.DataFrame(
        {
            "text": [
                "Click http://evil.com/login now!! URGENT verify your ACCOUNT 123",
                "hello world",
            ],
            "had_html": [True, False],
        }
    )
    m = features.handcrafted_matrix(frame)
    assert m.shape == (2, len(features.HANDCRAFTED_NAMES))
    url_idx = features.HANDCRAFTED_NAMES.index("hc_url_count")
    assert m[0, url_idx] == 1
    assert m[1, url_idx] == 0
    html_idx = features.HANDCRAFTED_NAMES.index("hc_had_html")
    assert m[0, html_idx] == 1.0
    assert m[1, html_idx] == 0.0
    assert np.isfinite(m).all()
