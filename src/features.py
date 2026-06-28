"""Feature extraction: TF-IDF over text + handcrafted features.

No-leakage contract: the TF-IDF vectorizer and the handcrafted-feature scaler are
fit on the TRAIN split only, then applied to the test split (and later to LLM
rewrites via `transform_texts`).

Outputs (cached so detectors/evaluate don't recompute):
  data/processed/X_train.npz, X_test.npz   (sparse feature matrices)
  data/processed/y_train.npy, y_test.npy
  results/models/tfidf_vectorizer.joblib
  results/models/handcrafted_scaler.joblib
  results/models/feature_names.json
"""

from __future__ import annotations

import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from . import config

# Cache paths
X_TRAIN_NPZ = config.PROCESSED_DIR / "X_train.npz"
X_TEST_NPZ = config.PROCESSED_DIR / "X_test.npz"
Y_TRAIN_NPY = config.PROCESSED_DIR / "y_train.npy"
Y_TEST_NPY = config.PROCESSED_DIR / "y_test.npy"
SCALER_PATH = config.MODELS_DIR / "handcrafted_scaler.joblib"
FEATURE_NAMES_JSON = config.MODELS_DIR / "feature_names.json"

# Handcrafted features
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

HANDCRAFTED_NAMES: tuple[str, ...] = (
    "hc_char_len",
    "hc_word_len",
    "hc_url_count",
    "hc_urgency_count",
    "hc_exclaim_count",
    "hc_digit_ratio",
    "hc_caps_ratio",
    "hc_had_html",
)


def handcrafted_matrix(df: pd.DataFrame) -> np.ndarray:
    """Compute the dense handcrafted-feature matrix for a frame with text/had_html.

    URLs are counted on the (verbatim, pre-defang) text — see README defang note.
    """
    texts = df["text"].fillna("").astype(str).tolist()
    had_html = df["had_html"].astype(float).to_numpy()
    urgency = [w.lower() for w in config.URGENCY_WORDS]

    out = np.zeros((len(texts), len(HANDCRAFTED_NAMES)), dtype=float)
    for i, t in enumerate(texts):
        low = t.lower()
        n_chars = len(t)
        n_letters = sum(c.isalpha() for c in t)
        out[i, 0] = n_chars
        out[i, 1] = len(t.split())
        out[i, 2] = len(URL_RE.findall(t))
        out[i, 3] = sum(low.count(w) for w in urgency)
        out[i, 4] = t.count("!")
        out[i, 5] = (sum(c.isdigit() for c in t) / n_chars) if n_chars else 0.0
        out[i, 6] = (sum(c.isupper() for c in t) / n_letters) if n_letters else 0.0
    out[:, 7] = had_html
    return out


# Build / load
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(config.DATASET_CSV)
    df["text"] = df["text"].fillna("")
    df["had_html"] = df["had_html"].astype(bool)
    return df


def _combine(tfidf: sparse.spmatrix, hc_scaled: np.ndarray) -> sparse.csr_matrix:
    return sparse.hstack([tfidf, sparse.csr_matrix(hc_scaled)]).tocsr()


def fit_and_save() -> None:
    """Fit transformers on train, build + cache train/test matrices."""
    config.ensure_dirs()
    warnings.filterwarnings("ignore")  # silence bs4 XML notice etc. during load
    df = load_dataset()
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    tfidf_train = vectorizer.fit_transform(train["text"])
    tfidf_test = vectorizer.transform(test["text"])

    scaler = StandardScaler(with_mean=False)  # keep non-negative / sparse-friendly
    hc_train = scaler.fit_transform(handcrafted_matrix(train))
    hc_test = scaler.transform(handcrafted_matrix(test))

    x_train = _combine(tfidf_train, hc_train)
    x_test = _combine(tfidf_test, hc_test)
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    sparse.save_npz(X_TRAIN_NPZ, x_train)
    sparse.save_npz(X_TEST_NPZ, x_test)
    np.save(Y_TRAIN_NPY, y_train)
    np.save(Y_TEST_NPY, y_test)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    feature_names = list(vectorizer.get_feature_names_out()) + list(HANDCRAFTED_NAMES)
    FEATURE_NAMES_JSON.write_text(json.dumps(feature_names))

    print(f"Features built: train {x_train.shape}, test {x_test.shape}")
    print(
        f"  TF-IDF terms: {len(vectorizer.get_feature_names_out()):,} "
        f"+ {len(HANDCRAFTED_NAMES)} handcrafted"
    )
    print(f"  cached -> {X_TRAIN_NPZ.name}, {X_TEST_NPZ.name}, transformers in {config.MODELS_DIR}")


def ensure_features() -> None:
    """Build features if the cache is missing."""
    needed = (X_TRAIN_NPZ, X_TEST_NPZ, Y_TRAIN_NPY, Y_TEST_NPY, config.VECTORIZER_PATH, SCALER_PATH)
    if not all(p.exists() for p in needed):
        fit_and_save()


def load_features() -> tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix, np.ndarray]:
    ensure_features()
    x_train = sparse.load_npz(X_TRAIN_NPZ)
    x_test = sparse.load_npz(X_TEST_NPZ)
    y_train = np.load(Y_TRAIN_NPY)
    y_test = np.load(Y_TEST_NPY)
    return x_train, y_train, x_test, y_test


def transform_texts(texts: list[str], had_html: list[bool] | np.ndarray) -> sparse.csr_matrix:
    """Featurize arbitrary texts with the saved transformers (used by the attack)."""
    vectorizer = joblib.load(config.VECTORIZER_PATH)
    scaler = joblib.load(SCALER_PATH)
    frame = pd.DataFrame({"text": texts, "had_html": list(had_html)})
    tfidf = vectorizer.transform(frame["text"].fillna(""))
    hc = scaler.transform(handcrafted_matrix(frame))
    return _combine(tfidf, hc)


if __name__ == "__main__":
    fit_and_save()
