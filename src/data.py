"""Data loading for the UCI Heart Disease (Cleveland) dataset.

Fetches via ucimlrepo on first run and caches a local CSV so later runs
(training, tests, notebook) don't require network access.
"""
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE_PATH = os.path.join(DATA_DIR, "heart_disease.csv")

RAW_FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def fetch_raw():
    """Fetch features + multiclass target (0-4) from the UCI repository."""
    from ucimlrepo import fetch_ucirepo

    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()
    df = X.copy()
    df["num"] = y["num"]
    return df


def load_data(use_cache=True):
    """Return the raw dataframe (13 features + multiclass 'num' target).

    Uses a local cache at data/heart_disease.csv when available so the
    pipeline is reproducible offline; falls back to fetch_ucirepo() otherwise.
    """
    if use_cache and os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH)

    df = fetch_raw()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def binarize_target(df):
    """Collapse the 5-class 'num' severity target (0-4) into a binary
    heart-disease-present indicator (0 = absent, 1 = present), which is the
    standard framing for this dataset."""
    y = (df["num"] > 0).astype(int)
    X = df[RAW_FEATURE_COLUMNS].copy()
    return X, y


if __name__ == "__main__":
    df = load_data(use_cache=False)
    print(f"Fetched {len(df)} rows, cached to {CACHE_PATH}")
    print(df.head())
