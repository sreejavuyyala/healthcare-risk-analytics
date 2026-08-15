import numpy as np
import pandas as pd
import pytest

from src.data import RAW_FEATURE_COLUMNS
from src.features import add_age_band, build_preprocessing_pipeline, risk_category


def make_sample_df(n=10):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "age": rng.randint(29, 77, n),
        "sex": rng.randint(0, 2, n),
        "cp": rng.randint(1, 5, n),
        "trestbps": rng.randint(94, 200, n),
        "chol": rng.randint(126, 564, n),
        "fbs": rng.randint(0, 2, n),
        "restecg": rng.randint(0, 3, n),
        "thalach": rng.randint(71, 202, n),
        "exang": rng.randint(0, 2, n),
        "oldpeak": rng.uniform(0, 6.2, n),
        "slope": rng.randint(1, 4, n),
        "ca": rng.choice([0.0, 1.0, 2.0, 3.0, np.nan], n),
        "thal": rng.choice([3.0, 6.0, 7.0, np.nan], n),
    })
    return df[RAW_FEATURE_COLUMNS]


def test_add_age_band_buckets_correctly():
    df = pd.DataFrame({"age": [25, 45, 55, 70]})
    banded = add_age_band(df)
    assert banded["age_band"].tolist() == ["under_40", "40_49", "50_59", "60_plus"]


def test_add_age_band_does_not_mutate_input():
    df = pd.DataFrame({"age": [25, 45]})
    add_age_band(df)
    assert "age_band" not in df.columns


@pytest.mark.parametrize("probability,expected", [
    (0.0, "Low"),
    (0.32, "Low"),
    (0.33, "Medium"),
    (0.65, "Medium"),
    (0.66, "High"),
    (1.0, "High"),
])
def test_risk_category_thresholds(probability, expected):
    assert risk_category(probability) == expected


def test_preprocessing_pipeline_handles_missing_ca_and_thal():
    df = make_sample_df()
    assert df["ca"].isna().any() or df["thal"].isna().any()

    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(df)

    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == len(df)


def test_preprocessing_pipeline_output_is_numeric_matrix():
    df = make_sample_df()
    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(df)

    assert np.issubdtype(transformed.dtype, np.floating)
