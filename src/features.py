"""Feature engineering for the heart disease risk model.

Builds a single sklearn Pipeline that:
  1. Buckets age into clinical risk bands (engineered feature).
  2. Imputes the dataset's known missing values in `ca` and `thal`.
  3. One-hot encodes nominal categorical clinical fields.
  4. Scales continuous features.

The resulting ColumnTransformer is meant to be the first step of a
Pipeline whose second step is the classifier, so the whole thing
(preprocessing + model) can be persisted and served as one object.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Continuous clinical measurements, scaled.
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# Nominal categorical fields (including the engineered age_band), one-hot encoded.
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal", "age_band"]

# Already-binary fields, passed through untouched.
BINARY_FEATURES = ["sex", "fbs", "exang"]

AGE_BAND_BINS = [0, 40, 50, 60, 120]
AGE_BAND_LABELS = ["under_40", "40_49", "50_59", "60_plus"]


def add_age_band(df):
    df = df.copy()
    df["age_band"] = pd.cut(
        df["age"], bins=AGE_BAND_BINS, labels=AGE_BAND_LABELS, right=False
    ).astype(str)
    return df


class AgeBandAdder(BaseEstimator, TransformerMixin):
    """Bucket `age` into clinical risk bands as an additional categorical feature."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_age_band(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features) + ["age_band"])


def build_preprocessing_pipeline():
    """ColumnTransformer covering impute -> encode/scale for all raw fields."""
    continuous_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    binary_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])

    column_transformer = ColumnTransformer([
        ("continuous", continuous_pipeline, CONTINUOUS_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("binary", binary_pipeline, BINARY_FEATURES),
    ])

    return Pipeline([
        ("age_band", AgeBandAdder()),
        ("columns", column_transformer),
    ])


def risk_category(probability):
    """Map a predicted probability of disease to a human-readable risk band."""
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Medium"
    return "High"
