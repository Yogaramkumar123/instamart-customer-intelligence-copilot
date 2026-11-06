"""Leakage-free preprocessing pipeline and data partitioning for lapse prediction.

Ensures:
- Train-test split (80/20 stratified, random_state=42).
- Preprocessing transformer fitted ONLY on training split.
- Encoders and scalers never observe test data during fitting.
"""

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "average_discount_pct",
    "cancellation_rate",
    "average_delivery_minutes",
    "age",
    "tenure_days"
]

CATEGORICAL_FEATURES = [
    "favourite_category",
    "city",
    "gender"
]


def create_preprocessor() -> ColumnTransformer:
    """Build ColumnTransformer for numeric standardisation and nominal one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )


def prepare_train_test_data(
    df_features: pd.DataFrame,
    y: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Split data into stratified train/test partitions and compute preprocessing validation metrics."""
    # Exclude customer_id from model features
    X = df_features[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    # Assert no missing values
    assert X.isna().sum().sum() == 0, "Input features contain missing values"

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Preprocessor fitted strictly on training data
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)

    # Validate frequency means: full cohort vs training scaler mean
    full_cohort_freq_mean = float(X["frequency"].mean())
    freq_idx = NUMERIC_FEATURES.index("frequency")
    train_scaler_freq_mean = float(preprocessor.named_transformers_["num"].mean_[freq_idx])

    # Assert split dimensions
    assert len(X_train) == 924, f"Expected 924 train rows, got {len(X_train)}"
    assert len(X_test) == 232, f"Expected 232 test rows, got {len(X_test)}"

    metrics = {
        "full_cohort_frequency_mean": round(full_cohort_freq_mean, 4),
        "train_scaler_frequency_mean": round(train_scaler_freq_mean, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_lapsed_count": int(np.sum(y_train)),
        "test_lapsed_count": int(np.sum(y_test)),
        "feature_names": NUMERIC_FEATURES + CATEGORICAL_FEATURES
    }

    return X_train, X_test, y_train, y_test, metrics


if __name__ == "__main__":
    from src.part_b.lapse_features import build_lapse_dataset
    X_raw, y_all, _ = build_lapse_dataset()
    X_tr, X_te, y_tr, y_te, split_meta = prepare_train_test_data(X_raw, y_all)
    print("Train/Test Split Metadata:")
    for k, v in split_meta.items():
        print(f"  {k}: {v}")
