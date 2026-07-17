"""
Preprocessing Module for ML Bot Detection System
Handles data loading, cleaning, feature engineering, and train/test splitting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "request_rate",
    "request_interval_std",
    "ua_bot_score",
    "ua_entropy",
    "mouse_movement_score",
    "click_interval_std",
    "ip_request_count",
    "ip_unique_paths",
    "session_duration",
    "error_rate",
    "payload_size_avg",
    "time_of_day",
]
TARGET_COL = "label"


def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate the dataset."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Run: python data/generate_data.py"
        )

    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the data."""
    original_len = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = df.drop_duplicates()

    # Remove extreme outliers (beyond 5 sigma)
    for col in FEATURE_COLS:
        mean, std = df[col].mean(), df[col].std()
        df = df[(df[col] >= mean - 5 * std) & (df[col] <= mean + 5 * std)]

    logger.info(f"Cleaned data: {original_len} -> {len(df)} rows")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional engineered features from raw features."""
    df = df.copy()

    # Interaction features
    df["req_rate_x_ua_bot"] = df["request_rate"] * df["ua_bot_score"]
    df["ip_density"] = df["ip_request_count"] / (df["ip_unique_paths"] + 1)
    df["human_behavior_score"] = (
        df["mouse_movement_score"] * (1 - df["ua_bot_score"]) * (1 - df["error_rate"])
    )
    df["session_request_ratio"] = df["request_rate"] / (df["session_duration"] + 1)
    df["is_night_hour"] = ((df["time_of_day"] < 6) | (df["time_of_day"] >= 22)).astype(int)
    df["log_request_rate"] = np.log1p(df["request_rate"])
    df["log_ip_requests"] = np.log1p(df["ip_request_count"])

    logger.info(f"Engineered features: {df.shape[1]} total columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all feature column names (base + engineered)."""
    return [c for c in df.columns if c != TARGET_COL]


def split_and_scale(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scaler_type: str = "robust",
) -> Tuple:
    """
    Split into train/val/test and scale features.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=random_state, stratify=y_temp
    )

    # Scale features
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(
        f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )
    logger.info(f"Train bot rate: {y_train.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols


def save_scaler(scaler, output_path: str = "models/scaler.pkl") -> None:
    """Persist the fitted scaler."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)
    logger.info(f"Scaler saved to {output_path}")


def load_scaler(path: str = "models/scaler.pkl"):
    """Load a persisted scaler."""
    return joblib.load(path)


def full_preprocessing_pipeline(
    data_path: str = "data/sample_data.csv",
    engineer: bool = True,
) -> Tuple:
    """End-to-end preprocessing from raw CSV to scaled arrays."""
    df = load_data(data_path)
    df = clean_data(df)
    if engineer:
        df = engineer_features(df)
    feature_cols = get_feature_columns(df)
    result = split_and_scale(df, feature_cols)
    return result


if __name__ == "__main__":
    result = full_preprocessing_pipeline()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, features = result
    print(f"Features used ({len(features)}): {features}")
    print(f"X_train shape: {X_train.shape}")
    save_scaler(scaler)
