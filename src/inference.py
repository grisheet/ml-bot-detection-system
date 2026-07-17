"""
Inference Engine for ML Bot Detection System
Loads the trained model and runs predictions on new data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict
import joblib
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
THRESHOLD = 0.5


class BotDetector:
    """
    Inference engine that loads the trained model and scaler
    to predict whether a request comes from a bot or human.
    """

    def __init__(
        self,
        model_path: str = "models/best_model.pkl",
        scaler_path: str = "models/scaler.pkl",
        feature_cols_path: str = "models/feature_cols.json",
        threshold: float = THRESHOLD,
    ):
        self.threshold = threshold
        self._load_artifacts(model_path, scaler_path, feature_cols_path)

    def _load_artifacts(self, model_path, scaler_path, feature_cols_path):
        """Load model, scaler, and feature column list."""
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run: python src/train.py"
            )
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(feature_cols_path) as f:
            self.feature_cols = json.load(f)
        logger.info(
            f"Loaded model from {model_path} with {len(self.feature_cols)} features"
        )

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering as during training."""
        df = df.copy()
        df["req_rate_x_ua_bot"] = df["request_rate"] * df["ua_bot_score"]
        df["ip_density"] = df["ip_request_count"] / (df["ip_unique_paths"] + 1)
        df["human_behavior_score"] = (
            df["mouse_movement_score"] * (1 - df["ua_bot_score"]) * (1 - df["error_rate"])
        )
        df["session_request_ratio"] = df["request_rate"] / (df["session_duration"] + 1)
        df["is_night_hour"] = (
            (df["time_of_day"] < 6) | (df["time_of_day"] >= 22)
        ).astype(int)
        df["log_request_rate"] = np.log1p(df["request_rate"])
        df["log_ip_requests"] = np.log1p(df["ip_request_count"])
        return df

    def predict_proba(self, features: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """Return raw bot probabilities for input features."""
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        df = self._engineer_features(df)
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_scaled)[:, 1]
        else:
            proba = self.model.decision_function(X_scaled)

        return proba

    def predict(self, features: Union[Dict, pd.DataFrame]) -> List[Dict]:
        """
        Run inference and return structured result dicts.
        Each dict contains: classification, bot_probability, confidence.
        """
        probas = self.predict_proba(features)
        results = []
        for p in probas:
            label = "bot" if p >= self.threshold else "human"
            if p >= 0.8 or p <= 0.2:
                confidence = "high"
            elif p >= 0.65 or p <= 0.35:
                confidence = "medium"
            else:
                confidence = "low"
            results.append({
                "classification": label,
                "bot_probability": round(float(p), 4),
                "confidence": confidence,
            })
        return results

    def predict_single(self, features: Dict) -> Dict:
        """Predict a single request."""
        return self.predict(features)[0]


def run_demo():
    """Run a quick demo with example bot and human feature vectors."""
    detector = BotDetector()

    examples = [
        {
            "name": "Obvious Bot",
            "request_rate": 85.0,
            "request_interval_std": 0.005,
            "ua_bot_score": 0.95,
            "ua_entropy": 1.5,
            "mouse_movement_score": 0.05,
            "click_interval_std": 0.01,
            "ip_request_count": 1500.0,
            "ip_unique_paths": 3.0,
            "session_duration": 5.0,
            "error_rate": 0.4,
            "payload_size_avg": 256.0,
            "time_of_day": 3.0,
        },
        {
            "name": "Normal Human",
            "request_rate": 0.8,
            "request_interval_std": 4.5,
            "ua_bot_score": 0.05,
            "ua_entropy": 4.8,
            "mouse_movement_score": 0.9,
            "click_interval_std": 1.2,
            "ip_request_count": 25.0,
            "ip_unique_paths": 18.0,
            "session_duration": 420.0,
            "error_rate": 0.02,
            "payload_size_avg": 3200.0,
            "time_of_day": 14.0,
        },
        {
            "name": "Borderline Case",
            "request_rate": 12.0,
            "request_interval_std": 0.3,
            "ua_bot_score": 0.45,
            "ua_entropy": 3.0,
            "mouse_movement_score": 0.4,
            "click_interval_std": 0.5,
            "ip_request_count": 150.0,
            "ip_unique_paths": 8.0,
            "session_duration": 60.0,
            "error_rate": 0.15,
            "payload_size_avg": 800.0,
            "time_of_day": 21.0,
        },
    ]

    print("\n" + "=" * 60)
    print("Bot Detection Inference Demo")
    print("=" * 60)
    for ex in examples:
        name = ex.pop("name")
        result = detector.predict_single(ex)
        print(f"\n[{name}]")
        print(f"  Classification : {result['classification'].upper()}")
        print(f"  Bot Probability: {result['bot_probability']:.4f}")
        print(f"  Confidence     : {result['confidence']}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
