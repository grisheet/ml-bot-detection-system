"""
Model Training Module for ML Bot Detection System
Trains multiple models, evaluates performance, and saves the best model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import joblib
import json
import logging
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed. Skipping XGBoost model.")

from src.preprocess import full_preprocessing_pipeline, save_scaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def get_models() -> Dict[str, Any]:
    """Return dict of model instances to train."""
    models = {
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, n_jobs=-1
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        "neural_network": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    return models


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    return metrics


def train_all_models(
    data_path: str = "data/sample_data.csv",
) -> Dict[str, Dict]:
    """Train all models and return results."""
    logger.info("Starting model training pipeline...")

    (
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols
    ) = full_preprocessing_pipeline(data_path=data_path)

    save_scaler(scaler, MODELS_DIR / "scaler.pkl")

    # Save feature columns
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(list(feature_cols), f)

    models = get_models()
    results = {}
    best_f1 = -1
    best_model_name = None
    best_model = None

    for name, model in models.items():
        logger.info(f"Training {name}...")
        try:
            model.fit(X_train, y_train)

            val_metrics = evaluate_model(model, X_val, y_val)
            test_metrics = evaluate_model(model, X_test, y_test)

            results[name] = {
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "classification_report": classification_report(
                    y_test, model.predict(X_test), output_dict=True
                ),
            }

            logger.info(
                f"  {name} -> Val F1: {val_metrics['f1']:.4f} | "
                f"Test AUC: {test_metrics['roc_auc']:.4f}"
            )

            # Save individual model
            model_path = MODELS_DIR / f"{name}.pkl"
            joblib.dump(model, model_path)

            if test_metrics["f1"] > best_f1:
                best_f1 = test_metrics["f1"]
                best_model_name = name
                best_model = model

        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")

    # Save best model
    if best_model is not None:
        joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
        with open(MODELS_DIR / "best_model_name.txt", "w") as f:
            f.write(best_model_name)
        logger.info(f"Best model: {best_model_name} (F1={best_f1:.4f})")

    # Save all results
    with open(MODELS_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print_summary(results)
    return results


def print_summary(results: Dict) -> None:
    """Print a formatted summary table of model performance."""
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'AUC':>8}")
    print("=" * 70)
    for name, res in results.items():
        m = res["test_metrics"]
        print(
            f"{name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1']:>8.4f} {m['roc_auc']:>8.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_data.csv"
    train_all_models(data_path=data_path)
