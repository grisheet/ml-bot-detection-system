"""
Synthetic Data Generator for ML Bot Detection System
Generates realistic labeled dataset with bot and human traffic patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

np.random.seed(42)


def generate_bot_features(n: int) -> dict:
    """Generate feature vectors for bot traffic."""
    return {
        # Bots make requests very frequently with low variance
        "request_rate": np.random.uniform(10, 100, n),
        "request_interval_std": np.random.uniform(0.001, 0.05, n),
        # Bots often have suspicious or known bot user-agents
        "ua_bot_score": np.random.uniform(0.6, 1.0, n),
        "ua_entropy": np.random.uniform(1.0, 2.5, n),
        # Bots have no natural mouse movement
        "mouse_movement_score": np.random.uniform(0.0, 0.25, n),
        "click_interval_std": np.random.uniform(0.001, 0.03, n),
        # Bots hammer the same IP with many requests
        "ip_request_count": np.random.randint(200, 2000, n).astype(float),
        "ip_unique_paths": np.random.randint(1, 10, n).astype(float),
        # Short sessions, high error rates
        "session_duration": np.random.uniform(1, 30, n),
        "error_rate": np.random.uniform(0.1, 0.6, n),
        "payload_size_avg": np.random.uniform(64, 1024, n),
        # Bots often active at unusual hours
        "time_of_day": np.random.choice(
            list(range(0, 6)) + list(range(22, 24)), n
        ).astype(float),
        "label": np.ones(n, dtype=int),
    }


def generate_human_features(n: int) -> dict:
    """Generate feature vectors for human traffic."""
    return {
        # Humans browse at natural, variable rates
        "request_rate": np.random.uniform(0.1, 5.0, n),
        "request_interval_std": np.random.uniform(0.5, 10.0, n),
        # Humans have diverse, legitimate user-agents
        "ua_bot_score": np.random.uniform(0.0, 0.3, n),
        "ua_entropy": np.random.uniform(3.0, 5.5, n),
        # Humans move mouse naturally
        "mouse_movement_score": np.random.uniform(0.65, 1.0, n),
        "click_interval_std": np.random.uniform(0.3, 3.0, n),
        # Humans come from varied IPs with moderate request counts
        "ip_request_count": np.random.randint(1, 100, n).astype(float),
        "ip_unique_paths": np.random.randint(5, 50, n).astype(float),
        # Longer sessions, low error rates
        "session_duration": np.random.uniform(30, 1800, n),
        "error_rate": np.random.uniform(0.0, 0.1, n),
        "payload_size_avg": np.random.uniform(512, 8192, n),
        # Humans active during normal hours
        "time_of_day": np.random.choice(
            list(range(7, 23)), n
        ).astype(float),
        "label": np.zeros(n, dtype=int),
    }


def add_noise(df: pd.DataFrame, noise_fraction: float = 0.05) -> pd.DataFrame:
    """Add random noise to make the dataset more realistic."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "label"]

    n_noisy = int(len(df) * noise_fraction)
    noisy_indices = np.random.choice(df.index, n_noisy, replace=False)

    for col in numeric_cols:
        noise = np.random.normal(0, df[col].std() * 0.1, n_noisy)
        df.loc[noisy_indices, col] += noise

    return df


def generate_dataset(
    n_samples: int = 10000,
    bot_ratio: float = 0.4,
    output_path: str = "data/sample_data.csv",
) -> pd.DataFrame:
    """Generate the complete bot detection dataset."""
    n_bots = int(n_samples * bot_ratio)
    n_humans = n_samples - n_bots

    print(f"Generating {n_samples} samples ({n_bots} bots, {n_humans} humans)...")

    bot_data = generate_bot_features(n_bots)
    human_data = generate_human_features(n_humans)

    df_bots = pd.DataFrame(bot_data)
    df_humans = pd.DataFrame(human_data)

    df = pd.concat([df_bots, df_humans], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = add_noise(df)

    # Clip values to valid ranges
    df["ua_bot_score"] = df["ua_bot_score"].clip(0.0, 1.0)
    df["mouse_movement_score"] = df["mouse_movement_score"].clip(0.0, 1.0)
    df["error_rate"] = df["error_rate"].clip(0.0, 1.0)
    df["request_rate"] = df["request_rate"].clip(0.01)
    df["request_interval_std"] = df["request_interval_std"].clip(0.0001)
    df["ip_request_count"] = df["ip_request_count"].clip(1)
    df["ip_unique_paths"] = df["ip_unique_paths"].clip(1)
    df["session_duration"] = df["session_duration"].clip(0.1)
    df["time_of_day"] = df["time_of_day"].clip(0, 23)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"\nFeature summary:\n{df.describe().round(3)}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bot detection dataset")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--bot-ratio", type=float, default=0.4)
    parser.add_argument("--output", type=str, default="data/sample_data.csv")
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.samples,
        bot_ratio=args.bot_ratio,
        output_path=args.output,
    )
