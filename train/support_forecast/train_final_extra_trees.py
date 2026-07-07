from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "support_dataset.csv"
OUTPUT_ROOT = ROOT_DIR / "train" / "support_forecast" / "runs"
ARTIFACT_PATH = ROOT_DIR / "models" / "artifacts" / "support_extra_trees_model.pkl"
METADATA_PATH = ROOT_DIR / "models" / "metadata" / "support_extra_trees_model.json"
TARGET = "support_tickets_next_1d"
TEST_START_DATE = "2026-03-01"
RANDOM_STATE = 42

HISTORY_COLUMNS = [
    "support_tickets",
    "network_latency_ms",
    "capacity_used_pct",
    "power_usage_mw",
    "avg_rack_temperature_c",
]


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("final_support_extra_trees")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["region", "date"]).copy()
    grouped = df.groupby("region", sort=False)
    for column in HISTORY_COLUMNS:
        shifted = grouped[column].shift(1)
        lag7 = grouped[column].shift(7)
        df[f"{column}_lag1"] = shifted
        df[f"{column}_lag7"] = lag7
        df[f"{column}_diff1"] = df[column] - shifted
        df[f"{column}_rolling_mean_3"] = shifted.groupby(df["region"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{column}_rolling_mean_7"] = shifted.groupby(df["region"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{column}_rolling_std_7"] = shifted.groupby(df["region"]).rolling(7, min_periods=2).std().reset_index(level=0, drop=True)
        df[f"{column}_rolling_max_7"] = shifted.groupby(df["region"]).rolling(7, min_periods=1).max().reset_index(level=0, drop=True)

    df["infra_pressure"] = (df["capacity_used_pct"] + df["network_latency_ms"] + df["avg_rack_temperature_c"]) / 3
    df["maintenance_latency_interaction"] = df["scheduled_maintenance"] * df["network_latency_ms"]
    df["capacity_power_interaction"] = df["capacity_used_pct"] * df["power_usage_mw"]
    return df.sort_values("date").reset_index(drop=True)


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                categorical_features,
            ),
        ]
    )


def main() -> None:
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(output_dir)
    logger.info("Training final ExtraTrees support forecast model")

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = add_historical_features(df)

    test_start = pd.Timestamp(TEST_START_DATE)
    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()
    X_train = train_df.drop(columns=[TARGET, "date"])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET, "date"])
    y_test = test_df[TARGET]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=400,
                    max_depth=5,
                    max_features=0.6,
                    min_samples_leaf=10,
                    min_samples_split=20,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = np.clip(pipeline.predict(X_test), 0, None)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "safe_mape": float(np.mean(np.abs(y_test - predictions) / np.maximum(y_test, 1))),
    }
    logger.info("Test metrics: %s", metrics)

    run_model_path = output_dir / "support_extra_trees_model.pkl"
    joblib.dump(pipeline, run_model_path)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_PATH)

    metadata = {
        "name": "support_extra_trees_model",
        "problem": "next-day support_tickets count regression",
        "artifact_path": str(ARTIFACT_PATH.relative_to(ROOT_DIR)),
        "saved_with": "joblib",
        "model_type": "ExtraTreesRegressor",
        "selection_metric": "MAE on temporal test set",
        **metrics,
        "best_params": {
            "max_depth": 5,
            "max_features": 0.6,
            "min_samples_leaf": 10,
            "min_samples_split": 20,
            "n_estimators": 400,
        },
        "source_run": str(output_dir.relative_to(ROOT_DIR)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved model to %s", ARTIFACT_PATH.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
