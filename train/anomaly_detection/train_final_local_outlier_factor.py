from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "raw" / "ml_training_dataset.csv"
OUTPUT_ROOT = ROOT_DIR / "train" / "anomaly_detection" / "runs"
ARTIFACT_PATH = ROOT_DIR / "models" / "artifacts" / "anomaly_local_outlier_factor.pkl"
METADATA_PATH = ROOT_DIR / "models" / "metadata" / "anomaly_local_outlier_factor.json"
TARGET = "overload_anomaly"
TEST_START_DATE = "2026-03-01"
MAX_FIT_ROWS = 30000
RANDOM_STATE = 42
DROP_COLUMNS = ["customer_id", "incident_next_7d", TARGET]


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("final_anomaly_lof")
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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    start_date = df["date"].min()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["days_since_start"] = (df["date"] - start_date).dt.days
    df["cpu_ram_pressure"] = df["cpu_util_pct"] * df["ram_util_pct"] / 100
    df["thermal_pressure"] = df["temperature_c"] * df["cpu_util_pct"] / 100
    df["network_total_gb"] = df["net_in_gb"] + df["net_out_gb"]
    df["network_balance_gb"] = df["net_in_gb"] - df["net_out_gb"]
    df["utilization_pressure"] = (
        df["cpu_util_pct"] + df["ram_util_pct"] + df["disk_util_pct"] + df["capacity_used_pct"]
    ) / 4
    return df.sort_values("date").reset_index(drop=True)


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )


def anomaly_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return -pipeline.decision_function(X)


def reference_values(X_train_fit: pd.DataFrame) -> dict[str, dict[str, object]]:
    numeric_reference = {}
    for column in X_train_fit.select_dtypes(include="number").columns:
        numeric_reference[column] = {
            "median": float(X_train_fit[column].median()),
            "mean": float(X_train_fit[column].mean()),
            "std": float(X_train_fit[column].std(ddof=0)),
        }

    categorical_reference = {}
    for column in X_train_fit.select_dtypes(include=["object", "string"]).columns:
        categorical_reference[column] = str(X_train_fit[column].mode().iloc[0])

    return {"numeric": numeric_reference, "categorical": categorical_reference}


def main() -> None:
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(output_dir)
    logger.info("Training final LocalOutlierFactor anomaly model")

    df = add_features(pd.read_csv(DATA_PATH))
    test_start = pd.Timestamp(TEST_START_DATE)
    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()
    drop_columns = [*DROP_COLUMNS, "date", "server_id"]
    X_train = train_df.drop(columns=drop_columns)
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=drop_columns)
    y_test = test_df[TARGET]

    sampled_index = X_train.sample(n=min(MAX_FIT_ROWS, len(X_train)), random_state=RANDOM_STATE).index
    X_train_fit = X_train.loc[sampled_index]
    y_train_fit = y_train.loc[sampled_index]
    contamination = float(max(y_train.mean(), 0.001))

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train_fit)),
            (
                "model",
                LocalOutlierFactor(
                    n_neighbors=35,
                    contamination=contamination,
                    novelty=True,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train_fit)
    train_scores = anomaly_scores(pipeline, X_train_fit)
    threshold = float(np.quantile(train_scores, 1 - contamination))
    test_scores = anomaly_scores(pipeline, X_test)
    y_pred = (test_scores >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "contamination": contamination,
        "train_rows": int(len(X_train)),
        "train_fit_rows": int(len(X_train_fit)),
        "test_rows": int(len(X_test)),
        "train_fit_anomaly_rate": float(y_train_fit.mean()),
        "test_anomaly_rate": float(y_test.mean()),
        "predicted_anomaly_rate": float(y_pred.mean()),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, test_scores)),
        "roc_auc": float(roc_auc_score(y_test, test_scores)),
    }
    logger.info("Metrics: %s", metrics)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_model_path = output_dir / "anomaly_local_outlier_factor.pkl"
    joblib.dump(pipeline, run_model_path)
    joblib.dump(pipeline, ARTIFACT_PATH)

    metadata = {
        "name": "anomaly_local_outlier_factor",
        "problem": "overload anomaly detection",
        "artifact_path": str(ARTIFACT_PATH.relative_to(ROOT_DIR)),
        "saved_with": "joblib",
        "model_type": "LocalOutlierFactor",
        "target_used_for_evaluation_only": TARGET,
        "selection_metric": "average_precision on temporal test set",
        **metrics,
        "reference_values": reference_values(X_train_fit),
        "source_run": str(output_dir.relative_to(ROOT_DIR)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved model to %s", ARTIFACT_PATH.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
