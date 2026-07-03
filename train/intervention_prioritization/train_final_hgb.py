from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "incident_dataset.csv"
OUTPUT_ROOT = ROOT_DIR / "train" / "intervention_prioritization" / "runs"
ARTIFACT_PATH = ROOT_DIR / "models" / "artifacts" / "intervention_prioritization_hgb.pkl"
METADATA_PATH = ROOT_DIR / "models" / "metadata" / "intervention_prioritization_hgb.json"
TARGET = "incident_next_7d"
TEST_START_DATE = "2026-03-01"
TOP_K = 50
RANDOM_STATE = 42

HISTORY_COLUMNS = [
    "cpu_util_pct",
    "ram_util_pct",
    "disk_util_pct",
    "temperature_c",
    "net_in_gb",
    "net_out_gb",
    "network_latency_ms",
    "support_tickets",
    "capacity_used_pct",
]


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("final_intervention_prioritization")
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
    df = df.sort_values(["server_id", "date"]).copy()
    grouped = df.groupby("server_id", sort=False)
    for column in HISTORY_COLUMNS:
        shifted = grouped[column].shift(1)
        df[f"{column}_lag1"] = shifted
        df[f"{column}_diff1"] = df[column] - shifted
        df[f"{column}_rolling_mean_3"] = shifted.groupby(df["server_id"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{column}_rolling_mean_7"] = shifted.groupby(df["server_id"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{column}_rolling_max_7"] = shifted.groupby(df["server_id"]).rolling(7, min_periods=1).max().reset_index(level=0, drop=True)
    df["cpu_ram_pressure"] = df["cpu_util_pct"] * df["ram_util_pct"] / 100
    df["thermal_pressure"] = df["temperature_c"] * df["cpu_util_pct"] / 100
    df["network_total_gb"] = df["net_in_gb"] + df["net_out_gb"]
    df["network_balance_gb"] = df["net_in_gb"] - df["net_out_gb"]
    df["utilization_pressure"] = (
        df["cpu_util_pct"] + df["ram_util_pct"] + df["disk_util_pct"] + df["capacity_used_pct"]
    ) / 4
    return df.sort_values("date").reset_index(drop=True)


def business_value(df: pd.DataFrame) -> pd.Series:
    support_plan_weight = df["support_plan"].map({"basic": 1.0, "standard": 1.2, "premium": 1.5, "critical": 2.0}).fillna(1.0)
    hardware_weight = 1 + 0.25 * df["has_gpu"] + 0.15 * df["is_managed"]
    pressure_weight = 1 + df["capacity_used_pct"].clip(0, 100) / 200
    return (df["monthly_spend_eur"].clip(lower=10) * support_plan_weight * hardware_weight * pressure_weight).astype(float)


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features),
        ]
    )


def topk_metrics(traceability: pd.DataFrame, probabilities, top_k: int) -> dict[str, float | int]:
    scored = traceability.copy()
    scored["incident_probability"] = probabilities
    scored["priority_score"] = scored["incident_probability"] * scored["business_value"]
    selected = scored.sort_values(["date", "priority_score"], ascending=[True, False]).groupby("date").head(top_k)
    total_incidents = int(scored[TARGET].sum())
    captured_incidents = int(selected[TARGET].sum())
    total_incident_value = float((scored[TARGET] * scored["business_value"]).sum())
    captured_incident_value = float((selected[TARGET] * selected["business_value"]).sum())
    return {
        "top_k_per_day": int(top_k),
        "selected_total": int(len(selected)),
        "captured_incidents": captured_incidents,
        "total_incidents": total_incidents,
        "capture_rate": float(captured_incidents / total_incidents) if total_incidents else 0.0,
        "precision_at_k": float(captured_incidents / len(selected)) if len(selected) else 0.0,
        "captured_incident_value": captured_incident_value,
        "total_incident_value": total_incident_value,
        "value_capture_rate": float(captured_incident_value / total_incident_value) if total_incident_value else 0.0,
    }


def main() -> None:
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(output_dir)
    logger.info("Training final intervention prioritization model")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = add_historical_features(df)
    df["business_value"] = business_value(df)

    test_start = pd.Timestamp(TEST_START_DATE)
    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()
    X_train = train_df.drop(columns=[TARGET, "date", "server_id", "business_value"])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET, "date", "server_id", "business_value"])
    y_test = test_df[TARGET]
    traceability_test = test_df[["date", "server_id", TARGET, "business_value"]].copy()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.2, random_state=RANDOM_STATE)),
        ]
    )
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        **topk_metrics(traceability_test, probabilities, TOP_K),
    }
    logger.info("Metrics: %s", metrics)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_model_path = output_dir / "intervention_prioritization_hgb.pkl"
    joblib.dump(pipeline, run_model_path)
    joblib.dump(pipeline, ARTIFACT_PATH)
    metadata = {
        "name": "intervention_prioritization_hgb",
        "problem": "preventive intervention prioritization",
        "artifact_path": str(ARTIFACT_PATH.relative_to(ROOT_DIR)),
        "saved_with": "joblib",
        "model_type": "HistGradientBoostingClassifier",
        "ranking_formula": "priority_score = incident_probability * business_value",
        "business_value_formula": "monthly_spend_eur * support_plan_weight * hardware_weight * pressure_weight",
        "selection_metric": "value_capture_rate at top 50 per day",
        **metrics,
        "source_run": str(output_dir.relative_to(ROOT_DIR)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved model to %s", ARTIFACT_PATH.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
