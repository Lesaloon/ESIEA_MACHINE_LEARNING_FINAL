from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part6_intervention_prioritization"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "processed" / "incident_dataset.csv"
TARGET = "incident_next_7d"
TEST_START_DATE = "2026-03-01"
RANDOM_STATE = 42
TOP_K = 50

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark preventive intervention prioritization models.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-start-date", default=TEST_START_DATE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("intervention_prioritization")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
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


def load_dataset(data_path: Path, logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = add_historical_features(df)
    df["business_value"] = business_value(df)
    logger.info("Dataset: %s", data_path.relative_to(ROOT_DIR))
    logger.info("Shape after features: %s", df.shape)
    logger.info("Date range: %s -> %s", df["date"].min().date(), df["date"].max().date())
    logger.info("Incident rate: %.4f", df[TARGET].mean())
    return df


def temporal_split(df: pd.DataFrame, test_start_date: str, logger: logging.Logger):
    test_start = pd.Timestamp(test_start_date)
    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()
    traceability_test = test_df[["date", "server_id", TARGET, "business_value"]].copy()
    drop_columns = [TARGET, "date", "server_id", "business_value"]
    X_train = train_df.drop(columns=drop_columns)
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=drop_columns)
    y_test = test_df[TARGET]
    logger.info("Train: %s | positives %.4f", X_train.shape, y_train.mean())
    logger.info("Test: %s | positives %.4f", X_test.shape, y_test.mean())
    return X_train, y_train, X_test, y_test, traceability_test


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features),
        ]
    )


def model_spaces() -> dict[str, object]:
    return {
        "random_forest": RandomForestClassifier(n_estimators=260, max_depth=18, min_samples_leaf=8, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=260, max_depth=18, min_samples_leaf=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.2, random_state=RANDOM_STATE),
    }


def topk_metrics(traceability: pd.DataFrame, probabilities: np.ndarray, top_k: int) -> dict[str, float | int]:
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
    args = parse_args()
    run_dir = EXPERIMENT_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(run_dir)
    logger.info("Intervention prioritization experiment started")
    logger.info("Run directory: %s", run_dir.relative_to(ROOT_DIR))
    df = load_dataset(args.data_path, logger)
    X_train, y_train, X_test, y_test, traceability_test = temporal_split(df, args.test_start_date, logger)

    results = []
    trained_models = {}
    for model_name, model in model_spaces().items():
        logger.info("Training %s", model_name)
        pipeline = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", model)])
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "model": model_name,
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            **topk_metrics(traceability_test, probabilities, args.top_k),
        }
        logger.info("%s metrics: %s", model_name, metrics)
        pd.DataFrame({**traceability_test, "incident_probability": probabilities}).to_csv(run_dir / f"predictions_test_{model_name}.csv", index=False)
        results.append(metrics)
        trained_models[model_name] = pipeline

    metrics_df = pd.DataFrame(results).sort_values(["value_capture_rate", "capture_rate", "average_precision"], ascending=[False, False, False])
    metrics_path = run_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    best_model_name = str(metrics_df.iloc[0]["model"])
    model_path = run_dir / f"best_prioritization_{best_model_name}.pkl"
    joblib.dump(trained_models[best_model_name], model_path)

    summary = {
        "problem_type": "preventive_intervention_prioritization",
        "selection_metric": "value_capture_rate at top 50 per day",
        "best_model": best_model_name,
        "best_model_path": str(model_path.relative_to(ROOT_DIR)),
        "metrics_path": str(metrics_path.relative_to(ROOT_DIR)),
        "top_k_per_day": int(args.top_k),
        "best_metrics": metrics_df.iloc[0].to_dict(),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Model ranking:\n%s", metrics_df.to_string(index=False))
    logger.info("Best model: %s", best_model_name)


if __name__ == "__main__":
    main()
