from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part6_intervention_prioritization"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "processed" / "incident_dataset.csv"
INCIDENT_MODEL_PATH = ROOT_DIR / "models" / "artifacts" / "incident_random_forest_model.pkl"
ANOMALY_MODEL_PATH = ROOT_DIR / "models" / "artifacts" / "anomaly_local_outlier_factor.pkl"
ANOMALY_METADATA_PATH = ROOT_DIR / "models" / "metadata" / "anomaly_local_outlier_factor.json"
TARGET = "incident_next_7d"
TEST_START_DATE = "2026-03-01"
TOP_K = 50

HISTORY_COLUMNS = ["cpu_util_pct", "ram_util_pct", "disk_util_pct", "temperature_c", "net_in_gb", "net_out_gb", "network_latency_ms", "support_tickets", "capacity_used_pct"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate business-rule intervention prioritization.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-start-date", default=TEST_START_DATE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("intervention_prioritization_rules")
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
    df["utilization_pressure"] = (df["cpu_util_pct"] + df["ram_util_pct"] + df["disk_util_pct"] + df["capacity_used_pct"]) / 4
    return df.sort_values("date").reset_index(drop=True)


def business_value(df: pd.DataFrame) -> pd.Series:
    support_plan_weight = df["support_plan"].map({"basic": 1.0, "standard": 1.2, "premium": 1.5, "critical": 2.0}).fillna(1.0)
    hardware_weight = 1 + 0.25 * df["has_gpu"] + 0.15 * df["is_managed"]
    pressure_weight = 1 + df["capacity_used_pct"].clip(0, 100) / 200
    return (df["monthly_spend_eur"].clip(lower=10) * support_plan_weight * hardware_weight * pressure_weight).astype(float)


def anomaly_multiplier(scores: np.ndarray, threshold: float) -> np.ndarray:
    margins = scores - threshold
    return np.select([margins >= 0.1, margins >= 0.03, margins >= 0], [1.6, 1.35, 1.15], default=1.0)


def temperature_multiplier(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["temperature_c"] >= 70, 1.25, np.where(df["temperature_c"] >= 60, 1.1, 1.0))


def topk_metrics(scored: pd.DataFrame, score_column: str, top_k: int) -> dict[str, float | int | str]:
    selected = scored.sort_values(["date", score_column], ascending=[True, False]).groupby("date").head(top_k)
    total_incidents = int(scored[TARGET].sum())
    captured_incidents = int(selected[TARGET].sum())
    total_incident_value = float((scored[TARGET] * scored["business_value"]).sum())
    captured_incident_value = float((selected[TARGET] * selected["business_value"]).sum())
    return {
        "rule": score_column,
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
    logger.info("Business-rule prioritization experiment started")

    df = pd.read_csv(args.data_path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = add_historical_features(df)
    df["business_value"] = business_value(df)
    test_df = df[df["date"] >= pd.Timestamp(args.test_start_date)].copy()

    incident_model = joblib.load(INCIDENT_MODEL_PATH)
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
    anomaly_metadata = json.loads(ANOMALY_METADATA_PATH.read_text(encoding="utf-8"))
    threshold = float(anomaly_metadata["threshold"])

    incident_features = test_df.drop(columns=[TARGET, "date", "server_id", "business_value"])
    anomaly_features = incident_features[[
        "server_type", "region", "os_family", "cpu_cores", "ram_gb", "disk_tb", "age_days", "has_gpu", "is_managed", "cpu_util_pct", "ram_util_pct", "disk_util_pct", "net_in_gb", "net_out_gb", "temperature_c", "backup_success", "scheduled_maintenance", "avg_rack_temperature_c", "power_usage_mw", "network_latency_ms", "support_tickets", "capacity_used_pct", "segment", "country", "contract_months", "support_plan", "tenure_days", "monthly_spend_eur", "day_of_week", "day_of_month", "days_since_start", "cpu_ram_pressure", "thermal_pressure", "network_total_gb", "network_balance_gb", "utilization_pressure"
    ]]

    scored = test_df[["date", "server_id", TARGET, "business_value", "temperature_c"]].copy()
    scored["incident_probability"] = incident_model.predict_proba(incident_features)[:, 1]
    scored["anomaly_score"] = -anomaly_model.decision_function(anomaly_features)
    scored["incident_only"] = scored["incident_probability"]
    scored["incident_value"] = scored["incident_probability"] * scored["business_value"]
    scored["incident_value_anomaly"] = scored["incident_value"] * anomaly_multiplier(scored["anomaly_score"].to_numpy(), threshold)
    scored["incident_value_anomaly_temperature"] = scored["incident_value_anomaly"] * temperature_multiplier(scored)
    scored.to_csv(run_dir / "prioritization_scores.csv", index=False)

    score_columns = ["incident_only", "incident_value", "incident_value_anomaly", "incident_value_anomaly_temperature"]
    metrics_df = pd.DataFrame([topk_metrics(scored, column, args.top_k) for column in score_columns]).sort_values("value_capture_rate", ascending=False)
    metrics_df.to_csv(run_dir / "rule_metrics.csv", index=False)
    summary = {
        "problem_type": "business_rule_intervention_prioritization",
        "selection_metric": "value_capture_rate at top 50 per day",
        "best_rule": str(metrics_df.iloc[0]["rule"]),
        "top_k_per_day": int(args.top_k),
        "best_metrics": metrics_df.iloc[0].to_dict(),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Rule ranking:\n%s", metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
