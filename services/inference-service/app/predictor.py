from typing import Any

import hashlib
import json

import pandas as pd

from app.model_loader import load_model, load_support_model
from app.schemas import IncidentFeatures, PredictionRequest, PredictionResponse, SupportForecastFeatures, SupportForecastResponse


MODEL_THRESHOLD = 0.21265613301105626

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

SUPPORT_HISTORY_COLUMNS = [
    "support_tickets",
    "network_latency_ms",
    "capacity_used_pct",
    "power_usage_mw",
    "avg_rack_temperature_c",
]

SUPPORT_RAW_FEATURE_COLUMNS = [
    "region",
    "scheduled_maintenance",
    "avg_rack_temperature_c",
    "power_usage_mw",
    "network_latency_ms",
    "capacity_used_pct",
    "day_of_week",
    "day_of_month",
    "days_since_start",
]

RAW_FEATURE_COLUMNS = [
    "server_type",
    "region",
    "os_family",
    "cpu_cores",
    "ram_gb",
    "disk_tb",
    "age_days",
    "has_gpu",
    "is_managed",
    "cpu_util_pct",
    "ram_util_pct",
    "disk_util_pct",
    "net_in_gb",
    "net_out_gb",
    "temperature_c",
    "backup_success",
    "scheduled_maintenance",
    "avg_rack_temperature_c",
    "power_usage_mw",
    "network_latency_ms",
    "support_tickets",
    "capacity_used_pct",
    "segment",
    "country",
    "contract_months",
    "support_plan",
    "tenure_days",
    "monthly_spend_eur",
    "day_of_week",
    "day_of_month",
    "days_since_start",
]


def _payload_to_features(payload: PredictionRequest) -> IncidentFeatures:
    data: dict[str, Any] = payload.inputs or payload.model_extra or {}
    return IncidentFeatures(**data)


def _payload_to_support_features(payload: PredictionRequest) -> SupportForecastFeatures:
    data: dict[str, Any] = payload.inputs or payload.model_extra or {}
    return SupportForecastFeatures(**data)


def _risk_level(probability: float) -> str:
    if probability >= 0.5:
        return "critical"
    if probability >= MODEL_THRESHOLD:
        return "high"
    if probability >= 0.1:
        return "medium"
    return "low"


def _input_hash(inputs: dict[str, Any]) -> str:
    payload = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_feature_frame(features: IncidentFeatures) -> pd.DataFrame:
    data = features.model_dump()
    observed_date = pd.to_datetime(data["date"])

    if data.get("day_of_week") is None:
        data["day_of_week"] = int(observed_date.dayofweek)
    if data.get("day_of_month") is None:
        data["day_of_month"] = int(observed_date.day)
    if data.get("days_since_start") is None:
        data["days_since_start"] = int((observed_date - pd.Timestamp("2026-01-01")).days)

    row = {column: data[column] for column in RAW_FEATURE_COLUMNS}

    for column in HISTORY_COLUMNS:
        # At online inference time we may only receive the current observation.
        # Use the current value as a conservative proxy for recent history instead
        # of sending missing values for the rolling features used by the model.
        row[f"{column}_lag1"] = row[column]
        row[f"{column}_diff1"] = 0
        row[f"{column}_rolling_mean_3"] = row[column]
        row[f"{column}_rolling_mean_7"] = row[column]
        row[f"{column}_rolling_max_7"] = row[column]

    row["cpu_ram_pressure"] = row["cpu_util_pct"] * row["ram_util_pct"] / 100
    row["thermal_pressure"] = row["temperature_c"] * row["cpu_util_pct"] / 100
    row["network_total_gb"] = row["net_in_gb"] + row["net_out_gb"]
    row["network_balance_gb"] = row["net_in_gb"] - row["net_out_gb"]
    row["utilization_pressure"] = (
        row["cpu_util_pct"] + row["ram_util_pct"] + row["disk_util_pct"] + row["capacity_used_pct"]
    ) / 4

    return pd.DataFrame([row])


def build_support_feature_frame(features: SupportForecastFeatures) -> pd.DataFrame:
    data = features.model_dump()
    observed_date = pd.to_datetime(data["date"])

    if data.get("day_of_week") is None:
        data["day_of_week"] = int(observed_date.dayofweek)
    if data.get("day_of_month") is None:
        data["day_of_month"] = int(observed_date.day)
    if data.get("days_since_start") is None:
        data["days_since_start"] = int((observed_date - pd.Timestamp("2026-01-01")).days)

    row = {column: data[column] for column in SUPPORT_RAW_FEATURE_COLUMNS}
    history_values = {
        "support_tickets": data["recent_support_tickets"],
        "network_latency_ms": data["network_latency_ms"],
        "capacity_used_pct": data["capacity_used_pct"],
        "power_usage_mw": data["power_usage_mw"],
        "avg_rack_temperature_c": data["avg_rack_temperature_c"],
    }

    for column in SUPPORT_HISTORY_COLUMNS:
        value = history_values[column]
        row[f"{column}_lag1"] = value
        row[f"{column}_lag7"] = value
        row[f"{column}_diff1"] = 0
        row[f"{column}_rolling_mean_3"] = value
        row[f"{column}_rolling_mean_7"] = value
        row[f"{column}_rolling_std_7"] = 0
        row[f"{column}_rolling_max_7"] = value

    row["infra_pressure"] = (
        row["capacity_used_pct"] + row["network_latency_ms"] + row["avg_rack_temperature_c"]
    ) / 3
    row["maintenance_latency_interaction"] = row["scheduled_maintenance"] * row["network_latency_ms"]
    row["capacity_power_interaction"] = row["capacity_used_pct"] * row["power_usage_mw"]

    return pd.DataFrame([row])


def predict(payload: PredictionRequest) -> PredictionResponse:
    model = load_model()
    features = _payload_to_features(payload)
    feature_frame = build_feature_frame(features)
    probability = float(model.predict_proba(feature_frame)[0, 1])
    prediction = int(probability >= MODEL_THRESHOLD)

    return PredictionResponse(
        prediction=prediction,
        incident_probability=probability,
        risk_level=_risk_level(probability),
        metadata={
            "model_loaded": True,
            "model_type": "RandomForestClassifier",
            "threshold": MODEL_THRESHOLD,
            "server_id": features.server_id,
            "date": features.date,
            "input_hash": _input_hash(features.model_dump()),
            "feature_count": int(feature_frame.shape[1]),
        },
    )


def predict_support(payload: PredictionRequest) -> SupportForecastResponse:
    model = load_support_model()
    features = _payload_to_support_features(payload)
    feature_frame = build_support_feature_frame(features)
    prediction = float(max(model.predict(feature_frame)[0], 0))

    return SupportForecastResponse(
        prediction=prediction,
        rounded_prediction=int(round(prediction)),
        metadata={
            "model_loaded": True,
            "model_type": "ExtraTreesRegressor",
            "region": features.region,
            "date": features.date,
            "input_hash": _input_hash(features.model_dump()),
            "feature_count": int(feature_frame.shape[1]),
        },
    )
