from typing import Any

import hashlib
import json
from numbers import Number

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from inference.model_loader import (
    load_anomaly_metadata,
    load_anomaly_model,
    load_incident_metadata,
    load_model,
    load_segmentation_metadata,
    load_segmentation_model,
    load_support_metadata,
    load_support_model,
)
from inference.schemas import (
    AnomalyResponse,
    IncidentFeatures,
    PredictionRequest,
    PredictionResponse,
    PrioritizationRequest,
    PrioritizationResponse,
    SegmentationFeatures,
    SegmentationResponse,
    SupportForecastFeatures,
    SupportForecastResponse,
)


MODEL_THRESHOLD = 0.21265613301105626
_EXPLAINER_CACHE: dict[int, Any] = {}

HISTORY_COLUMNS = [
    "cpu_util_pct",
    "ram_util_pct",
    "disk_util_pct",
    "temperature_c",
    "net_in_gb",
    "net_out_gb",
    "network_latency_ms_prev_day",
    "support_tickets_prev_day",
    "capacity_used_pct_prev_day",
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
    "support_tickets",
    "day_of_week",
    "day_of_month",
    "days_since_start",
]

SEGMENTATION_CATEGORICAL_COLUMNS = ["server_type", "region", "os_family", "segment", "country", "support_plan"]

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

INCIDENT_RAW_FEATURE_COLUMNS = [
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
    "scheduled_maintenance_prev_day",
    "avg_rack_temperature_c_prev_day",
    "power_usage_mw_prev_day",
    "network_latency_ms_prev_day",
    "support_tickets_prev_day",
    "capacity_used_pct_prev_day",
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

INCIDENT_REGION_INPUT_MAP = {
    "scheduled_maintenance_prev_day": "scheduled_maintenance",
    "avg_rack_temperature_c_prev_day": "avg_rack_temperature_c",
    "power_usage_mw_prev_day": "power_usage_mw",
    "network_latency_ms_prev_day": "network_latency_ms",
    "support_tickets_prev_day": "support_tickets",
    "capacity_used_pct_prev_day": "capacity_used_pct",
}


def _payload_to_features(payload: PredictionRequest) -> IncidentFeatures:
    data: dict[str, Any] = payload.inputs or payload.model_extra or {}
    return IncidentFeatures(**data)


def _payload_to_support_features(payload: PredictionRequest) -> SupportForecastFeatures:
    data: dict[str, Any] = payload.inputs or payload.model_extra or {}
    return SupportForecastFeatures(**data)


def _payload_to_segmentation_features(payload: PredictionRequest) -> SegmentationFeatures:
    data: dict[str, Any] = payload.inputs or payload.model_extra or {}
    return SegmentationFeatures(**data)


def _payload_to_anomaly_features(payload: PredictionRequest) -> IncidentFeatures:
    data: dict[str, Any] = payload.inputs or payload.model_extra or {}
    return IncidentFeatures(**data)


def _risk_level(probability: float, threshold: float = MODEL_THRESHOLD) -> str:
    if probability >= 0.5:
        return "critical"
    if probability >= threshold:
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

    row = {}
    for column in INCIDENT_RAW_FEATURE_COLUMNS:
        row[column] = data[INCIDENT_REGION_INPUT_MAP.get(column, column)]

    for raw_column in INCIDENT_REGION_INPUT_MAP.values():
        row[raw_column] = data[raw_column]

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
        row["cpu_util_pct"] + row["ram_util_pct"] + row["disk_util_pct"] + row["capacity_used_pct_prev_day"]
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

    data["support_tickets"] = data["recent_support_tickets"]
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


def build_segmentation_feature_frame(features: SegmentationFeatures) -> pd.DataFrame:
    data = features.model_dump()
    observation_count = max(int(data["observation_count"]), 1)

    row: dict[str, Any] = {}
    aggregate_specs = {
        "cpu_util_pct": ["mean", "max", "std"],
        "ram_util_pct": ["mean", "max", "std"],
        "disk_util_pct": ["mean", "max", "std"],
        "net_in_gb": ["mean", "max", "sum"],
        "net_out_gb": ["mean", "max", "sum"],
        "temperature_c": ["mean", "max", "std"],
        "network_latency_ms": ["mean", "max", "std"],
        "capacity_used_pct": ["mean", "max", "std"],
    }
    for column, stats in aggregate_specs.items():
        for stat in stats:
            if stat == "std":
                row[f"{column}_{stat}"] = 0
            elif stat == "sum":
                row[f"{column}_{stat}"] = data[column] * observation_count
            else:
                row[f"{column}_{stat}"] = data[column]

    row["backup_success_mean"] = data["backup_success"]
    row["backup_success_min"] = data["backup_success"]
    row["scheduled_maintenance_mean"] = data["scheduled_maintenance"]
    row["avg_rack_temperature_c_mean"] = data["avg_rack_temperature_c"]
    row["avg_rack_temperature_c_max"] = data["avg_rack_temperature_c"]
    row["power_usage_mw_mean"] = data["power_usage_mw"]
    row["power_usage_mw_max"] = data["power_usage_mw"]
    row["cpu_cores_first"] = data["cpu_cores"]
    row["ram_gb_first"] = data["ram_gb"]
    row["disk_tb_first"] = data["disk_tb"]
    row["age_days_first"] = data["age_days"]
    row["has_gpu_first"] = data["has_gpu"]
    row["is_managed_first"] = data["is_managed"]
    row["contract_months_first"] = data["contract_months"]
    row["tenure_days_first"] = data["tenure_days"]
    row["monthly_spend_eur_first"] = data["monthly_spend_eur"]
    row["observation_count"] = observation_count

    for column in SEGMENTATION_CATEGORICAL_COLUMNS:
        row[column] = data[column]

    row["utilization_pressure_mean"] = (
        row["cpu_util_pct_mean"] + row["ram_util_pct_mean"] + row["disk_util_pct_mean"] + row["capacity_used_pct_mean"]
    ) / 4
    row["network_total_gb_sum"] = row["net_in_gb_sum"] + row["net_out_gb_sum"]
    row["thermal_cpu_pressure"] = row["temperature_c_mean"] * row["cpu_util_pct_mean"] / 100
    row["backup_failure_rate"] = 1 - row["backup_success_mean"]
    return pd.DataFrame([row])


def build_anomaly_feature_frame(features: IncidentFeatures) -> pd.DataFrame:
    data = features.model_dump()
    observed_date = pd.to_datetime(data["date"])
    if data.get("day_of_week") is None:
        data["day_of_week"] = int(observed_date.dayofweek)
    if data.get("day_of_month") is None:
        data["day_of_month"] = int(observed_date.day)
    if data.get("days_since_start") is None:
        data["days_since_start"] = int((observed_date - pd.Timestamp("2026-01-01")).days)

    row = {column: data[column] for column in RAW_FEATURE_COLUMNS}
    row["cpu_ram_pressure"] = row["cpu_util_pct"] * row["ram_util_pct"] / 100
    row["thermal_pressure"] = row["temperature_c"] * row["cpu_util_pct"] / 100
    row["network_total_gb"] = row["net_in_gb"] + row["net_out_gb"]
    row["network_balance_gb"] = row["net_in_gb"] - row["net_out_gb"]
    row["utilization_pressure"] = (
        row["cpu_util_pct"] + row["ram_util_pct"] + row["disk_util_pct"] + row["capacity_used_pct"]
    ) / 4
    return pd.DataFrame([row])


def anomaly_severity(score: float, threshold: float) -> str:
    margin = score - threshold
    if margin >= 0.1:
        return "critical"
    if margin >= 0.03:
        return "high"
    if margin >= 0:
        return "medium"
    return "low"


def score_anomaly_frame(model: Pipeline, frame: pd.DataFrame) -> float:
    return float(-model.decision_function(frame)[0])


def feature_direction(value: Any, reference: Any) -> str:
    if isinstance(value, Number) and isinstance(reference, Number):
        if value > reference:
            return "above_normal"
        if value < reference:
            return "below_normal"
        return "near_normal"
    if value == reference:
        return "usual_category"
    return "unusual_category"


def readable_feature_name(feature: str) -> str:
    clean_feature = feature.split("__", 1)[-1]
    names = {
        "cpu_util_pct": "CPU usage",
        "ram_util_pct": "RAM usage",
        "disk_util_pct": "disk usage",
        "temperature_c": "server temperature",
        "net_in_gb": "incoming network traffic",
        "net_out_gb": "outgoing network traffic",
        "network_total_gb": "total network traffic",
        "network_balance_gb": "network traffic imbalance",
        "capacity_used_pct": "capacity usage",
        "support_tickets": "support tickets",
        "scheduled_maintenance": "scheduled maintenance",
        "thermal_pressure": "thermal pressure",
        "cpu_ram_pressure": "CPU/RAM pressure",
        "utilization_pressure": "global utilization pressure",
        "power_usage_mw": "power usage",
        "network_latency_ms": "network latency",
        "is_managed": "managed server flag",
        "has_gpu": "GPU flag",
        "contract_months": "contract duration",
        "monthly_spend_eur": "monthly spend",
    }
    return names.get(clean_feature, clean_feature.replace("_", " "))


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _feature_names_after_transforms(model: Pipeline) -> list[str]:
    names = list(model.named_steps["preprocessor"].get_feature_names_out())
    if "variance_filter" in model.named_steps:
        mask = model.named_steps["variance_filter"].get_support()
        names = [name for name, keep in zip(names, mask) if keep]
    if "selector" in model.named_steps:
        mask = model.named_steps["selector"].get_support()
        names = [name for name, keep in zip(names, mask) if keep]
    return names


def _shap_explainer(estimator: object) -> Any | None:
    cache_key = id(estimator)
    if cache_key in _EXPLAINER_CACHE:
        return _EXPLAINER_CACHE[cache_key]
    try:
        import shap
    except ImportError:
        return None
    explainer = shap.TreeExplainer(estimator)
    _EXPLAINER_CACHE[cache_key] = explainer
    return explainer


def _select_shap_values(raw_values: Any, class_index: int | None = None) -> np.ndarray:
    if isinstance(raw_values, list):
        selected = raw_values[class_index or 0]
        return np.asarray(selected)[0]

    values = np.asarray(raw_values)
    if values.ndim == 3:
        return values[0, :, class_index or 0]
    if values.ndim == 2:
        return values[0]
    return values


def explain_tree_pipeline(
    model: Pipeline,
    feature_frame: pd.DataFrame,
    *,
    class_index: int | None = None,
    top_n: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    transformed = model[:-1].transform(feature_frame)
    estimator = model.named_steps["model"]
    explainer = _shap_explainer(estimator)
    if explainer is None:
        return explain_by_permutation(model, feature_frame, class_index=class_index, top_n=top_n)

    feature_names = _feature_names_after_transforms(model)
    shap_values = _select_shap_values(explainer.shap_values(transformed), class_index=class_index)
    dense_values = np.asarray(transformed[0]).ravel()
    explanations = []
    for index, impact in enumerate(shap_values):
        feature = feature_names[index] if index < len(feature_names) else f"feature_{index}"
        explanations.append(
            {
                "feature": feature,
                "label": readable_feature_name(feature),
                "value": _json_value(dense_values[index]) if index < len(dense_values) else None,
                "impact": float(impact),
                "abs_impact": float(abs(impact)),
                "direction": "increases_prediction" if impact >= 0 else "decreases_prediction",
                "method": "shap",
            }
        )
    return sorted(explanations, key=lambda item: item["abs_impact"], reverse=True)[:top_n], ""


def explain_by_permutation(
    model: Pipeline,
    feature_frame: pd.DataFrame,
    *,
    class_index: int | None = None,
    top_n: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    if class_index is None:
        base_prediction = float(model.predict(feature_frame)[0])
    else:
        base_prediction = float(model.predict_proba(feature_frame)[0, class_index])

    explanations = []
    numeric_columns = feature_frame.select_dtypes(include="number").columns.tolist()
    for feature in numeric_columns:
        value = feature_frame.loc[0, feature]
        perturbed = feature_frame.copy()
        perturbed.loc[0, feature] = 0
        if class_index is None:
            new_prediction = float(model.predict(perturbed)[0])
        else:
            new_prediction = float(model.predict_proba(perturbed)[0, class_index])
        impact = base_prediction - new_prediction
        explanations.append(
            {
                "feature": feature,
                "label": readable_feature_name(feature),
                "value": _json_value(value),
                "reference": 0,
                "impact": float(impact),
                "abs_impact": float(abs(impact)),
                "direction": "increases_prediction" if impact >= 0 else "decreases_prediction",
                "method": "permutation_fallback",
            }
        )
    return sorted(explanations, key=lambda item: item["abs_impact"], reverse=True)[:top_n], ""


def incident_human_explanation(explanations: list[dict[str, Any]], probability: float) -> str:
    if not explanations:
        return "Aucun facteur dominant n'a ete identifie; le risque vient de la combinaison globale des signaux."
    drivers = ", ".join(item["label"] for item in explanations[:3])
    return f"Le modele estime un risque incident de {probability * 100:.2f} %. Les facteurs qui expliquent le plus ce score sont: {drivers}."


def support_human_explanation(explanations: list[dict[str, Any]], prediction: float) -> str:
    if not explanations:
        return "Aucun facteur dominant n'a ete identifie; la prevision vient de la combinaison globale des signaux regionaux."
    drivers = ", ".join(item["label"] for item in explanations[:3])
    return f"La prevision est de {prediction:.2f} tickets. Les facteurs les plus influents sont: {drivers}."


def segmentation_explanations(profile: dict[str, Any], probabilities: dict[str, float]) -> tuple[list[dict[str, Any]], str]:
    driver = profile.get("profile_driver") or "aucun ecart dominant"
    profile_name = str(profile.get("profile_name", "profil inconnu"))
    cluster = profile.get("cluster")
    confidence = probabilities.get(str(cluster)) if cluster is not None else None
    explanations = [
        {
            "feature": "cluster_profile",
            "label": "profil de cluster",
            "value": profile_name,
            "impact": float(confidence) if isinstance(confidence, Number) else 0.0,
            "abs_impact": float(confidence) if isinstance(confidence, Number) else 0.0,
            "direction": "assigned_profile",
            "method": "cluster_profile",
        },
        {
            "feature": "profile_driver",
            "label": "raison du profil",
            "value": driver,
            "impact": 0.0,
            "abs_impact": 0.0,
            "direction": "descriptive",
            "method": "cluster_profile",
        },
    ]
    confidence_text = f" avec une probabilite de cluster de {confidence * 100:.1f} %" if isinstance(confidence, Number) else ""
    human_explanation = f"Ce serveur est classe dans le profil '{profile_name}'{confidence_text}. Raison principale: {driver}."
    return explanations, human_explanation


def explain_anomaly(model: Pipeline, feature_frame: pd.DataFrame, metadata: dict[str, object], base_score: float) -> tuple[list[dict[str, Any]], str]:
    references = metadata.get("reference_values", {})
    numeric_refs = references.get("numeric", {}) if isinstance(references, dict) else {}
    categorical_refs = references.get("categorical", {}) if isinstance(references, dict) else {}

    explanations = []
    for feature in feature_frame.columns:
        if feature in numeric_refs:
            reference = numeric_refs[feature].get("median")
        elif feature in categorical_refs:
            reference = categorical_refs[feature]
        else:
            continue

        current_value = feature_frame.loc[0, feature]
        perturbed = feature_frame.copy()
        perturbed.loc[0, feature] = reference
        perturbed_score = score_anomaly_frame(model, perturbed)
        impact = base_score - perturbed_score
        if impact <= 0:
            continue

        explanations.append(
            {
                "feature": feature,
                "label": readable_feature_name(feature),
                "value": current_value.item() if hasattr(current_value, "item") else current_value,
                "reference": reference,
                "direction": feature_direction(current_value, reference),
                "impact": float(impact),
                "abs_impact": float(abs(impact)),
                "method": "reference_perturbation",
            }
        )

    top_explanations = sorted(explanations, key=lambda item: item["impact"], reverse=True)[:5]
    if top_explanations:
        reasons = ", ".join(item["label"] for item in top_explanations[:3])
        human_explanation = f"This server is flagged mainly because these signals differ from normal behavior: {reasons}."
    else:
        human_explanation = "No single feature dominates this anomaly score; the alert comes from the combined server profile."
    return top_explanations, human_explanation


def predict(payload: PredictionRequest) -> PredictionResponse:
    model = load_model()
    model_metadata = load_incident_metadata()
    features = _payload_to_features(payload)
    feature_frame = build_feature_frame(features)
    threshold = float(model_metadata.get("threshold", MODEL_THRESHOLD))
    probability = float(model.predict_proba(feature_frame)[0, 1])
    prediction = int(probability >= threshold)
    top_explanations, _ = explain_tree_pipeline(model, feature_frame, class_index=1)

    return PredictionResponse(
        prediction=prediction,
        incident_probability=probability,
        risk_level=_risk_level(probability, threshold),
        top_explanations=top_explanations,
        human_explanation=incident_human_explanation(top_explanations, probability),
        metadata={
            "model_loaded": True,
            "model_type": model_metadata.get("model_type", "RandomForestClassifier"),
            "explanation_method": top_explanations[0].get("method") if top_explanations else "none",
            "threshold": threshold,
            "server_id": features.server_id,
            "date": features.date,
            "input_hash": _input_hash(features.model_dump()),
            "feature_count": int(feature_frame.shape[1]),
        },
    )


def predict_support(payload: PredictionRequest) -> SupportForecastResponse:
    model = load_support_model()
    model_metadata = load_support_metadata()
    features = _payload_to_support_features(payload)
    feature_frame = build_support_feature_frame(features)
    prediction = float(max(model.predict(feature_frame)[0], 0))
    error_margin = model_metadata.get("mae")
    top_explanations, _ = explain_tree_pipeline(model, feature_frame)

    return SupportForecastResponse(
        prediction=prediction,
        rounded_prediction=int(round(prediction)),
        top_explanations=top_explanations,
        human_explanation=support_human_explanation(top_explanations, prediction),
        metadata={
            "model_loaded": True,
            "model_type": model_metadata.get("model_type", "ExtraTreesRegressor"),
            "explanation_method": top_explanations[0].get("method") if top_explanations else "none",
            "region": features.region,
            "date": features.date,
            "error_margin_mae": error_margin,
            "expected_range_low": max(prediction - error_margin, 0) if isinstance(error_margin, int | float) else None,
            "expected_range_high": prediction + error_margin if isinstance(error_margin, int | float) else None,
            "rmse": model_metadata.get("rmse"),
            "r2": model_metadata.get("r2"),
            "selection_metric": model_metadata.get("selection_metric"),
            "input_hash": _input_hash(features.model_dump()),
            "feature_count": int(feature_frame.shape[1]),
        },
    )


def predict_segmentation(payload: PredictionRequest) -> SegmentationResponse:
    model = load_segmentation_model()
    metadata = load_segmentation_metadata()
    features = _payload_to_segmentation_features(payload)
    feature_frame = build_segmentation_feature_frame(features)
    cluster = int(model.predict(feature_frame)[0])
    profiles = {int(profile["cluster"]): profile for profile in metadata.get("profiles", [])}
    profile = profiles.get(cluster, {})
    probabilities = {}
    if hasattr(model.named_steps["model"], "predict_proba"):
        probabilities = {
            str(index): float(value)
            for index, value in enumerate(model.predict_proba(feature_frame)[0])
        }
    top_explanations, human_explanation = segmentation_explanations(profile, probabilities)

    return SegmentationResponse(
        cluster=cluster,
        profile_name=str(profile.get("profile_name", "profil inconnu")),
        profile_driver=profile.get("profile_driver"),
        probabilities=probabilities,
        top_explanations=top_explanations,
        human_explanation=human_explanation,
        metadata={
            "model_loaded": True,
            "model_type": metadata.get("model_type", "GaussianMixture"),
            "explanation_method": "cluster_profile",
            "n_clusters": metadata.get("n_clusters"),
            "silhouette": metadata.get("silhouette"),
            "server_id": features.server_id,
            "input_hash": _input_hash(features.model_dump()),
            "feature_count": int(feature_frame.shape[1]),
        },
    )


def predict_anomaly(payload: PredictionRequest) -> AnomalyResponse:
    model = load_anomaly_model()
    metadata = load_anomaly_metadata()
    features = _payload_to_anomaly_features(payload)
    feature_frame = build_anomaly_feature_frame(features)
    score = score_anomaly_frame(model, feature_frame)
    threshold = float(metadata.get("threshold", 0.0))
    prediction = int(score >= threshold)
    top_explanations, human_explanation = explain_anomaly(model, feature_frame, metadata, score)

    return AnomalyResponse(
        prediction=prediction,
        is_anomaly=bool(prediction),
        anomaly_score=score,
        threshold=threshold,
        severity=anomaly_severity(score, threshold),
        top_explanations=top_explanations,
        human_explanation=human_explanation,
        metadata={
            "model_loaded": True,
            "model_type": metadata.get("model_type", "LocalOutlierFactor"),
            "explanation_method": "reference_perturbation",
            "server_id": features.server_id,
            "date": features.date,
            "average_precision": metadata.get("average_precision"),
            "recall": metadata.get("recall"),
            "precision": metadata.get("precision"),
            "input_hash": _input_hash(features.model_dump()),
            "feature_count": int(feature_frame.shape[1]),
        },
    )


def business_value_for_row(data: dict[str, Any]) -> float:
    support_plan_weight = {"basic": 1.0, "standard": 1.2, "premium": 1.5, "critical": 2.0}.get(str(data.get("support_plan")), 1.0)
    hardware_weight = 1 + 0.25 * float(data.get("has_gpu", 0)) + 0.15 * float(data.get("is_managed", 0))
    pressure_weight = 1 + min(max(float(data.get("capacity_used_pct", 0)), 0), 100) / 200
    return float(max(float(data.get("monthly_spend_eur", 10)), 10) * support_plan_weight * hardware_weight * pressure_weight)


def anomaly_multiplier(score: float, threshold: float) -> float:
    margin = score - threshold
    if margin >= 0.1:
        return 1.6
    if margin >= 0.03:
        return 1.35
    if margin >= 0:
        return 1.15
    return 1.0


def segmentation_multiplier(profile_name: str) -> float:
    if "temperature" in profile_name:
        return 1.25
    if "stockage" in profile_name:
        return 1.15
    return 1.0


def predict_prioritization(payload: PrioritizationRequest) -> PrioritizationResponse:
    incident_model = load_model()
    anomaly_model = load_anomaly_model()
    anomaly_metadata = load_anomaly_metadata()
    segmentation_model = load_segmentation_model()
    segmentation_metadata = load_segmentation_metadata()
    segmentation_profiles = {int(profile["cluster"]): profile for profile in segmentation_metadata.get("profiles", [])}
    rows = []

    for raw_input in payload.inputs:
        features = IncidentFeatures(**raw_input)
        dumped = features.model_dump()
        incident_frame = build_feature_frame(features)
        anomaly_frame = build_anomaly_feature_frame(features)
        segmentation_features = SegmentationFeatures(**{**dumped, "observation_count": raw_input.get("observation_count", 1)})
        segmentation_frame = build_segmentation_feature_frame(segmentation_features)

        incident_probability = float(incident_model.predict_proba(incident_frame)[0, 1])
        anomaly_score_value = score_anomaly_frame(anomaly_model, anomaly_frame)
        anomaly_threshold = float(anomaly_metadata.get("threshold", 0.0))
        anomaly_factor = anomaly_multiplier(anomaly_score_value, anomaly_threshold)
        segment_cluster = int(segmentation_model.predict(segmentation_frame)[0])
        segment_profile = segmentation_profiles.get(segment_cluster, {})
        segment_name = str(segment_profile.get("profile_name", "profil inconnu"))
        segment_factor = segmentation_multiplier(segment_name)
        value = business_value_for_row(dumped)
        priority_score = incident_probability * value

        rows.append(
            {
                "server_id": features.server_id,
                "date": features.date,
                "incident_probability": incident_probability,
                "business_value": value,
                "anomaly_score": anomaly_score_value,
                "is_anomaly": anomaly_score_value >= anomaly_threshold,
                "anomaly_context_multiplier": anomaly_factor,
                "segment_cluster": segment_cluster,
                "segment_profile": segment_name,
                "segmentation_context_multiplier": segment_factor,
                "priority_score": float(priority_score),
            }
        )

    if not rows:
        return PrioritizationResponse(recommendations=[], metadata={"model_loaded": True, "input_count": 0})

    recommendations = sorted(rows, key=lambda item: item["priority_score"], reverse=True)[: max(int(payload.top_n), 0)]
    for rank, item in enumerate(recommendations, start=1):
        item["rank"] = rank

    return PrioritizationResponse(
        recommendations=recommendations,
        metadata={
            "model_loaded": True,
            "model_type": "business_rules_existing_models",
            "ranking_formula": "priority_score = incident_probability * business_value",
            "requested_top_n": payload.top_n,
            "input_count": len(payload.inputs),
            "returned_count": len(recommendations),
            "uses_models": ["incident_random_forest"],
            "context_models": ["anomaly_local_outlier_factor", "server_segmentation_gaussian_mixture"],
        },
    )
