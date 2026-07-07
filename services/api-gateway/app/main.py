import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException

from inference.schemas import PredictionRequest, PrioritizationRequest, PrioritizationResponse


app = FastAPI(title="ML API Gateway")

INCIDENT_SERVICE_URL = os.getenv("INCIDENT_SERVICE_URL", "http://incident-inference-service:8000")
SUPPORT_SERVICE_URL = os.getenv("SUPPORT_SERVICE_URL", "http://support-inference-service:8000")
SEGMENTATION_SERVICE_URL = os.getenv("SEGMENTATION_SERVICE_URL", "http://segmentation-inference-service:8000")
ANOMALY_SERVICE_URL = os.getenv("ANOMALY_SERVICE_URL", "http://anomaly-inference-service:8000")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service_type": "api-gateway"}


@app.post("/predict")
def make_prediction(payload: PredictionRequest) -> dict:
    return call_model_service(INCIDENT_SERVICE_URL, "/predict", payload.model_dump())


@app.post("/predict-support")
def make_support_prediction(payload: PredictionRequest) -> dict:
    return call_model_service(SUPPORT_SERVICE_URL, "/predict-support", payload.model_dump())


@app.post("/predict-segmentation")
def make_segmentation_prediction(payload: PredictionRequest) -> dict:
    return call_model_service(SEGMENTATION_SERVICE_URL, "/predict-segmentation", payload.model_dump())


@app.post("/predict-anomaly")
def make_anomaly_prediction(payload: PredictionRequest) -> dict:
    return call_model_service(ANOMALY_SERVICE_URL, "/predict-anomaly", payload.model_dump())


@app.post("/prioritize-interventions", response_model=PrioritizationResponse)
def make_prioritization(payload: PrioritizationRequest) -> PrioritizationResponse:
    recommendations = []
    for raw_input in payload.inputs:
        request_body = {"inputs": raw_input}
        incident = call_model_service(INCIDENT_SERVICE_URL, "/predict", request_body)
        anomaly = call_model_service(ANOMALY_SERVICE_URL, "/predict-anomaly", request_body)
        segmentation = call_model_service(SEGMENTATION_SERVICE_URL, "/predict-segmentation", request_body)
        incident_probability = float(incident["incident_probability"])
        value = business_value_for_row(raw_input)
        recommendations.append(
            {
                "server_id": raw_input.get("server_id"),
                "date": raw_input.get("date"),
                "incident_probability": incident_probability,
                "business_value": value,
                "priority_score": incident_probability * value,
                "is_anomaly": anomaly.get("is_anomaly"),
                "anomaly_score": anomaly.get("anomaly_score"),
                "anomaly_severity": anomaly.get("severity"),
                "segment_cluster": segmentation.get("cluster"),
                "segment_profile": segmentation.get("profile_name"),
            }
        )

    recommendations = sorted(recommendations, key=lambda item: item["priority_score"], reverse=True)[: max(int(payload.top_n), 0)]
    for rank, item in enumerate(recommendations, start=1):
        item["rank"] = rank

    return PrioritizationResponse(
        recommendations=recommendations,
        metadata={
            "model_loaded": True,
            "model_type": "gateway_business_rules_existing_models",
            "ranking_formula": "priority_score = incident_probability * business_value",
            "requested_top_n": payload.top_n,
            "input_count": len(payload.inputs),
            "returned_count": len(recommendations),
            "called_services": ["incident", "anomaly", "segmentation"],
        },
    )


def call_model_service(base_url: str, path: str, payload: dict) -> dict:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"Model service unavailable: {base_url}") from exc


def business_value_for_row(data: dict) -> float:
    """
    Estimate the business value of a customer/account row.

    The score is based on:
    - monthly spend
    - support plan level
    - whether the customer uses GPU hardware
    - whether the customer is managed
    - how much capacity they are using
    """

    # 1. Higher support plans increase the value
    support_plan = str(data.get("support_plan", "basic")).lower()

    support_plan_weights = {
        "basic": 1.0,
        "standard": 1.2,
        "premium": 1.5,
        "critical": 2.0,
    }

    support_plan_weight = support_plan_weights.get(support_plan, 1.0)

    # 2. Hardware / account setup adjustments
    has_gpu = int(data.get("has_gpu", 0))
    is_managed = int(data.get("is_managed", 0))

    hardware_weight = (
        1
        + 0.25 * has_gpu
        + 0.15 * is_managed
    )

    # 3. Capacity pressure adjustment
    capacity_used_pct = float(data.get("capacity_used_pct", 0))

    # Keep capacity between 0% and 100%
    capacity_used_pct = min(max(capacity_used_pct, 0), 100)

    # At 100% capacity, this adds a 50% boost
    pressure_weight = 1 + capacity_used_pct / 200

    # 4. Monthly spend, with a minimum value of €10
    monthly_spend_eur = float(data.get("monthly_spend_eur", 10))
    monthly_spend_eur = max(monthly_spend_eur, 10)

    # 5. Final business value score
    business_value = (
        monthly_spend_eur
        * support_plan_weight
        * hardware_weight
        * pressure_weight
    )

    return float(business_value)
