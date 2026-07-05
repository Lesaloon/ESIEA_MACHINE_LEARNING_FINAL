from fastapi import FastAPI

from inference.predictor import predict_anomaly
from inference.schemas import AnomalyResponse, PredictionRequest


app = FastAPI(title="Anomaly Inference Service")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service_type": "anomaly-inference"}


@app.post("/predict-anomaly", response_model=AnomalyResponse)
def make_prediction(payload: PredictionRequest) -> AnomalyResponse:
    return predict_anomaly(payload)
