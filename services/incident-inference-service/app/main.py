from fastapi import FastAPI

from inference.predictor import predict
from inference.schemas import PredictionRequest, PredictionResponse


app = FastAPI(title="Incident Inference Service")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service_type": "incident-inference"}


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(payload: PredictionRequest) -> PredictionResponse:
    return predict(payload)
