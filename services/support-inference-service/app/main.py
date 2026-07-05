from fastapi import FastAPI

from inference.predictor import predict_support
from inference.schemas import PredictionRequest, SupportForecastResponse


app = FastAPI(title="Support Forecast Inference Service")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service_type": "support-inference"}


@app.post("/predict-support", response_model=SupportForecastResponse)
def make_prediction(payload: PredictionRequest) -> SupportForecastResponse:
    return predict_support(payload)
