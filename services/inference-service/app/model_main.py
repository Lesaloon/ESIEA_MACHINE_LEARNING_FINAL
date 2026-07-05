import os

from fastapi import FastAPI, HTTPException

from app.predictor import predict, predict_anomaly, predict_segmentation, predict_support
from app.schemas import AnomalyResponse, PredictionRequest, PredictionResponse, SegmentationResponse, SupportForecastResponse


SERVICE_TYPE = os.getenv("MODEL_SERVICE_TYPE", "incident")
app = FastAPI(title=f"{SERVICE_TYPE.title()} Model Service")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service_type": SERVICE_TYPE}


@app.post("/predict", response_model=PredictionResponse)
def incident_prediction(payload: PredictionRequest) -> PredictionResponse:
    if SERVICE_TYPE != "incident":
        raise HTTPException(status_code=404, detail="Endpoint not served by this model service")
    return predict(payload)


@app.post("/predict-support", response_model=SupportForecastResponse)
def support_prediction(payload: PredictionRequest) -> SupportForecastResponse:
    if SERVICE_TYPE != "support":
        raise HTTPException(status_code=404, detail="Endpoint not served by this model service")
    return predict_support(payload)


@app.post("/predict-segmentation", response_model=SegmentationResponse)
def segmentation_prediction(payload: PredictionRequest) -> SegmentationResponse:
    if SERVICE_TYPE != "segmentation":
        raise HTTPException(status_code=404, detail="Endpoint not served by this model service")
    return predict_segmentation(payload)


@app.post("/predict-anomaly", response_model=AnomalyResponse)
def anomaly_prediction(payload: PredictionRequest) -> AnomalyResponse:
    if SERVICE_TYPE != "anomaly":
        raise HTTPException(status_code=404, detail="Endpoint not served by this model service")
    return predict_anomaly(payload)
