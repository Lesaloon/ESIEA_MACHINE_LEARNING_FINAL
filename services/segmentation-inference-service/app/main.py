from fastapi import FastAPI

from inference.predictor import predict_segmentation
from inference.schemas import PredictionRequest, SegmentationResponse


app = FastAPI(title="Segmentation Inference Service")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service_type": "segmentation-inference"}


@app.post("/predict-segmentation", response_model=SegmentationResponse)
def make_prediction(payload: PredictionRequest) -> SegmentationResponse:
    return predict_segmentation(payload)
