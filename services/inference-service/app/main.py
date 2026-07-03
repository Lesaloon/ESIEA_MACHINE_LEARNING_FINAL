from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

from app.predictor import predict, predict_segmentation, predict_support
from app.schemas import PredictionRequest, PredictionResponse, SegmentationResponse, SupportForecastResponse

app = FastAPI(title="ML Inference Service")
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.middleware("http")
async def add_no_cache_headers(request, call_next) -> Response:
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(payload: PredictionRequest) -> PredictionResponse:
    return predict(payload)


@app.post("/predict-support", response_model=SupportForecastResponse)
def make_support_prediction(payload: PredictionRequest) -> SupportForecastResponse:
    return predict_support(payload)


@app.post("/predict-segmentation", response_model=SegmentationResponse)
def make_segmentation_prediction(payload: PredictionRequest) -> SegmentationResponse:
    return predict_segmentation(payload)


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
