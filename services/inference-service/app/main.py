from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

from app.predictor import predict
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Incident Risk Inference Service")
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


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
