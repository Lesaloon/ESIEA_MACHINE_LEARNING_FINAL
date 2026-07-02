from functools import lru_cache
import os
from pathlib import Path

import joblib


CONTAINER_MODEL_PATH = Path("/models/artifacts/incident_random_forest_model.pkl")


def resolve_model_path() -> Path:
    if configured_path := os.getenv("MODEL_PATH"):
        return Path(configured_path)
    if CONTAINER_MODEL_PATH.exists():
        return CONTAINER_MODEL_PATH
    current_path = Path(__file__).resolve()
    if len(current_path.parents) > 3:
        return current_path.parents[3] / "models" / "artifacts" / "incident_random_forest_model.pkl"
    return CONTAINER_MODEL_PATH


@lru_cache
def load_model() -> object:
    model_path = resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)
