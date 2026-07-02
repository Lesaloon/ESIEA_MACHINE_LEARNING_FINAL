from functools import lru_cache
import json
import os
from pathlib import Path

import joblib


CONTAINER_INCIDENT_MODEL_PATH = Path("/models/artifacts/incident_random_forest_model.pkl")
CONTAINER_SUPPORT_MODEL_PATH = Path("/models/artifacts/support_extra_trees_model.pkl")
CONTAINER_SUPPORT_METADATA_PATH = Path("/models/metadata/support_extra_trees_model.json")


def resolve_incident_model_path() -> Path:
    if configured_path := os.getenv("MODEL_PATH"):
        return Path(configured_path)
    if CONTAINER_INCIDENT_MODEL_PATH.exists():
        return CONTAINER_INCIDENT_MODEL_PATH
    current_path = Path(__file__).resolve()
    if len(current_path.parents) > 3:
        return current_path.parents[3] / "models" / "artifacts" / "incident_random_forest_model.pkl"
    return CONTAINER_INCIDENT_MODEL_PATH


def resolve_support_model_path() -> Path:
    if configured_path := os.getenv("SUPPORT_MODEL_PATH"):
        return Path(configured_path)
    if CONTAINER_SUPPORT_MODEL_PATH.exists():
        return CONTAINER_SUPPORT_MODEL_PATH
    current_path = Path(__file__).resolve()
    if len(current_path.parents) > 3:
        return current_path.parents[3] / "models" / "artifacts" / "support_extra_trees_model.pkl"
    return CONTAINER_SUPPORT_MODEL_PATH


def resolve_support_metadata_path() -> Path:
    if configured_path := os.getenv("SUPPORT_MODEL_METADATA_PATH"):
        return Path(configured_path)
    if CONTAINER_SUPPORT_METADATA_PATH.exists():
        return CONTAINER_SUPPORT_METADATA_PATH
    current_path = Path(__file__).resolve()
    if len(current_path.parents) > 3:
        return current_path.parents[3] / "models" / "metadata" / "support_extra_trees_model.json"
    return CONTAINER_SUPPORT_METADATA_PATH


@lru_cache
def load_model() -> object:
    model_path = resolve_incident_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


@lru_cache
def load_support_model() -> object:
    model_path = resolve_support_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


@lru_cache
def load_support_metadata() -> dict[str, object]:
    metadata_path = resolve_support_metadata_path()
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))
