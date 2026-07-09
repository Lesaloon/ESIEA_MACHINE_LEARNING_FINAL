from functools import lru_cache
import json
import os
from pathlib import Path

import joblib


CONTAINER_INCIDENT_MODEL_PATH = Path("/models/artifacts/incident_gradient_boosting_model.pkl")
CONTAINER_INCIDENT_METADATA_PATH = Path("/models/metadata/incident_gradient_boosting_model.json")
CONTAINER_SUPPORT_MODEL_PATH = Path("/models/artifacts/support_extra_trees_model.pkl")
CONTAINER_SUPPORT_METADATA_PATH = Path("/models/metadata/support_extra_trees_model.json")
CONTAINER_SEGMENTATION_MODEL_PATH = Path("/models/artifacts/server_segmentation_kmeans.pkl")
CONTAINER_SEGMENTATION_METADATA_PATH = Path("/models/metadata/server_segmentation_kmeans.json")
CONTAINER_ANOMALY_MODEL_PATH = Path("/models/artifacts/anomaly_local_outlier_factor.pkl")
CONTAINER_ANOMALY_METADATA_PATH = Path("/models/metadata/anomaly_local_outlier_factor.json")


def resolve_incident_model_path() -> Path:
    if configured_path := os.getenv("MODEL_PATH"):
        return Path(configured_path)
    if CONTAINER_INCIDENT_MODEL_PATH.exists():
        return CONTAINER_INCIDENT_MODEL_PATH
    current_path = Path(__file__).resolve()
    if len(current_path.parents) > 3:
        return current_path.parents[3] / "models" / "artifacts" / "incident_gradient_boosting_model.pkl"
    return CONTAINER_INCIDENT_MODEL_PATH


def resolve_incident_metadata_path() -> Path:
    if configured_path := os.getenv("INCIDENT_MODEL_METADATA_PATH"):
        return Path(configured_path)
    if CONTAINER_INCIDENT_METADATA_PATH.exists():
        return CONTAINER_INCIDENT_METADATA_PATH
    current_path = Path(__file__).resolve()
    if len(current_path.parents) > 3:
        return current_path.parents[3] / "models" / "metadata" / "incident_gradient_boosting_model.json"
    return CONTAINER_INCIDENT_METADATA_PATH


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


def resolve_segmentation_model_path() -> Path:
    if configured_path := os.getenv("SEGMENTATION_MODEL_PATH"):
        return Path(configured_path)
    if CONTAINER_SEGMENTATION_MODEL_PATH.exists():
        return CONTAINER_SEGMENTATION_MODEL_PATH
    current_path = Path(__file__).resolve()
    if len(current_path.parents) > 3:
        return current_path.parents[3] / "models" / "artifacts" / "server_segmentation_kmeans.pkl"
    return CONTAINER_SEGMENTATION_MODEL_PATH


def resolve_segmentation_metadata_path() -> Path:
    if configured_path := os.getenv("SEGMENTATION_MODEL_METADATA_PATH"):
        return Path(configured_path)
    if CONTAINER_SEGMENTATION_METADATA_PATH.exists():
        return CONTAINER_SEGMENTATION_METADATA_PATH
    current_path = Path(__file__).resolve()
    if len(current_path.parents) > 3:
        return current_path.parents[3] / "models" / "metadata" / "server_segmentation_kmeans.json"
    return CONTAINER_SEGMENTATION_METADATA_PATH


def resolve_anomaly_model_path() -> Path:
    if configured_path := os.getenv("ANOMALY_MODEL_PATH"):
        return Path(configured_path)
    if CONTAINER_ANOMALY_MODEL_PATH.exists():
        return CONTAINER_ANOMALY_MODEL_PATH
    current_path = Path(__file__).resolve()
    if len(current_path.parents) > 3:
        return current_path.parents[3] / "models" / "artifacts" / "anomaly_local_outlier_factor.pkl"
    return CONTAINER_ANOMALY_MODEL_PATH


def resolve_anomaly_metadata_path() -> Path:
    if configured_path := os.getenv("ANOMALY_MODEL_METADATA_PATH"):
        return Path(configured_path)
    if CONTAINER_ANOMALY_METADATA_PATH.exists():
        return CONTAINER_ANOMALY_METADATA_PATH
    current_path = Path(__file__).resolve()
    if len(current_path.parents) > 3:
        return current_path.parents[3] / "models" / "metadata" / "anomaly_local_outlier_factor.json"
    return CONTAINER_ANOMALY_METADATA_PATH




@lru_cache
def load_model() -> object:
    model_path = resolve_incident_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


@lru_cache
def load_incident_metadata() -> dict[str, object]:
    metadata_path = resolve_incident_metadata_path()
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


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


@lru_cache
def load_segmentation_model() -> object:
    model_path = resolve_segmentation_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


@lru_cache
def load_segmentation_metadata() -> dict[str, object]:
    metadata_path = resolve_segmentation_metadata_path()
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@lru_cache
def load_anomaly_model() -> object:
    model_path = resolve_anomaly_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


@lru_cache
def load_anomaly_metadata() -> dict[str, object]:
    metadata_path = resolve_anomaly_metadata_path()
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))
