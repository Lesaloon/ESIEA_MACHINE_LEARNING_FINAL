from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "ml_training_dataset.csv"
OUTPUT_ROOT = ROOT_DIR / "train" / "incident_prediction" / "runs"
ARTIFACT_PATH = ROOT_DIR / "models" / "artifacts" / "incident_random_forest_model.pkl"
METADATA_PATH = ROOT_DIR / "models" / "metadata" / "incident_random_forest_model.json"
TARGET = "incident_next_7d"
OTHER_TARGETS = ["overload_anomaly"]
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final raw ExtraTrees incident model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--include-ids", action="store_true", help="Keep server_id/customer_id as categorical features.")
    parser.add_argument("--top-k", type=int, nargs="+", default=[50, 100, 250, 500])
    return parser.parse_args()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("final_raw_extra_trees_incident")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_data(data_path: Path, include_ids: bool, logger: logging.Logger) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    logger.info("Loading %s", data_path.relative_to(ROOT_DIR))
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    logger.info("Raw shape: %s", df.shape)
    logger.info("Target positive rate: %.4f", df[TARGET].mean())

    traceability = df[["date", "server_id", TARGET]].copy()
    drop_columns = [TARGET, *OTHER_TARGETS, "date"]
    if not include_ids:
        drop_columns.extend(["server_id", "customer_id"])

    X = df.drop(columns=drop_columns)
    y = df[TARGET]
    logger.info("Feature shape: %s | include_ids=%s", X.shape, include_ids)
    return X, y, traceability


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    model = ExtraTreesClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def best_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds, dtype=float),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    index = int(np.nanargmax(f1))
    return float(thresholds[index]), {
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
    }


def topk_metrics(traceability: pd.DataFrame, y_score: np.ndarray, top_k_values: list[int]) -> pd.DataFrame:
    scored = traceability.copy()
    scored["risk_score"] = y_score
    daily_total = scored.groupby("date")[TARGET].sum().rename("daily_positives")
    rows = []

    for top_k in top_k_values:
        selected = scored.sort_values(["date", "risk_score"], ascending=[True, False]).groupby("date").head(top_k)
        daily_hits = selected.groupby("date")[TARGET].sum().rename("hits")
        daily_selected = selected.groupby("date")[TARGET].size().rename("selected")
        daily = pd.concat([daily_total, daily_hits, daily_selected], axis=1).fillna(0)
        total_positives = daily["daily_positives"].sum()
        total_selected = daily["selected"].sum()
        rows.append(
            {
                "top_k_per_day": top_k,
                "selected_total": int(total_selected),
                "captured_incidents": int(daily["hits"].sum()),
                "total_incidents": int(total_positives),
                "capture_rate": float(daily["hits"].sum() / total_positives) if total_positives else 0.0,
                "precision_at_k": float(daily["hits"].sum() / total_selected) if total_selected else 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_dir: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix).plot(ax=ax, colorbar=False)
    ax.set_title("Final raw ExtraTrees confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(output_dir)
    logger.info("Training final raw ExtraTrees incident model")

    X, y, traceability = load_data(args.data_path, args.include_ids, logger)
    X_train, X_test, y_train, y_test, traceability_train, traceability_test = train_test_split(
        X,
        y,
        traceability,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info("Train: %s | positives: %.4f", X_train.shape, y_train.mean())
    logger.info("Test: %s | positives: %.4f", X_test.shape, y_test.mean())

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]
    threshold, threshold_scores = best_threshold(y_test, y_score)
    y_pred = (y_score >= threshold).astype(int)
    topk = topk_metrics(traceability_test, y_score, args.top_k)

    metrics = {
        "average_precision": float(average_precision_score(y_test, y_score)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "best_threshold": threshold,
        "precision": threshold_scores["precision"],
        "recall": threshold_scores["recall"],
        "f1": threshold_scores["f1"],
    }

    model_path = output_dir / "final_raw_extra_trees_model.pkl"
    joblib.dump(model, model_path)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_PATH)

    pd.DataFrame([metrics]).to_csv(output_dir / "final_metrics.csv", index=False)
    (output_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    topk.to_csv(output_dir / "topk_metrics.csv", index=False)
    save_confusion_matrix(y_test, y_pred, output_dir)

    summary = {
        "name": "incident_random_forest_model",
        "model": "ExtraTreesClassifier",
        "artifact_path": str(ARTIFACT_PATH.relative_to(ROOT_DIR)),
        "run_model_path": str(model_path.relative_to(ROOT_DIR)),
        "saved_with": "joblib",
        "model_type": "ExtraTreesClassifier",
        "data_source": str(args.data_path.relative_to(ROOT_DIR)),
        "include_ids": args.include_ids,
        "threshold": threshold,
        **metrics,
        "metrics_path": str((output_dir / "final_metrics.json").relative_to(ROOT_DIR)),
        "topk_path": str((output_dir / "topk_metrics.csv").relative_to(ROOT_DIR)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "best_params": {
            "model__n_estimators": "400",
            "model__min_samples_leaf": "3",
            "model__class_weight": "balanced",
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Final metrics: %s", metrics)
    logger.info("Top-K metrics:\n%s", topk.to_string(index=False))
    logger.info("Saved final model to %s", ARTIFACT_PATH.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
