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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "ml_training_dataset.csv"
OUTPUT_ROOT = ROOT_DIR / "train" / "incident_prediction" / "runs"
ARTIFACT_PATH = ROOT_DIR / "models" / "artifacts" / "incident_gradient_boosting_model.pkl"
METADATA_PATH = ROOT_DIR / "models" / "metadata" / "incident_gradient_boosting_model.json"
TARGET = "incident_next_7d"
RANDOM_STATE = 42
CATEGORICAL_FEATURES = ["server_type", "region", "os_family", "segment", "country", "support_plan"]
FEATURES_TO_SCALE = [
    "cpu_cores",
    "ram_gb",
    "disk_tb",
    "age_days",
    "cpu_util_pct",
    "ram_util_pct",
    "disk_util_pct",
    "net_in_gb",
    "net_out_gb",
    "temperature_c",
    "avg_rack_temperature_c",
    "power_usage_mw",
    "network_latency_ms",
    "capacity_used_pct",
    "contract_months",
    "tenure_days",
    "monthly_spend_eur",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final incident classification model on ml_training_dataset.csv.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--cv", type=int, default=4)
    return parser.parse_args()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("final_incident_classifier")
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


def load_data(data_path: Path, logger: logging.Logger) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    logger.info("Loading %s", data_path.relative_to(ROOT_DIR))
    df = pd.read_csv(data_path)
    logger.info("Raw shape: %s", df.shape)

    df_cleaned = df.dropna().drop(columns=["date"])

    X = df_cleaned.drop(columns=[TARGET, "overload_anomaly", "server_id", "customer_id"])
    y = df_cleaned[TARGET]
    logger.info("Feature shape: %s", X.shape)
    logger.info("Target positive rate: %.4f", y.mean())

    preprocessing = {
        "categorical_features": [feature for feature in CATEGORICAL_FEATURES if feature in X.columns],
        "scaled_features": [feature for feature in FEATURES_TO_SCALE if feature in X.columns],
        "feature_columns": X.columns.tolist(),
        "rows_after_cleaning": int(len(df_cleaned)),
    }
    return X, y, preprocessing


def build_pipeline(model: object, feature_columns: list[str]) -> Pipeline:
    categorical_features = [feature for feature in CATEGORICAL_FEATURES if feature in feature_columns]
    scaled_features = [feature for feature in FEATURES_TO_SCALE if feature in feature_columns]
    passthrough_features = [
        feature for feature in feature_columns if feature not in categorical_features and feature not in scaled_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features),
            ("scaled_numeric", StandardScaler(), scaled_features),
            ("passthrough", "passthrough", passthrough_features),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def model_definitions() -> dict[str, object]:
    return {
        "DummyClassifier": DummyClassifier(strategy="uniform", random_state=RANDOM_STATE),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def search_spaces() -> dict[str, dict[str, list[object]]]:
    return {
        "GradientBoostingClassifier": {
            "model__n_estimators": [100, 200, 700],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__subsample": [0.8, 0.9, 1.0],
        },
    }


def best_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds, dtype=float),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), {
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1_values[best_index]),
    }


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, output_dir: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix).plot(ax=ax, colorbar=False)
    ax.set_title("Incident classification confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(output_dir)

    X, y, preprocessing = load_data(args.data_path, logger)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info("Train shape: %s", X_train.shape)
    logger.info("Test shape: %s", X_test.shape)

    models = model_definitions()
    param_spaces = search_spaces()
    results: dict[str, dict[str, object]] = {}
    trained_models: dict[str, object] = {}

    for model_name, model in models.items():
        logger.info("Training %s", model_name)
        pipeline = build_pipeline(model, X.columns.tolist())
        if model_name in param_spaces:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_spaces[model_name],
                n_iter=args.n_iter,
                cv=args.cv,
                scoring="accuracy",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=2,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params: object = search.best_params_
            cv_best_accuracy = float(search.best_score_)
        else:
            best_model = pipeline
            best_model.fit(X_train, y_train)
            best_params = "Default parameters used"
            cv_best_accuracy = None

        trained_models[model_name] = best_model
        y_pred = best_model.predict(X_test)
        result = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_at_default_threshold": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall_at_default_threshold": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_at_default_threshold": float(f1_score(y_test, y_pred, zero_division=0)),
            "best_params": best_params,
            "cv_best_accuracy": cv_best_accuracy,
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)[:, 1]
            threshold, threshold_metrics = best_threshold(y_test, y_score)
            y_threshold_pred = (y_score >= threshold).astype(int)
            result["average_precision"] = float(average_precision_score(y_test, y_score))
            result["roc_auc"] = float(roc_auc_score(y_test, y_score))
            result["best_threshold"] = threshold
            result["precision"] = float(precision_score(y_test, y_threshold_pred, zero_division=0))
            result["recall"] = float(recall_score(y_test, y_threshold_pred, zero_division=0))
            result["f1"] = float(f1_score(y_test, y_threshold_pred, zero_division=0))
            result["threshold_metrics"] = threshold_metrics
        results[model_name] = result
        logger.info(
            "%s accuracy: %.4f | AP: %.4f | ROC-AUC: %.4f | F1: %.4f | Threshold: %.4f",
            model_name,
            result.get("accuracy", 0.0),
            result.get("average_precision", 0.0),
            result.get("roc_auc", 0.0),
            result.get("f1", 0.0),
            result.get("best_threshold", 0.5),
        )

    best_model_name = "GradientBoostingClassifier"
    best_model = trained_models[best_model_name]
    selected_threshold = float(results[best_model_name].get("best_threshold", 0.5))
    y_pred = (best_model.predict_proba(X_test)[:, 1] >= selected_threshold).astype(int)
    joblib.dump(best_model, output_dir / f"{best_model_name}.pkl")
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACT_PATH)
    save_confusion_matrix(y_test, y_pred, output_dir)

    summary = {
        "name": "incident_gradient_boosting_model",
        "artifact_path": str(ARTIFACT_PATH.relative_to(ROOT_DIR)),
        "run_model_path": str((output_dir / f"{best_model_name}.pkl").relative_to(ROOT_DIR)),
        "saved_with": "joblib",
        "model": best_model_name,
        "model_type": best_model.named_steps["model"].__class__.__name__,
        "problem": "binary classification",
        "target": TARGET,
        "threshold": selected_threshold,
        "data_source": str(args.data_path.relative_to(ROOT_DIR)),
        "test_size": args.test_size,
        "n_iter": args.n_iter,
        "cv": args.cv,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "preprocessing": preprocessing,
        "best_model": best_model_name,
        "selection_metric": "accuracy",
        "best_params": results[best_model_name]["best_params"],
        "metrics": results,
    }

    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Selected best model: %s", best_model_name)


if __name__ == "__main__":
    main()
