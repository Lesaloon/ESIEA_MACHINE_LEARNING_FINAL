from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part4_anomaly_detection"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "ml_training_dataset.csv"
TARGET = "overload_anomaly"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw supervised anomaly experiment.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model", choices=["gradient_boosting", "random_forest", "extra_trees"], default="gradient_boosting")
    return parser.parse_args()


def build_model(model_name: str):
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1)
    if model_name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=400, random_state=RANDOM_STATE, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
    return GradientBoostingClassifier(n_estimators=300, random_state=RANDOM_STATE, min_samples_leaf=5)


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
    return float(thresholds[index]), {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index])}


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_path)
    X = df.drop(columns=[TARGET, "incident_next_7d", "date", "server_id", "customer_id"])
    y = df[TARGET]
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("model", build_model(args.model))])

    print("Fitting the model...")
    model.fit(X_train, y_train)
    print("Predicting on the test set...")
    y_score = model.predict_proba(X_test)[:, 1]
    threshold, threshold_scores = best_threshold(y_test, y_score)
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "model": args.model,
        "rows": int(len(df)),
        "positive_rate": float(y.mean()),
        "average_precision": float(average_precision_score(y_test, y_score)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "best_threshold": threshold,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "threshold_metrics": threshold_scores,
    }
    print("Average precision:", metrics["average_precision"])
    print("ROC AUC:", metrics["roc_auc"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])

    run_dir = EXPERIMENT_DIR / "raw_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / f"raw_anomaly_{args.model}.pkl")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
