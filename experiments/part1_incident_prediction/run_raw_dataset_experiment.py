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
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part1_incident_prediction"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "ml_training_dataset.csv"
RANDOM_STATE = 42
TARGETS = ["incident_next_7d", "overload_anomaly"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple raw-dataset classification experiment.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--target", choices=TARGETS, default="incident_next_7d")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--include-ids", action="store_true", help="Keep server_id/customer_id as categorical features.")
    parser.add_argument(
        "--model",
        choices=["gradient_boosting", "random_forest", "extra_trees"],
        default="gradient_boosting",
    )
    return parser.parse_args()


def build_model(model_name: str):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
        )
    return GradientBoostingClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        min_samples_leaf=5,
    )


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
    cutoff = float(thresholds[index])
    return cutoff, {
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_path)

    drop_columns = [target for target in TARGETS if target != args.target]
    drop_columns.append(args.target)
    if not args.include_ids:
        drop_columns.extend(["server_id", "customer_id"])

    X = df.drop(columns=drop_columns)
    y = df[args.target]

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    raw_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_model(args.model)),
        ]
    )

    print("Fitting the model...")
    raw_model.fit(X_train, y_train)

    print("Predicting on the test set...")
    y_score = raw_model.predict_proba(X_test)[:, 1]
    threshold, threshold_metrics = best_threshold(y_test, y_score)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "target": args.target,
        "model": args.model,
        "include_ids": args.include_ids,
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "average_precision": float(average_precision_score(y_test, y_score)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "best_threshold": threshold,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "threshold_metrics": threshold_metrics,
    }

    print("Average precision:", metrics["average_precision"])
    print("ROC AUC:", metrics["roc_auc"])
    print("Best threshold:", metrics["best_threshold"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])

    run_dir = EXPERIMENT_DIR / "raw_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(raw_model, run_dir / f"raw_{args.target}_{args.model}.pkl")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
