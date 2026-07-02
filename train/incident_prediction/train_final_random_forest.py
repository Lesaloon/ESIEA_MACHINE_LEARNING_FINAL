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
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "processed" / "incident_dataset.csv"
OUTPUT_ROOT = ROOT_DIR / "train" / "incident_prediction" / "runs"
TARGET = "incident_next_7d"
TEST_START_DATE = "2026-03-01"
RANDOM_STATE = 42

HISTORY_COLUMNS = [
    "cpu_util_pct",
    "ram_util_pct",
    "disk_util_pct",
    "temperature_c",
    "net_in_gb",
    "net_out_gb",
    "network_latency_ms",
    "support_tickets",
    "capacity_used_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final RandomForest incident model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-start-date", default=TEST_START_DATE)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--top-k", type=int, nargs="+", default=[50, 100, 250, 500])
    return parser.parse_args()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("final_random_forest")
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


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["server_id", "date"]).copy()
    grouped = df.groupby("server_id", sort=False)

    for column in HISTORY_COLUMNS:
        shifted = grouped[column].shift(1)
        df[f"{column}_lag1"] = shifted
        df[f"{column}_diff1"] = df[column] - shifted
        df[f"{column}_rolling_mean_3"] = shifted.groupby(df["server_id"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{column}_rolling_mean_7"] = shifted.groupby(df["server_id"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{column}_rolling_max_7"] = shifted.groupby(df["server_id"]).rolling(7, min_periods=1).max().reset_index(level=0, drop=True)

    df["cpu_ram_pressure"] = df["cpu_util_pct"] * df["ram_util_pct"] / 100
    df["thermal_pressure"] = df["temperature_c"] * df["cpu_util_pct"] / 100
    df["network_total_gb"] = df["net_in_gb"] + df["net_out_gb"]
    df["network_balance_gb"] = df["net_in_gb"] - df["net_out_gb"]
    df["utilization_pressure"] = (
        df["cpu_util_pct"] + df["ram_util_pct"] + df["disk_util_pct"] + df["capacity_used_pct"]
    ) / 4
    return df.sort_values("date").reset_index(drop=True)


def load_data(data_path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading %s", data_path.relative_to(ROOT_DIR))
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    logger.info("Raw shape: %s", df.shape)
    logger.info("Target positive rate: %.4f", df[TARGET].mean())
    df = add_historical_features(df)
    logger.info("Shape after feature engineering: %s", df.shape)
    return df


def temporal_split(df: pd.DataFrame, test_start_date: str, logger: logging.Logger):
    test_start = pd.Timestamp(test_start_date)
    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()
    traceability_test = test_df[["date", "server_id", TARGET]].copy()

    X_train = train_df.drop(columns=[TARGET, "date", "server_id"])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET, "date", "server_id"])
    y_test = test_df[TARGET]

    logger.info("Train: %s | positives: %.4f", X_train.shape, y_train.mean())
    logger.info("Test: %s | positives: %.4f", X_test.shape, y_test.mean())
    return X_train, y_train, X_test, y_test, traceability_test


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("variance_filter", VarianceThreshold()),
            ("selector", SelectKBest(score_func=f_classif, k=70)),
            ("model", model),
        ]
    )


def param_distributions() -> dict[str, object]:
    return {
        "selector__k": [50, 60, 70, 80],
        "model__n_estimators": randint(200, 420),
        "model__max_depth": [18, 24, None],
        "model__max_features": [0.5, 0.7, 0.9, "sqrt"],
        "model__min_samples_leaf": randint(5, 14),
        "model__min_samples_split": randint(10, 24),
    }


def best_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    index = int(np.nanargmax(f1[:-1]))
    threshold = float(thresholds[index])
    y_pred = (y_score >= threshold).astype(int)
    return threshold, {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
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
        rows.append(
            {
                "top_k_per_day": top_k,
                "selected_total": int(daily["selected"].sum()),
                "captured_incidents": int(daily["hits"].sum()),
                "total_incidents": int(daily["daily_positives"].sum()),
                "capture_rate": float(daily["hits"].sum() / daily["daily_positives"].sum()),
                "precision_at_k": float(daily["hits"].sum() / daily["selected"].sum()),
            }
        )
    return pd.DataFrame(rows)


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_dir: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix).plot(ax=ax, colorbar=False)
    ax.set_title("Final RandomForest confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(output_dir)
    logger.info("Training final RandomForest incident model")

    df = load_data(args.data_path, logger)
    X_train, y_train, X_test, y_test, traceability_test = temporal_split(df, args.test_start_date, logger)

    search = RandomizedSearchCV(
        estimator=build_pipeline(X_train),
        param_distributions=param_distributions(),
        n_iter=args.n_iter,
        scoring="average_precision",
        cv=TimeSeriesSplit(n_splits=args.cv_splits),
        random_state=RANDOM_STATE,
        n_jobs=args.n_jobs,
        verbose=2,
        refit=True,
    )

    logger.info("Starting RandomizedSearchCV with %s iterations", args.n_iter)
    search.fit(X_train, y_train)
    logger.info("Best CV average_precision: %.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)

    best_model = search.best_estimator_
    y_score = best_model.predict_proba(X_test)[:, 1]
    threshold, threshold_scores = best_threshold(y_test, y_score)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "average_precision": float(average_precision_score(y_test, y_score)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "best_threshold": threshold,
        "precision": threshold_scores["precision"],
        "recall": threshold_scores["recall"],
        "f1": threshold_scores["f1"],
        "best_cv_average_precision": float(search.best_score_),
        "best_params": {key: str(value) for key, value in search.best_params_.items()},
    }

    model_path = output_dir / "final_random_forest_model.pkl"
    joblib.dump(best_model, model_path)
    pd.DataFrame(search.cv_results_).to_csv(output_dir / "cv_results.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "final_metrics.csv", index=False)
    (output_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    topk = topk_metrics(traceability_test, y_score, args.top_k)
    topk.to_csv(output_dir / "topk_metrics.csv", index=False)
    save_confusion_matrix(y_test, y_pred, output_dir)

    summary = {
        "model": "RandomForestClassifier",
        "model_path": str(model_path.relative_to(ROOT_DIR)),
        "metrics_path": str((output_dir / "final_metrics.json").relative_to(ROOT_DIR)),
        "topk_path": str((output_dir / "topk_metrics.csv").relative_to(ROOT_DIR)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_positive_rate": float(y_test.mean()),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Final metrics: %s", metrics)
    logger.info("Top-K metrics:\n%s", topk.to_string(index=False))
    logger.info("Saved final model to %s", model_path.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
