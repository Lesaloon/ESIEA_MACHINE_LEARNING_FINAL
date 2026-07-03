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
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part4_anomaly_detection"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "ml_training_dataset.csv"
TARGET = "overload_anomaly"
TEST_START_DATE = "2026-03-01"
RANDOM_STATE = 42

DROP_COLUMNS = ["customer_id", "incident_next_7d", TARGET]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark anomaly detection models.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-start-date", default=TEST_START_DATE)
    parser.add_argument("--max-fit-rows", type=int, default=30000)
    return parser.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("anomaly_detection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    start_date = df["date"].min()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["days_since_start"] = (df["date"] - start_date).dt.days
    df["cpu_ram_pressure"] = df["cpu_util_pct"] * df["ram_util_pct"] / 100
    df["thermal_pressure"] = df["temperature_c"] * df["cpu_util_pct"] / 100
    df["network_total_gb"] = df["net_in_gb"] + df["net_out_gb"]
    df["network_balance_gb"] = df["net_in_gb"] - df["net_out_gb"]
    df["utilization_pressure"] = (
        df["cpu_util_pct"] + df["ram_util_pct"] + df["disk_util_pct"] + df["capacity_used_pct"]
    ) / 4
    return df.sort_values("date").reset_index(drop=True)


def load_dataset(data_path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    df = add_features(df)
    logger.info("Dataset: %s", data_path.relative_to(ROOT_DIR))
    logger.info("Shape: %s", df.shape)
    logger.info("Date range: %s -> %s", df["date"].min().date(), df["date"].max().date())
    logger.info("Anomaly rate: %.4f", df[TARGET].mean())
    logger.info("Missing values: %s", int(df.isna().sum().sum()))
    logger.info("Duplicate rows: %s", int(df.duplicated().sum()))
    return df


def temporal_split(df: pd.DataFrame, test_start_date: str, logger: logging.Logger):
    test_start = pd.Timestamp(test_start_date)
    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()
    traceability_test = test_df[["date", "server_id", TARGET]].copy()
    drop_columns = [*DROP_COLUMNS, "date", "server_id"]
    X_train = train_df.drop(columns=drop_columns)
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=drop_columns)
    y_test = test_df[TARGET]
    logger.info("Train: %s | anomaly rate %.4f", X_train.shape, y_train.mean())
    logger.info("Test: %s | anomaly rate %.4f", X_test.shape, y_test.mean())
    return X_train, y_train, X_test, y_test, traceability_test


def fit_sample(X_train: pd.DataFrame, y_train: pd.Series, max_rows: int) -> tuple[pd.DataFrame, pd.Series]:
    if len(X_train) <= max_rows:
        return X_train, y_train
    sampled_index = X_train.sample(n=max_rows, random_state=RANDOM_STATE).index
    return X_train.loc[sampled_index], y_train.loc[sampled_index]


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )


def model_spaces(contamination: float) -> dict[str, object]:
    return {
        "isolation_forest": IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "local_outlier_factor": LocalOutlierFactor(
            n_neighbors=35,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        ),
        "sgd_one_class_svm": SGDOneClassSVM(
            nu=contamination,
            random_state=RANDOM_STATE,
            max_iter=2000,
            tol=1e-4,
        ),
    }


def anomaly_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "decision_function"):
        return -pipeline.decision_function(X)
    return -pipeline.score_samples(X)


def threshold_from_train(scores: np.ndarray, contamination: float) -> float:
    return float(np.quantile(scores, 1 - contamination))


def evaluate_model(
    pipeline: Pipeline,
    model_name: str,
    X_train_fit: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_fit: pd.Series,
    y_test: pd.Series,
    traceability_test: pd.DataFrame,
    contamination: float,
    run_dir: Path,
) -> dict[str, object]:
    train_scores = anomaly_scores(pipeline, X_train_fit)
    threshold = threshold_from_train(train_scores, contamination)
    test_scores = anomaly_scores(pipeline, X_test)
    y_pred = (test_scores >= threshold).astype(int)

    predictions = traceability_test.copy()
    predictions["anomaly_score"] = test_scores
    predictions["prediction"] = y_pred
    predictions["model"] = model_name
    predictions.to_csv(run_dir / f"predictions_test_{model_name}.csv", index=False)

    return {
        "model": model_name,
        "threshold": threshold,
        "train_fit_rows": int(len(X_train_fit)),
        "train_fit_anomaly_rate": float(y_train_fit.mean()),
        "predicted_anomaly_rate": float(y_pred.mean()),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, test_scores)),
        "roc_auc": float(roc_auc_score(y_test, test_scores)),
    }


def plot_score_distribution(best_predictions: pd.DataFrame, run_dir: Path, model_name: str) -> None:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    normal_scores = best_predictions[best_predictions[TARGET] == 0]["anomaly_score"]
    anomaly_scores_values = best_predictions[best_predictions[TARGET] == 1]["anomaly_score"]
    plt.hist(normal_scores, bins=50, alpha=0.65, label="normal")
    plt.hist(anomaly_scores_values, bins=50, alpha=0.65, label="anomaly")
    plt.title(f"Anomaly score distribution - {model_name}")
    plt.xlabel("Anomaly score")
    plt.ylabel("Rows")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "anomaly_score_distribution.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = EXPERIMENT_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(run_dir)
    logger.info("Anomaly detection experiment started")
    logger.info("Run directory: %s", run_dir.relative_to(ROOT_DIR))
    logger.info("Arguments: %s", vars(args))

    df = load_dataset(args.data_path, logger)
    X_train, y_train, X_test, y_test, traceability_test = temporal_split(df, args.test_start_date, logger)
    X_train_fit, y_train_fit = fit_sample(X_train, y_train, args.max_fit_rows)
    contamination = float(max(y_train.mean(), 0.001))
    logger.info("Using contamination %.4f", contamination)

    results = []
    trained_models = {}
    for model_name, model in model_spaces(contamination).items():
        logger.info("Training %s", model_name)
        pipeline = Pipeline([("preprocessor", build_preprocessor(X_train_fit)), ("model", model)])
        pipeline.fit(X_train_fit)
        metrics = evaluate_model(
            pipeline,
            model_name,
            X_train_fit,
            X_test,
            y_train_fit,
            y_test,
            traceability_test,
            contamination,
            run_dir,
        )
        logger.info("%s metrics: %s", model_name, metrics)
        results.append(metrics)
        trained_models[model_name] = pipeline

    metrics_df = pd.DataFrame(results).sort_values(["average_precision", "f1"], ascending=[False, False])
    metrics_path = run_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    best_model_name = str(metrics_df.iloc[0]["model"])
    best_pipeline = trained_models[best_model_name]
    model_path = run_dir / f"best_anomaly_{best_model_name}.pkl"
    joblib.dump(best_pipeline, model_path)

    best_predictions = pd.read_csv(run_dir / f"predictions_test_{best_model_name}.csv")
    plot_score_distribution(best_predictions, run_dir, best_model_name)

    summary = {
        "problem_type": "unsupervised_anomaly_detection",
        "target_used_for_evaluation_only": TARGET,
        "selection_metric": "average_precision on temporal test set",
        "best_model": best_model_name,
        "best_model_path": str(model_path.relative_to(ROOT_DIR)),
        "metrics_path": str(metrics_path.relative_to(ROOT_DIR)),
        "contamination": contamination,
        "train_rows": int(len(X_train)),
        "train_fit_rows": int(len(X_train_fit)),
        "test_rows": int(len(X_test)),
        "best_metrics": metrics_df.iloc[0].to_dict(),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Model ranking:\n%s", metrics_df.to_string(index=False))
    logger.info("Best model: %s", best_model_name)
    logger.info("Anomaly detection experiment finished")


if __name__ == "__main__":
    main()
