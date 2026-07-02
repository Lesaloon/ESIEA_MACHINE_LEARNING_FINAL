from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform, uniform
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
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
from sklearn.feature_selection import VarianceThreshold


matplotlib.use("Agg")


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part1_incident_prediction"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "processed" / "incident_dataset.csv"
TARGET = "incident_next_7d"
RANDOM_STATE = 42
TEST_START_DATE = "2026-03-01"

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
    parser = argparse.ArgumentParser(description="Second experiment for incident prediction.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-start-date", default=TEST_START_DATE)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--top-k", type=int, nargs="+", default=[50, 100, 250, 500])
    return parser.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("incident_experiments_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(run_dir / "experiment_v2.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def make_jsonable(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: make_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "get_params"):
        return repr(value)
    return value


def load_dataset(data_path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading dataset from %s", data_path.relative_to(ROOT_DIR))
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values(["server_id", "date"]).reset_index(drop=True)

    logger.info("Dataset shape: %s", df.shape)
    logger.info("Date range: %s -> %s", df["date"].min().date(), df["date"].max().date())
    logger.info("Servers: %s", df["server_id"].nunique())
    logger.info("Target positive rate: %.4f", df[TARGET].mean())
    logger.info("Missing values: %s", int(df.isna().sum().sum()))
    logger.info("Duplicate rows: %s", int(df.duplicated().sum()))
    return df


def add_historical_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Adding historical server-level features")
    df = df.sort_values(["server_id", "date"]).copy()
    grouped = df.groupby("server_id", sort=False)

    for column in HISTORY_COLUMNS:
        if column not in df.columns:
            continue
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

    created = [column for column in df.columns if "lag1" in column or "rolling" in column or "pressure" in column]
    logger.info("Created %s engineered features", len(created))
    logger.info("Shape after feature engineering: %s", df.shape)
    return df.sort_values("date").reset_index(drop=True)


def split_temporal(
    df: pd.DataFrame,
    test_start_date: str,
    max_train_rows: int | None,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    test_start = pd.Timestamp(test_start_date)
    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()

    if max_train_rows is not None and len(train_df) > max_train_rows:
        train_df = train_df.tail(max_train_rows).copy()
        logger.info("Using latest %s training rows for faster experimentation", max_train_rows)

    traceability_test = test_df[["date", "server_id", TARGET]].copy()
    drop_columns = [TARGET, "date", "server_id"]
    X_train = train_df.drop(columns=drop_columns)
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=drop_columns)
    y_test = test_df[TARGET]

    logger.info("Train shape: %s | positive rate: %.4f", train_df.shape, y_train.mean())
    logger.info("Test shape: %s | positive rate: %.4f", test_df.shape, y_test.mean())
    return X_train, y_train, X_test, y_test, traceability_test


def build_preprocessor(X_train: pd.DataFrame, logger: logging.Logger) -> ColumnTransformer:
    categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    logger.info("Numeric features: %s", len(numeric_features))
    logger.info("Categorical features: %s", categorical_features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def model_spaces() -> dict[str, tuple[object, dict[str, object]]]:
    common_reducers = [
        "passthrough",
        SelectKBest(score_func=f_classif, k=30),
        SelectKBest(score_func=f_classif, k=50),
        PCA(n_components=0.90, random_state=RANDOM_STATE),
        PCA(n_components=0.95, random_state=RANDOM_STATE),
    ]
    return {
        "logistic_regression": (
            LogisticRegression(class_weight="balanced", max_iter=1200, random_state=RANDOM_STATE),
            {
                "reducer": common_reducers,
                "model__C": loguniform(1e-3, 1e2),
            },
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "reducer": ["passthrough", SelectKBest(score_func=f_classif, k=40), SelectKBest(score_func=f_classif, k=70)],
                "model__n_estimators": randint(180, 520),
                "model__max_depth": [8, 12, 18, 24, None],
                "model__min_samples_split": randint(2, 20),
                "model__min_samples_leaf": randint(1, 16),
                "model__max_features": ["sqrt", "log2", 0.4, 0.7, 1.0],
            },
        ),
        "extra_trees": (
            ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "reducer": ["passthrough", SelectKBest(score_func=f_classif, k=40), SelectKBest(score_func=f_classif, k=70)],
                "model__n_estimators": randint(180, 520),
                "model__max_depth": [8, 12, 18, 24, None],
                "model__min_samples_split": randint(2, 20),
                "model__min_samples_leaf": randint(1, 16),
                "model__max_features": ["sqrt", "log2", 0.4, 0.7, 1.0],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "reducer": ["passthrough", SelectKBest(score_func=f_classif, k=50), SelectKBest(score_func=f_classif, k=80)],
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__max_iter": randint(100, 320),
                "model__max_leaf_nodes": randint(15, 80),
                "model__min_samples_leaf": randint(10, 100),
                "model__l2_regularization": uniform(0.0, 1.5),
            },
        ),
    }


def positive_scores(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    scores = estimator.decision_function(X)
    score_range = scores.max() - scores.min()
    if score_range == 0:
        return np.zeros_like(scores)
    return (scores - scores.min()) / score_range


def threshold_metrics(y_true: pd.Series, y_score: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_index = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    threshold = float(thresholds[best_index]) if len(thresholds) else 0.5
    y_pred = (y_score >= threshold).astype(int)
    return {
        "best_threshold": threshold,
        "threshold_precision_positive": float(precision_score(y_true, y_pred, zero_division=0)),
        "threshold_recall_positive": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold_f1_positive": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def topk_metrics(
    traceability: pd.DataFrame,
    y_score: np.ndarray,
    top_k_values: list[int],
) -> pd.DataFrame:
    scored = traceability.copy()
    scored["risk_score"] = y_score
    daily_positive_total = scored.groupby("date")[TARGET].sum().rename("daily_positives")
    rows = []

    for top_k in top_k_values:
        selected = scored.sort_values(["date", "risk_score"], ascending=[True, False]).groupby("date").head(top_k)
        daily_hits = selected.groupby("date")[TARGET].sum().rename("hits")
        daily_selected = selected.groupby("date")[TARGET].size().rename("selected")
        daily = pd.concat([daily_positive_total, daily_hits, daily_selected], axis=1).fillna(0)

        rows.append(
            {
                "top_k_per_day": int(top_k),
                "selected_total": int(daily["selected"].sum()),
                "captured_incidents": int(daily["hits"].sum()),
                "total_incidents": int(daily["daily_positives"].sum()),
                "capture_rate": float(daily["hits"].sum() / daily["daily_positives"].sum()),
                "precision_at_k": float(daily["hits"].sum() / daily["selected"].sum()),
                "mean_daily_capture_rate": float((daily["hits"] / daily["daily_positives"].replace(0, np.nan)).mean()),
            }
        )

    return pd.DataFrame(rows)


def evaluate(
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    traceability_test: pd.DataFrame,
    model_name: str,
    run_dir: Path,
    top_k_values: list[int],
    logger: logging.Logger,
) -> dict[str, object]:
    y_score = positive_scores(estimator, X_test)
    y_pred_default = estimator.predict(X_test)
    optimized_threshold_metrics = threshold_metrics(y_test, y_score)

    metrics = {
        "model": model_name,
        "precision_positive_default": float(precision_score(y_test, y_pred_default, zero_division=0)),
        "recall_positive_default": float(recall_score(y_test, y_pred_default, zero_division=0)),
        "f1_positive_default": float(f1_score(y_test, y_pred_default, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "average_precision": float(average_precision_score(y_test, y_score)),
        **optimized_threshold_metrics,
    }

    logger.info(
        "%s | AP=%.4f ROC-AUC=%.4f F1@best=%.4f Recall@best=%.4f Precision@best=%.4f",
        model_name,
        metrics["average_precision"],
        metrics["roc_auc"],
        metrics["threshold_f1_positive"],
        metrics["threshold_recall_positive"],
        metrics["threshold_precision_positive"],
    )

    report = classification_report(y_test, y_pred_default, output_dict=True, zero_division=0)
    (run_dir / "classification_reports").mkdir(exist_ok=True)
    (run_dir / "classification_reports" / f"{model_name}_default_threshold.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(y_test, y_pred_default)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix).plot(ax=ax, colorbar=False)
    ax.set_title(f"Default threshold confusion matrix - {model_name}")
    fig.tight_layout()
    (run_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(run_dir / "figures" / f"confusion_matrix_default_{model_name}.png", dpi=160)
    plt.close(fig)

    topk_df = topk_metrics(traceability_test, y_score, top_k_values)
    topk_df.insert(0, "model", model_name)
    topk_df.to_csv(run_dir / f"topk_metrics_{model_name}.csv", index=False)
    for _, row in topk_df.iterrows():
        metrics[f"top_{int(row['top_k_per_day'])}_capture_rate"] = float(row["capture_rate"])
        metrics[f"top_{int(row['top_k_per_day'])}_precision"] = float(row["precision_at_k"])

    return metrics


def run_searches(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    traceability_test: pd.DataFrame,
    args: argparse.Namespace,
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[list[dict[str, object]], Pipeline, str, dict[str, object]]:
    cv = TimeSeriesSplit(n_splits=args.cv_splits)
    results = []
    best_estimator = None
    best_model = ""
    best_params: dict[str, object] = {}
    best_score = -np.inf

    for model_name, (model, param_distributions) in model_spaces().items():
        logger.info("Starting v2 RandomizedSearchCV for %s", model_name)
        start = perf_counter()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("variance_filter", VarianceThreshold()),
                ("reducer", "passthrough"),
                ("model", model),
            ]
        )
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=args.n_iter,
            scoring="average_precision",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=args.n_jobs,
            verbose=2,
            refit=True,
        )
        search.fit(X_train, y_train)
        elapsed = perf_counter() - start
        logger.info("Finished %s in %.1fs", model_name, elapsed)
        logger.info("%s best CV average_precision: %.4f", model_name, search.best_score_)
        logger.info("%s best params: %s", model_name, search.best_params_)

        pd.DataFrame(search.cv_results_).to_csv(run_dir / f"cv_results_v2_{model_name}.csv", index=False)

        metrics = evaluate(
            search.best_estimator_,
            X_test,
            y_test,
            traceability_test,
            model_name,
            run_dir,
            args.top_k,
            logger,
        )
        metrics["best_cv_average_precision"] = float(search.best_score_)
        metrics["training_seconds"] = float(elapsed)
        metrics["best_params"] = make_jsonable(search.best_params_)
        results.append(metrics)

        if metrics["average_precision"] > best_score:
            best_score = metrics["average_precision"]
            best_estimator = search.best_estimator_
            best_model = model_name
            best_params = metrics["best_params"]

    if best_estimator is None:
        raise RuntimeError("No model trained successfully")
    return results, best_estimator, best_model, best_params


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = EXPERIMENT_DIR / "runs_v2" / timestamp
    logger = setup_logging(run_dir)

    logger.info("Incident prediction second experiment started")
    logger.info("Run directory: %s", run_dir.relative_to(ROOT_DIR))
    logger.info("Arguments: %s", vars(args))

    df = load_dataset(args.data_path, logger)
    df = add_historical_features(df, logger)
    X_train, y_train, X_test, y_test, traceability_test = split_temporal(
        df, args.test_start_date, args.max_train_rows, logger
    )
    preprocessor = build_preprocessor(X_train, logger)

    results, best_estimator, best_model, best_params = run_searches(
        preprocessor, X_train, y_train, X_test, y_test, traceability_test, args, run_dir, logger
    )

    metrics_df = pd.DataFrame(results).sort_values("average_precision", ascending=False)
    metrics_df.to_csv(run_dir / "benchmark_metrics_v2.csv", index=False)

    best_model_path = run_dir / "best_incident_model_v2.joblib"
    joblib.dump(best_estimator, best_model_path)

    summary = {
        "target": TARGET,
        "problem_type": "binary_classification",
        "main_metric": "average_precision",
        "business_metrics": [f"top_{k}_capture_rate" for k in args.top_k],
        "best_model": best_model,
        "best_params": best_params,
        "best_model_path": str(best_model_path.relative_to(ROOT_DIR)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "metrics_path": str((run_dir / "benchmark_metrics_v2.csv").relative_to(ROOT_DIR)),
        "notes": "Second experiment adds historical server features and tunes feature reducers including SelectKBest and PCA.",
    }
    (run_dir / "run_summary_v2.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Benchmark ranking:\n%s", metrics_df.to_string(index=False))
    logger.info("Best model: %s", best_model)
    logger.info("Saved best model to %s", best_model_path.relative_to(ROOT_DIR))
    logger.info("Incident prediction second experiment finished")


if __name__ == "__main__":
    main()
