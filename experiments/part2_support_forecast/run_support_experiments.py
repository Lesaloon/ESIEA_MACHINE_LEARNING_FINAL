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
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part2_support_forecast"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "processed" / "support_dataset.csv"
INCIDENT_DATA_PATH = ROOT_DIR / "data" / "processed" / "incident_dataset.csv"
TARGET = "support_tickets"
TEST_START_DATE = "2026-03-01"
RANDOM_STATE = 42

HISTORY_COLUMNS = [
    "support_tickets",
    "network_latency_ms",
    "capacity_used_pct",
    "power_usage_mw",
    "avg_rack_temperature_c",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark support ticket forecasting models.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-start-date", default=TEST_START_DATE)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--use-server-aggregates", action="store_true")
    return parser.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("support_forecast_experiments")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
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
    return repr(value)


def load_dataset(data_path: Path, use_server_aggregates: bool, logger: logging.Logger) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values(["region", "date"]).reset_index(drop=True)

    logger.info("Dataset: %s", data_path.relative_to(ROOT_DIR))
    logger.info("Shape: %s", df.shape)
    logger.info("Date range: %s -> %s", df["date"].min().date(), df["date"].max().date())
    logger.info("Regions: %s", sorted(df["region"].unique()))
    logger.info("Target mean/std/min/max: %.3f / %.3f / %.0f / %.0f", df[TARGET].mean(), df[TARGET].std(), df[TARGET].min(), df[TARGET].max())
    logger.info("Missing values: %s", int(df.isna().sum().sum()))
    logger.info("Duplicate rows: %s", int(df.duplicated().sum()))
    if use_server_aggregates:
        df = add_server_region_aggregates(df, logger)
    return df


def add_server_region_aggregates(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if not INCIDENT_DATA_PATH.exists():
        logger.info("No incident dataset found; skipping server-region aggregates")
        return df

    logger.info("Adding server-region/day aggregates from %s", INCIDENT_DATA_PATH.relative_to(ROOT_DIR))
    server_df = pd.read_csv(INCIDENT_DATA_PATH)
    server_df["date"] = pd.to_datetime(server_df["date"], errors="raise")

    # Do not aggregate support_tickets here: it is the target in this task.
    aggregations = {
        "server_id": "nunique",
        "cpu_util_pct": ["mean", "max"],
        "ram_util_pct": ["mean", "max"],
        "disk_util_pct": ["mean", "max"],
        "temperature_c": ["mean", "max"],
        "net_in_gb": ["mean", "sum"],
        "net_out_gb": ["mean", "sum"],
        "backup_success": "mean",
        "has_gpu": "mean",
        "is_managed": "mean",
        "age_days": "mean",
        "monthly_spend_eur": "mean",
    }
    aggregate_df = server_df.groupby(["date", "region"]).agg(aggregations)
    aggregate_df.columns = ["server_" + "_".join(column).strip("_") for column in aggregate_df.columns]
    aggregate_df = aggregate_df.reset_index()

    pressure_df = server_df.assign(
        server_high_cpu=(server_df["cpu_util_pct"] >= 80).astype(int),
        server_high_ram=(server_df["ram_util_pct"] >= 80).astype(int),
        server_high_disk=(server_df["disk_util_pct"] >= 80).astype(int),
        server_hot=(server_df["temperature_c"] >= 70).astype(int),
        server_backup_failure=(server_df["backup_success"] == 0).astype(int),
    )
    pressure_agg = pressure_df.groupby(["date", "region"])[
        ["server_high_cpu", "server_high_ram", "server_high_disk", "server_hot", "server_backup_failure"]
    ].sum().reset_index()

    aggregate_df = aggregate_df.merge(pressure_agg, on=["date", "region"], how="left")
    merged = df.merge(aggregate_df, on=["date", "region"], how="left")
    logger.info("Shape after server aggregates: %s", merged.shape)
    return merged


def add_historical_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Adding region-level historical features")
    df = df.sort_values(["region", "date"]).copy()
    grouped = df.groupby("region", sort=False)

    for column in HISTORY_COLUMNS:
        shifted = grouped[column].shift(1)
        lag7 = grouped[column].shift(7)
        df[f"{column}_lag1"] = shifted
        df[f"{column}_lag7"] = lag7
        if column == TARGET:
            df[f"{column}_diff1"] = shifted - lag7
        else:
            df[f"{column}_diff1"] = df[column] - shifted
        df[f"{column}_rolling_mean_3"] = shifted.groupby(df["region"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{column}_rolling_mean_7"] = shifted.groupby(df["region"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{column}_rolling_std_7"] = shifted.groupby(df["region"]).rolling(7, min_periods=2).std().reset_index(level=0, drop=True)
        df[f"{column}_rolling_max_7"] = shifted.groupby(df["region"]).rolling(7, min_periods=1).max().reset_index(level=0, drop=True)

    df["infra_pressure"] = (
        df["capacity_used_pct"] + df["network_latency_ms"] + df["avg_rack_temperature_c"]
    ) / 3
    df["maintenance_latency_interaction"] = df["scheduled_maintenance"] * df["network_latency_ms"]
    df["capacity_power_interaction"] = df["capacity_used_pct"] * df["power_usage_mw"]

    logger.info("Shape after feature engineering: %s", df.shape)
    return df.sort_values("date").reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame,
    test_start_date: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    test_start = pd.Timestamp(test_start_date)
    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()

    traceability_test = test_df[["date", "region", TARGET]].copy()
    drop_columns = [TARGET, "date"]
    X_train = train_df.drop(columns=drop_columns)
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=drop_columns)
    y_test = test_df[TARGET]

    logger.info("Temporal split: train < %s, test >= %s", test_start.date(), test_start.date())
    logger.info("Train shape: %s | target mean: %.3f", train_df.shape, y_train.mean())
    logger.info("Test shape: %s | target mean: %.3f", test_df.shape, y_test.mean())
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
    return {
        "ridge": (
            Ridge(random_state=RANDOM_STATE),
            {"model__alpha": loguniform(1e-3, 1e2)},
        ),
        "random_forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": randint(120, 420),
                "model__max_depth": [3, 5, 8, 12, None],
                "model__min_samples_split": randint(2, 16),
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": ["sqrt", "log2", 0.6, 0.9, 1.0],
            },
        ),
        "extra_trees": (
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": randint(120, 420),
                "model__max_depth": [3, 5, 8, 12, None],
                "model__min_samples_split": randint(2, 16),
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": ["sqrt", "log2", 0.6, 0.9, 1.0],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__max_iter": randint(80, 280),
                "model__max_leaf_nodes": randint(8, 45),
                "model__min_samples_leaf": randint(5, 50),
                "model__l2_regularization": uniform(0.0, 1.5),
            },
        ),
    }


def clipped_predictions(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.clip(estimator.predict(X), 0, None)


def evaluate_model(
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    traceability_test: pd.DataFrame,
    model_name: str,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, object]:
    y_pred = clipped_predictions(estimator, X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    safe_mape = float(np.mean(np.abs(y_test - y_pred) / np.maximum(y_test, 1)))

    predictions = traceability_test.copy()
    predictions["prediction"] = y_pred
    predictions["absolute_error"] = np.abs(predictions[TARGET] - predictions["prediction"])
    predictions["model"] = model_name
    predictions.to_csv(run_dir / f"predictions_test_{model_name}.csv", index=False)

    by_region = predictions.groupby("region").agg(
        rows=(TARGET, "size"),
        actual_mean=(TARGET, "mean"),
        prediction_mean=("prediction", "mean"),
        mae=("absolute_error", "mean"),
    )
    by_region.to_csv(run_dir / f"metrics_by_region_{model_name}.csv")

    logger.info("%s test metrics | MAE=%.4f RMSE=%.4f R2=%.4f safeMAPE=%.4f", model_name, mae, rmse, r2, safe_mape)
    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "safe_mape": safe_mape,
    }


def plot_best_predictions(predictions: pd.DataFrame, run_dir: Path, model_name: str) -> None:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(predictions[TARGET], predictions["prediction"], alpha=0.75)
    max_value = max(predictions[TARGET].max(), predictions["prediction"].max())
    plt.plot([0, max_value], [0, max_value], color="red", linestyle="--")
    plt.xlabel("Actual support_tickets")
    plt.ylabel("Predicted support_tickets")
    plt.title(f"Predicted vs actual - {model_name}")
    plt.tight_layout()
    plt.savefig(figures_dir / "predicted_vs_actual.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    predictions.boxplot(column="absolute_error", by="region")
    plt.suptitle("")
    plt.title(f"Absolute error by region - {model_name}")
    plt.xlabel("Region")
    plt.ylabel("Absolute error")
    plt.tight_layout()
    plt.savefig(figures_dir / "absolute_error_by_region.png", dpi=160)
    plt.close()

    daily = predictions.groupby("date", as_index=False)[[TARGET, "prediction"]].mean()
    plt.figure(figsize=(12, 5))
    plt.plot(daily["date"], daily[TARGET], label="actual")
    plt.plot(daily["date"], daily["prediction"], label="prediction")
    plt.legend()
    plt.title(f"Daily average support tickets - {model_name}")
    plt.xlabel("Date")
    plt.ylabel("Support tickets")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "daily_average_prediction.png", dpi=160)
    plt.close()


def run_baselines(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    traceability_test: pd.DataFrame,
    run_dir: Path,
    logger: logging.Logger,
) -> list[dict[str, object]]:
    results = []
    baselines = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
    }
    for model_name, model in baselines.items():
        logger.info("Training baseline %s", model_name)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        results.append(evaluate_model(pipeline, X_test, y_test, traceability_test, model_name, run_dir, logger))
    return results


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
    results = []
    best_estimator = None
    best_model_name = ""
    best_params: dict[str, object] = {}
    best_mae = np.inf
    cv = TimeSeriesSplit(n_splits=args.cv_splits)

    for model_name, (model, param_distributions) in model_spaces().items():
        logger.info("Starting RandomizedSearchCV for %s", model_name)
        start = perf_counter()
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=args.n_iter,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=args.n_jobs,
            verbose=2,
            refit=True,
        )
        search.fit(X_train, y_train)
        elapsed = perf_counter() - start
        logger.info("Finished %s in %.1fs", model_name, elapsed)
        logger.info("%s best CV MAE: %.4f", model_name, -search.best_score_)
        logger.info("%s best params: %s", model_name, search.best_params_)

        pd.DataFrame(search.cv_results_).to_csv(run_dir / f"cv_results_{model_name}.csv", index=False)
        metrics = evaluate_model(search.best_estimator_, X_test, y_test, traceability_test, model_name, run_dir, logger)
        metrics["best_cv_mae"] = float(-search.best_score_)
        metrics["training_seconds"] = float(elapsed)
        metrics["best_params"] = make_jsonable(search.best_params_)
        results.append(metrics)

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_estimator = search.best_estimator_
            best_model_name = model_name
            best_params = metrics["best_params"]

    if best_estimator is None:
        raise RuntimeError("No model trained successfully")
    return results, best_estimator, best_model_name, best_params


def main() -> None:
    args = parse_args()
    run_dir = EXPERIMENT_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(run_dir)
    logger.info("Support tickets forecasting experiment started")
    logger.info("Run directory: %s", run_dir.relative_to(ROOT_DIR))
    logger.info("Arguments: %s", vars(args))

    df = load_dataset(args.data_path, args.use_server_aggregates, logger)
    df = add_historical_features(df, logger)
    X_train, y_train, X_test, y_test, traceability_test = temporal_split(df, args.test_start_date, logger)
    preprocessor = build_preprocessor(X_train, logger)

    results = run_baselines(preprocessor, X_train, y_train, X_test, y_test, traceability_test, run_dir, logger)
    search_results, best_estimator, best_model_name, best_params = run_searches(
        preprocessor, X_train, y_train, X_test, y_test, traceability_test, args, run_dir, logger
    )
    results.extend(search_results)

    metrics_df = pd.DataFrame(results).sort_values("mae", ascending=True)
    metrics_path = run_dir / "benchmark_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    best_predictions_path = run_dir / f"predictions_test_{best_model_name}.csv"
    best_predictions = pd.read_csv(best_predictions_path)
    best_predictions["date"] = pd.to_datetime(best_predictions["date"])
    plot_best_predictions(best_predictions, run_dir, best_model_name)

    model_path = run_dir / "best_support_model.pkl"
    joblib.dump(best_estimator, model_path)

    summary = {
        "target": TARGET,
        "problem_type": "count_regression",
        "selection_metric": "MAE on temporal test set",
        "best_model": best_model_name,
        "best_params": best_params,
        "best_model_path": str(model_path.relative_to(ROOT_DIR)),
        "metrics_path": str(metrics_path.relative_to(ROOT_DIR)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_target_mean": float(y_train.mean()),
        "test_target_mean": float(y_test.mean()),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Benchmark ranking:\n%s", metrics_df.to_string(index=False))
    logger.info("Best model: %s", best_model_name)
    logger.info("Saved best model to %s", model_path.relative_to(ROOT_DIR))
    logger.info("Support tickets forecasting experiment finished")


if __name__ == "__main__":
    main()
