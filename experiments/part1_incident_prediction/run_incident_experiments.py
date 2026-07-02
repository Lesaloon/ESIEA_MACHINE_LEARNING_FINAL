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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part1_incident_prediction"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "processed" / "incident_dataset.csv"
TARGET = "incident_next_7d"
TRACEABILITY_COLUMNS = ["date", "server_id"]
TEST_START_DATE = "2026-03-01"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark models for incident_next_7d prediction."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-start-date", default=TEST_START_DATE)
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-rows", type=int, default=None)
    return parser.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("incident_experiments")
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


def load_dataset(data_path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading dataset from %s", data_path.relative_to(ROOT_DIR))
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Run `python scripts/preprocess_data.py` first."
        )

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Dataset shape: %s rows, %s columns", df.shape[0], df.shape[1])
    logger.info("Date range: %s -> %s", df["date"].min().date(), df["date"].max().date())
    logger.info("Target positive rate: %.4f", df[TARGET].mean())
    logger.info("Missing values: %s", int(df.isna().sum().sum()))
    logger.info("Duplicate rows: %s", int(df.duplicated().sum()))
    return df


def split_temporal(
    df: pd.DataFrame,
    test_start_date: str,
    max_train_rows: int | None,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    test_start = pd.Timestamp(test_start_date)
    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()

    if max_train_rows is not None and len(train_df) > max_train_rows:
        train_df = train_df.tail(max_train_rows).copy()
        logger.info("Using only the latest %s training rows for faster experimentation", max_train_rows)

    logger.info("Temporal split: train < %s, test >= %s", test_start.date(), test_start.date())
    logger.info("Train shape: %s | positive rate: %.4f", train_df.shape, train_df[TARGET].mean())
    logger.info("Test shape: %s | positive rate: %.4f", test_df.shape, test_df[TARGET].mean())

    drop_columns = [TARGET, *TRACEABILITY_COLUMNS]
    X_train = train_df.drop(columns=drop_columns)
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=drop_columns)
    y_test = test_df[TARGET]
    return X_train, y_train, X_test, y_test


def build_preprocessor(X_train: pd.DataFrame, logger: logging.Logger) -> ColumnTransformer:
    categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()

    logger.info("Numeric features (%s): %s", len(numeric_features), numeric_features)
    logger.info("Categorical features (%s): %s", len(categorical_features), categorical_features)

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


def model_search_spaces() -> dict[str, tuple[object, dict[str, object]]]:
    return {
        "logistic_regression": (
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            ),
            {
                "model__C": loguniform(1e-3, 1e2),
            },
        ),
        "random_forest": (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": randint(150, 450),
                "model__max_depth": [None, 6, 10, 14, 18],
                "model__min_samples_split": randint(2, 16),
                "model__min_samples_leaf": randint(1, 12),
                "model__max_features": ["sqrt", "log2", 0.5, 0.8],
            },
        ),
        "extra_trees": (
            ExtraTreesClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": randint(150, 450),
                "model__max_depth": [None, 6, 10, 14, 18],
                "model__min_samples_split": randint(2, 16),
                "model__min_samples_leaf": randint(1, 12),
                "model__max_features": ["sqrt", "log2", 0.5, 0.8],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__learning_rate": loguniform(0.01, 0.25),
                "model__max_iter": randint(80, 260),
                "model__max_leaf_nodes": randint(15, 63),
                "model__min_samples_leaf": randint(10, 80),
                "model__l2_regularization": uniform(0.0, 1.0),
            },
        ),
    }


def positive_class_scores(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        score_range = scores.max() - scores.min()
        if score_range == 0:
            return np.zeros_like(scores)
        return (scores - scores.min()) / score_range
    return estimator.predict(X)


def evaluate_model(
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, object]:
    y_pred = estimator.predict(X_test)
    y_score = positive_class_scores(estimator, X_test)

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_positive": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_positive": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_positive": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "average_precision": float(average_precision_score(y_test, y_score)),
    }

    logger.info(
        "%s test metrics | AP=%.4f ROC-AUC=%.4f F1+=%.4f Recall+=%.4f Precision+=%.4f",
        model_name,
        metrics["average_precision"],
        metrics["roc_auc"],
        metrics["f1_positive"],
        metrics["recall_positive"],
        metrics["precision_positive"],
    )

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    (run_dir / "classification_reports").mkdir(exist_ok=True)
    (run_dir / "classification_reports" / f"{model_name}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {model_name}")
    fig.tight_layout()
    (run_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(run_dir / "figures" / f"confusion_matrix_{model_name}.png", dpi=160)
    plt.close(fig)

    return metrics


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_jsonable(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: make_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    return value


def run_baseline(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, object]:
    logger.info("Training baseline DummyClassifier(strategy='prior')")
    baseline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="prior", random_state=RANDOM_STATE)),
        ]
    )
    baseline.fit(X_train, y_train)
    return evaluate_model(baseline, X_test, y_test, "dummy_prior", run_dir, logger)


def run_random_searches(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    args: argparse.Namespace,
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[list[dict[str, object]], Pipeline, str, dict[str, object]]:
    results = []
    best_estimator = None
    best_model_name = ""
    best_params: dict[str, object] = {}
    best_score = -np.inf
    cv = TimeSeriesSplit(n_splits=args.cv_splits)

    for model_name, (model, param_distributions) in model_search_spaces().items():
        logger.info("Starting RandomizedSearchCV for %s", model_name)
        start = perf_counter()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
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

        cv_results = pd.DataFrame(search.cv_results_)
        cv_results.to_csv(run_dir / f"cv_results_{model_name}.csv", index=False)

        metrics = evaluate_model(search.best_estimator_, X_test, y_test, model_name, run_dir, logger)
        metrics["best_cv_average_precision"] = float(search.best_score_)
        metrics["training_seconds"] = float(elapsed)
        metrics["best_params"] = make_jsonable(search.best_params_)
        results.append(metrics)

        if metrics["average_precision"] > best_score:
            best_score = metrics["average_precision"]
            best_estimator = search.best_estimator_
            best_model_name = model_name
            best_params = metrics["best_params"]

    if best_estimator is None:
        raise RuntimeError("No model was trained successfully.")

    return results, best_estimator, best_model_name, best_params


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = EXPERIMENT_DIR / "runs" / timestamp
    logger = setup_logging(run_dir)

    logger.info("Incident prediction experiment started")
    logger.info("Run directory: %s", run_dir.relative_to(ROOT_DIR))
    logger.info("Arguments: %s", vars(args))

    df = load_dataset(args.data_path, logger)
    X_train, y_train, X_test, y_test = split_temporal(
        df, args.test_start_date, args.max_train_rows, logger
    )
    preprocessor = build_preprocessor(X_train, logger)

    benchmark_results = [
        run_baseline(preprocessor, X_train, y_train, X_test, y_test, run_dir, logger)
    ]
    search_results, best_estimator, best_model_name, best_params = run_random_searches(
        preprocessor, X_train, y_train, X_test, y_test, args, run_dir, logger
    )
    benchmark_results.extend(search_results)

    metrics_df = pd.DataFrame(benchmark_results).sort_values(
        "average_precision", ascending=False
    )
    metrics_df.to_csv(run_dir / "benchmark_metrics.csv", index=False)

    model_path = run_dir / "best_incident_model.joblib"
    joblib.dump(best_estimator, model_path)

    summary = {
        "target": TARGET,
        "problem_type": "binary_classification",
        "selection_metric": "average_precision on temporal test set",
        "best_model": best_model_name,
        "best_params": best_params,
        "best_model_path": str(model_path.relative_to(ROOT_DIR)),
        "test_start_date": args.test_start_date,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "metrics_path": str((run_dir / "benchmark_metrics.csv").relative_to(ROOT_DIR)),
    }
    save_json(run_dir / "run_summary.json", summary)

    logger.info("Benchmark ranking:")
    logger.info("\n%s", metrics_df.to_string(index=False))
    logger.info("Best model: %s", best_model_name)
    logger.info("Saved best model to %s", model_path.relative_to(ROOT_DIR))
    logger.info("Incident prediction experiment finished")


if __name__ == "__main__":
    main()
