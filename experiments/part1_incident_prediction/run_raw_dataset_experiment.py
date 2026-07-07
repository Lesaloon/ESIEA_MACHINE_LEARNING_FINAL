from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part1_incident_prediction"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "ml_training_dataset.csv"
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
    parser = argparse.ArgumentParser(description="Run incident classification on ml_training_dataset.csv.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--cv", type=int, default=4)
    return parser.parse_args()


def encode_categorical_features(df: pd.DataFrame, categorical_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    available_features = [feature for feature in categorical_features if feature in df.columns]
    if not available_features:
        return df.copy(), []

    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded_features = encoder.fit_transform(df[available_features])
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=encoder.get_feature_names_out(available_features),
        index=df.index,
    )
    df = df.drop(columns=available_features)
    df = pd.concat([df, encoded_df], axis=1)
    return df, available_features


def scale_features(df: pd.DataFrame, features_to_scale: list[str]) -> tuple[pd.DataFrame, list[str]]:
    available_features = [feature for feature in features_to_scale if feature in df.columns]
    if not available_features:
        return df.copy(), []

    scaler = StandardScaler()
    df = df.copy()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(df[available_features]),
        columns=available_features,
        index=df.index,
    )
    df = df.drop(columns=available_features)
    df = pd.concat([df, scaled_df.astype(float)], axis=1)
    return df, available_features


def prepare_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    df = pd.read_csv(data_path)
    df_encoded, encoded_features = encode_categorical_features(df, CATEGORICAL_FEATURES)
    df_scaled, scaled_features = scale_features(df_encoded, FEATURES_TO_SCALE)
    df_cleaned = df_scaled.dropna().drop(columns=["date"])

    X = df_cleaned.drop(columns=[TARGET, "overload_anomaly", "server_id", "customer_id"])
    y = df_cleaned[TARGET]
    preprocessing = {
        "encoded_categorical_features": encoded_features,
        "scaled_numeric_features": scaled_features,
        "rows_after_cleaning": int(len(df_cleaned)),
        "feature_count": int(X.shape[1]),
    }
    return X, y, preprocessing


def model_definitions() -> dict[str, object]:
    return {
        "DummyClassifier": DummyClassifier(strategy="uniform", random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def search_spaces() -> dict[str, dict[str, list[object]]]:
    return {
        "RandomForestClassifier": {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 10, None],
            "min_samples_leaf": [1, 3, 5],
            "max_features": ["sqrt", "log2"],
        },
        "GradientBoostingClassifier": {
            "n_estimators": [100, 200, 700],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0],
        },
    }


def evaluate_model(model_name: str, estimator: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    y_pred = estimator.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))

    print(f"{model_name} accuracy: {metrics['accuracy']:.4f}")
    return metrics


def main() -> None:
    args = parse_args()
    X, y, preprocessing = prepare_dataset(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
    )

    models = model_definitions()
    param_spaces = search_spaces()
    results: dict[str, dict[str, object]] = {}
    trained_models: dict[str, object] = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        if model_name in param_spaces:
            print(f"Performing hyperparameter tuning for {model_name}...")
            random_search = RandomizedSearchCV(
                model,
                param_distributions=param_spaces[model_name],
                n_iter=args.n_iter,
                cv=args.cv,
                scoring="accuracy",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=2,
            )
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            best_params: object = random_search.best_params_
            cv_score = float(random_search.best_score_)
        else:
            print(f"No hyperparameter tuning for {model_name}. Using default parameters.")
            best_model = model
            best_model.fit(X_train, y_train)
            best_params = "Default parameters used"
            cv_score = None

        trained_models[model_name] = best_model
        metrics = evaluate_model(model_name, best_model, X_test, y_test)
        results[model_name] = {
            **metrics,
            "best_params": best_params,
            "cv_best_accuracy": cv_score,
        }

    best_model_name = max(results, key=lambda name: results[name]["accuracy"])
    run_dir = EXPERIMENT_DIR / "raw_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models[best_model_name], run_dir / f"incident_classifier_{best_model_name}.pkl")

    payload = {
        "data_source": str(args.data_path.relative_to(ROOT_DIR)),
        "target": TARGET,
        "test_size": args.test_size,
        "n_iter": args.n_iter,
        "cv": args.cv,
        "preprocessing": preprocessing,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "best_model": best_model_name,
        "results": results,
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({name: {"accuracy": values["accuracy"], "best_params": values["best_params"]} for name, values in results.items()}, indent=2))


if __name__ == "__main__":
    main()
