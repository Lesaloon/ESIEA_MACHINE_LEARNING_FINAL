from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part2_support_forecast"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "ml_training_dataset.csv"
TARGET = "support_tickets_next_1d"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw support forecast experiment.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model", choices=["gradient_boosting", "random_forest", "extra_trees"], default="extra_trees")
    return parser.parse_args()


def build_daily_region_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    aggregations = {
        "support_tickets": "mean",
        "scheduled_maintenance": "mean",
        "avg_rack_temperature_c": "mean",
        "power_usage_mw": "mean",
        "network_latency_ms": "mean",
        "capacity_used_pct": "mean",
        "cpu_util_pct": ["mean", "max"],
        "ram_util_pct": ["mean", "max"],
        "disk_util_pct": ["mean", "max"],
        "temperature_c": ["mean", "max"],
        "backup_success": "mean",
        "server_id": "nunique",
        "has_gpu": "mean",
        "is_managed": "mean",
        "monthly_spend_eur": "mean",
    }
    region_df = df.groupby(["date", "region"]).agg(aggregations)
    region_df.columns = ["_".join(column).strip("_") for column in region_df.columns]
    region_df = region_df.rename(columns={"server_id_nunique": "server_count"}).reset_index()
    region_df = region_df.sort_values(["region", "date"])
    region_df[TARGET] = region_df.groupby("region")["support_tickets_mean"].shift(-1)
    return region_df.dropna(subset=[TARGET]).reset_index(drop=True)


def build_model(model_name: str):
    if model_name == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, min_samples_leaf=5, n_jobs=-1)
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=300, random_state=RANDOM_STATE, min_samples_leaf=5)
    return ExtraTreesRegressor(n_estimators=400, random_state=RANDOM_STATE, min_samples_leaf=3, n_jobs=-1)


def main() -> None:
    args = parse_args()
    df = build_daily_region_dataset(pd.read_csv(args.data_path))

    X = df.drop(columns=[TARGET, "date"])
    y = df[TARGET]
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=RANDOM_STATE)
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
    y_pred = np.clip(model.predict(X_test), 0, None)

    metrics = {
        "model": args.model,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }
    print("MAE:", metrics["mae"])
    print("RMSE:", metrics["rmse"])
    print("R2:", metrics["r2"])

    run_dir = EXPERIMENT_DIR / "raw_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / f"raw_support_{args.model}.pkl")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
