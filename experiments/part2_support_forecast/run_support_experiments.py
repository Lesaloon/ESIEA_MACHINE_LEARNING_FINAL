from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part2_support_forecast"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "processed" / "support_dataset.csv"
TARGET = "support_tickets_next_1d"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple support ticket forecasting experiment.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model", choices=["gradient_boosting", "random_forest"], default="gradient_boosting")
    return parser.parse_args()


def build_model(model_name: str) -> GradientBoostingRegressor | RandomForestRegressor:
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            min_samples_leaf=5,
            n_jobs=-1,
        )

    return GradientBoostingRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        min_samples_leaf=5,
    )


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_path)

    X = df.drop(columns=[TARGET, "date"])
    y = df[TARGET]

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    support_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_model(args.model)),
        ]
    )

    print("Fitting the model...")
    support_model.fit(X_train, y_train)

    print("Predicting on the test set...")
    y_pred = np.clip(support_model.predict(X_test), 0, None)

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

    run_dir = EXPERIMENT_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(support_model, run_dir / "support_forecast_model.pkl")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
