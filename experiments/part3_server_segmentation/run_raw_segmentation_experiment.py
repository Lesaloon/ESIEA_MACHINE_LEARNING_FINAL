from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part3_server_segmentation"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "ml_training_dataset.csv"
RANDOM_STATE = 42
CATEGORICAL_COLUMNS = ["server_type", "region", "os_family", "segment", "country", "support_plan"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw server segmentation experiment.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", choices=["kmeans", "gaussian_mixture"], default="kmeans")
    parser.add_argument("--clusters", type=int, default=3)
    return parser.parse_args()


def build_server_dataset(df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "cpu_util_pct": ["mean", "max", "std"],
        "ram_util_pct": ["mean", "max", "std"],
        "disk_util_pct": ["mean", "max", "std"],
        "net_in_gb": ["mean", "sum"],
        "net_out_gb": ["mean", "sum"],
        "temperature_c": ["mean", "max", "std"],
        "backup_success": ["mean", "min"],
        "scheduled_maintenance": "mean",
        "network_latency_ms": ["mean", "max"],
        "capacity_used_pct": ["mean", "max"],
        "support_tickets": "mean",
        "cpu_cores": "first",
        "ram_gb": "first",
        "disk_tb": "first",
        "age_days": "first",
        "has_gpu": "first",
        "is_managed": "first",
        "contract_months": "first",
        "tenure_days": "first",
        "monthly_spend_eur": "first",
    }
    server_df = df.groupby("server_id").agg(aggregations)
    server_df.columns = ["_".join(column).strip("_") for column in server_df.columns]
    categorical = df.groupby("server_id")[CATEGORICAL_COLUMNS].agg(lambda values: values.mode().iloc[0])
    server_df = server_df.merge(categorical, on="server_id", how="left").reset_index()
    server_df["backup_failure_rate"] = 1 - server_df["backup_success_mean"]
    server_df["utilization_pressure_mean"] = (
        server_df["cpu_util_pct_mean"] + server_df["ram_util_pct_mean"] + server_df["disk_util_pct_mean"] + server_df["capacity_used_pct_mean"]
    ) / 4
    return server_df.fillna(0)


def build_model(model_name: str, clusters: int):
    if model_name == "gaussian_mixture":
        return GaussianMixture(n_components=clusters, covariance_type="diag", random_state=RANDOM_STATE, n_init=5)
    return KMeans(n_clusters=clusters, random_state=RANDOM_STATE, n_init=20)


def main() -> None:
    args = parse_args()
    df = build_server_dataset(pd.read_csv(args.data_path))
    X = df.drop(columns=["server_id"])
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("model", build_model(args.model, args.clusters))])

    print("Fitting the model...")
    labels = model.fit_predict(X)
    X_processed = model.named_steps["preprocessor"].transform(X)
    metrics = {
        "model": args.model,
        "clusters": args.clusters,
        "rows": int(len(df)),
        "silhouette": float(silhouette_score(X_processed, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_processed, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_processed, labels)),
    }
    print("Silhouette:", metrics["silhouette"])
    print("Davies-Bouldin:", metrics["davies_bouldin"])
    print("Calinski-Harabasz:", metrics["calinski_harabasz"])

    run_dir = EXPERIMENT_DIR / "raw_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / f"raw_segmentation_{args.model}.pkl")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
