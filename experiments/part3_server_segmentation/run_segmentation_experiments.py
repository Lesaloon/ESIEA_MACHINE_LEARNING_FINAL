from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part3_server_segmentation"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "processed" / "unsupervised_dataset.csv"
RANDOM_STATE = 42

NUMERIC_BASE_COLUMNS = [
    "cpu_cores",
    "ram_gb",
    "disk_tb",
    "age_days",
    "has_gpu",
    "is_managed",
    "cpu_util_pct",
    "ram_util_pct",
    "disk_util_pct",
    "net_in_gb",
    "net_out_gb",
    "temperature_c",
    "backup_success",
    "scheduled_maintenance",
    "avg_rack_temperature_c",
    "power_usage_mw",
    "network_latency_ms",
    "capacity_used_pct",
    "contract_months",
    "tenure_days",
    "monthly_spend_eur",
]

CATEGORICAL_COLUMNS = ["server_type", "region", "os_family", "segment", "country", "support_plan"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment servers into operational profiles.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--min-clusters", type=int, default=3)
    parser.add_argument("--max-clusters", type=int, default=8)
    return parser.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("server_segmentation")
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
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    logger.info("Dataset: %s", data_path.relative_to(ROOT_DIR))
    logger.info("Shape: %s", df.shape)
    logger.info("Date range: %s -> %s", df["date"].min().date(), df["date"].max().date())
    logger.info("Servers: %s", df["server_id"].nunique())
    logger.info("Missing values: %s", int(df.isna().sum().sum()))
    logger.info("Duplicate rows: %s", int(df.duplicated().sum()))
    return df


def build_server_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    aggregations: dict[str, list[str] | str] = {
        "cpu_util_pct": ["mean", "max", "std"],
        "ram_util_pct": ["mean", "max", "std"],
        "disk_util_pct": ["mean", "max", "std"],
        "net_in_gb": ["mean", "max", "sum"],
        "net_out_gb": ["mean", "max", "sum"],
        "temperature_c": ["mean", "max", "std"],
        "backup_success": ["mean", "min"],
        "scheduled_maintenance": "mean",
        "avg_rack_temperature_c": ["mean", "max"],
        "power_usage_mw": ["mean", "max"],
        "network_latency_ms": ["mean", "max", "std"],
        "capacity_used_pct": ["mean", "max", "std"],
        "cpu_cores": "first",
        "ram_gb": "first",
        "disk_tb": "first",
        "age_days": "first",
        "has_gpu": "first",
        "is_managed": "first",
        "contract_months": "first",
        "tenure_days": "first",
        "monthly_spend_eur": "first",
        "date": "count",
    }
    server_df = df.groupby("server_id").agg(aggregations)
    server_df.columns = ["_".join(column).strip("_") for column in server_df.columns]
    server_df = server_df.rename(columns={"date_count": "observation_count"}).reset_index()

    categorical_df = df.groupby("server_id")[CATEGORICAL_COLUMNS].agg(lambda values: values.mode().iloc[0])
    server_df = server_df.merge(categorical_df, on="server_id", how="left")
    server_df = server_df.fillna(0)

    server_df["utilization_pressure_mean"] = (
        server_df["cpu_util_pct_mean"] + server_df["ram_util_pct_mean"] + server_df["disk_util_pct_mean"] + server_df["capacity_used_pct_mean"]
    ) / 4
    server_df["network_total_gb_sum"] = server_df["net_in_gb_sum"] + server_df["net_out_gb_sum"]
    server_df["thermal_cpu_pressure"] = server_df["temperature_c_mean"] * server_df["cpu_util_pct_mean"] / 100
    server_df["backup_failure_rate"] = 1 - server_df["backup_success_mean"]

    logger.info("Server-level feature table: %s", server_df.shape)
    return server_df


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [column for column in feature_df.select_dtypes(include="number").columns]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ]
    )


def make_cluster_model(model_name: str, n_clusters: int) -> object:
    if model_name == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    if model_name == "gaussian_mixture_diag":
        return GaussianMixture(n_components=n_clusters, covariance_type="diag", random_state=RANDOM_STATE, n_init=2, max_iter=150)
    if model_name == "gaussian_mixture_tied":
        return GaussianMixture(n_components=n_clusters, covariance_type="tied", random_state=RANDOM_STATE, n_init=2, max_iter=150)
    if model_name == "birch":
        return Birch(n_clusters=n_clusters)
    if model_name == "agglomerative_ward":
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    raise ValueError(f"Unknown clustering model: {model_name}")


def model_names() -> list[str]:
    return ["kmeans", "gaussian_mixture_diag", "gaussian_mixture_tied", "birch", "agglomerative_ward"]


def fit_predict_labels(model: object, X_processed: Any) -> list[int]:
    if hasattr(model, "fit_predict"):
        return model.fit_predict(X_processed)
    model.fit(X_processed)
    return model.predict(X_processed)


def evaluate_clustering_models(feature_df: pd.DataFrame, run_dir: Path, args: argparse.Namespace, logger: logging.Logger) -> pd.DataFrame:
    preprocessor = build_preprocessor(feature_df)
    X = feature_df.drop(columns=["server_id"])
    X_processed = preprocessor.fit_transform(X)

    results = []
    for model_name in model_names():
        for n_clusters in range(args.min_clusters, args.max_clusters + 1):
            logger.info("Evaluating %s with %s clusters", model_name, n_clusters)
            model = make_cluster_model(model_name, n_clusters)
            labels = fit_predict_labels(model, X_processed)
            if len(set(labels)) < 2:
                logger.info("Skipping %s/%s: only one cluster produced", model_name, n_clusters)
                continue
            results.append(
                {
                    "model_name": model_name,
                    "n_clusters": n_clusters,
                    "inertia": float(getattr(model, "inertia_", 0.0)),
                    "silhouette": float(silhouette_score(X_processed, labels)),
                    "davies_bouldin": float(davies_bouldin_score(X_processed, labels)),
                    "calinski_harabasz": float(calinski_harabasz_score(X_processed, labels)),
                }
            )

    metrics_df = pd.DataFrame(results).sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
    metrics_df.to_csv(run_dir / "model_metrics.csv", index=False)
    logger.info("Model metrics:\n%s", metrics_df.to_string(index=False))
    return metrics_df


def assign_profile_names(profile_df: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
    candidate_profiles = [
        ("utilization_pressure_mean", "serveurs fortement sollicites", 3.0),
        ("disk_util_pct_mean", "serveurs stockage sollicite", 5.0),
        ("network_latency_ms_mean", "serveurs latence elevee", 1.5),
        ("temperature_c_mean", "serveurs temperature elevee", 2.0),
        ("backup_failure_rate", "serveurs instables sauvegarde", 0.01),
        ("monthly_spend_eur_first", "serveurs forte valeur client", 5.0),
    ]
    candidate_columns = [column for column, _, _ in candidate_profiles]
    global_means = profile_df[candidate_columns].mean()

    names = {}
    drivers = {}
    for _, row in profile_df.iterrows():
        cluster = int(row["cluster"])
        scored_candidates = []
        for column, label, minimum_gap in candidate_profiles:
            raw_gap = float(row[column] - global_means[column])
            if raw_gap >= minimum_gap:
                scored_candidates.append((raw_gap / minimum_gap, raw_gap, column, label))

        if not scored_candidates:
            names[cluster] = "serveurs standard"
            drivers[cluster] = "aucun ecart dominant"
        else:
            _, raw_gap, driver, label = sorted(scored_candidates, reverse=True)[0]
            names[cluster] = label
            drivers[cluster] = f"{driver} (+{raw_gap:.2f} vs moyenne clusters)"
    return names, drivers


def fit_final_model(
    feature_df: pd.DataFrame,
    model_name: str,
    n_clusters: int,
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    X = feature_df.drop(columns=["server_id"])
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_df)),
            ("model", make_cluster_model(model_name, n_clusters)),
        ]
    )
    labels = pipeline.fit_predict(X)

    assignments = feature_df[["server_id"]].copy()
    assignments["cluster"] = labels

    enriched = feature_df.copy()
    enriched["cluster"] = labels

    profile_columns = [
        "server_id",
        "utilization_pressure_mean",
        "cpu_util_pct_mean",
        "ram_util_pct_mean",
        "disk_util_pct_mean",
        "capacity_used_pct_mean",
        "network_latency_ms_mean",
        "temperature_c_mean",
        "backup_failure_rate",
        "monthly_spend_eur_first",
        "observation_count",
    ]
    profiles = enriched.groupby("cluster").agg(
        server_count=("server_id", "size"),
        utilization_pressure_mean=("utilization_pressure_mean", "mean"),
        cpu_util_pct_mean=("cpu_util_pct_mean", "mean"),
        ram_util_pct_mean=("ram_util_pct_mean", "mean"),
        disk_util_pct_mean=("disk_util_pct_mean", "mean"),
        capacity_used_pct_mean=("capacity_used_pct_mean", "mean"),
        network_latency_ms_mean=("network_latency_ms_mean", "mean"),
        temperature_c_mean=("temperature_c_mean", "mean"),
        backup_failure_rate=("backup_failure_rate", "mean"),
        monthly_spend_eur_first=("monthly_spend_eur_first", "mean"),
        observation_count=("observation_count", "mean"),
    ).reset_index()
    profile_names, profile_drivers = assign_profile_names(profiles)
    profiles["profile_name"] = profiles["cluster"].map(profile_names)
    profiles["profile_driver"] = profiles["cluster"].map(profile_drivers)
    assignments["profile_name"] = assignments["cluster"].map(profile_names)

    assignments.to_csv(run_dir / "server_cluster_assignments.csv", index=False)
    profiles.to_csv(run_dir / "cluster_profiles.csv", index=False)
    enriched[[*profile_columns, "cluster"]].to_csv(run_dir / "server_features_with_clusters.csv", index=False)
    logger.info("Cluster profiles:\n%s", profiles.to_string(index=False))
    return pipeline, assignments, profiles


def plot_clusters(feature_df: pd.DataFrame, assignments: pd.DataFrame, run_dir: Path) -> None:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    X = feature_df.drop(columns=["server_id"])
    preprocessor = build_preprocessor(feature_df)
    X_processed = preprocessor.fit_transform(X)
    coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_processed)

    plot_df = assignments.copy()
    plot_df["pca_1"] = coordinates[:, 0]
    plot_df["pca_2"] = coordinates[:, 1]

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(plot_df["pca_1"], plot_df["pca_2"], c=plot_df["cluster"], cmap="tab10", alpha=0.75, s=16)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.title("Segmentation des serveurs - projection PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(figures_dir / "server_segments_pca.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = EXPERIMENT_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(run_dir)
    logger.info("Server segmentation experiment started")
    logger.info("Run directory: %s", run_dir.relative_to(ROOT_DIR))
    logger.info("Arguments: %s", vars(args))

    raw_df = load_dataset(args.data_path, logger)
    feature_df = build_server_features(raw_df, logger)
    metrics_df = evaluate_clustering_models(feature_df, run_dir, args, logger)

    best_model_name = str(metrics_df.iloc[0]["model_name"])
    best_n_clusters = int(metrics_df.iloc[0]["n_clusters"])
    pipeline, assignments, profiles = fit_final_model(feature_df, best_model_name, best_n_clusters, run_dir, logger)
    plot_clusters(feature_df, assignments, run_dir)

    model_path = run_dir / f"server_segmentation_{best_model_name}.pkl"
    joblib.dump(pipeline, model_path)

    summary = {
        "problem_type": "unsupervised_server_segmentation",
        "algorithm": best_model_name,
        "selection_metric": "silhouette",
        "best_n_clusters": best_n_clusters,
        "best_silhouette": float(metrics_df.iloc[0]["silhouette"]),
        "best_davies_bouldin": float(metrics_df.iloc[0]["davies_bouldin"]),
        "best_calinski_harabasz": float(metrics_df.iloc[0]["calinski_harabasz"]),
        "server_rows": int(len(feature_df)),
        "source_rows": int(len(raw_df)),
        "model_path": str(model_path.relative_to(ROOT_DIR)),
        "profiles_path": str((run_dir / "cluster_profiles.csv").relative_to(ROOT_DIR)),
        "assignments_path": str((run_dir / "server_cluster_assignments.csv").relative_to(ROOT_DIR)),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Best model: %s", best_model_name)
    logger.info("Best n_clusters: %s", best_n_clusters)
    logger.info("Saved model to %s", model_path.relative_to(ROOT_DIR))
    logger.info("Server segmentation experiment finished")


if __name__ == "__main__":
    main()
