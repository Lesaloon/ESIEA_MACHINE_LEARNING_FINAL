from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "unsupervised_dataset.csv"
OUTPUT_ROOT = ROOT_DIR / "train" / "server_segmentation" / "runs"
ARTIFACT_PATH = ROOT_DIR / "models" / "artifacts" / "server_segmentation_gaussian_mixture.pkl"
METADATA_PATH = ROOT_DIR / "models" / "metadata" / "server_segmentation_gaussian_mixture.json"
RANDOM_STATE = 42
N_CLUSTERS = 3

CATEGORICAL_COLUMNS = ["server_type", "region", "os_family", "segment", "country", "support_plan"]


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("final_server_segmentation")
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


def build_server_features(df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
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
    server_df = server_df.merge(categorical_df, on="server_id", how="left").fillna(0)
    server_df["utilization_pressure_mean"] = (
        server_df["cpu_util_pct_mean"] + server_df["ram_util_pct_mean"] + server_df["disk_util_pct_mean"] + server_df["capacity_used_pct_mean"]
    ) / 4
    server_df["network_total_gb_sum"] = server_df["net_in_gb_sum"] + server_df["net_out_gb_sum"]
    server_df["thermal_cpu_pressure"] = server_df["temperature_c_mean"] * server_df["cpu_util_pct_mean"] / 100
    server_df["backup_failure_rate"] = 1 - server_df["backup_success_mean"]
    return server_df


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    numeric_features = feature_df.drop(columns=["server_id"]).select_dtypes(include="number").columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ]
    )


def assign_profile_names(profile_df: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
    candidate_profiles = [
        ("utilization_pressure_mean", "serveurs fortement sollicites", 3.0),
        ("disk_util_pct_mean", "serveurs stockage sollicite", 5.0),
        ("network_latency_ms_mean", "serveurs latence elevee", 1.5),
        ("temperature_c_mean", "serveurs temperature elevee", 2.0),
        ("backup_failure_rate", "serveurs instables sauvegarde", 0.01),
        ("monthly_spend_eur_first", "serveurs forte valeur client", 5.0),
    ]
    columns = [column for column, _, _ in candidate_profiles]
    global_means = profile_df[columns].mean()
    names = {}
    drivers = {}
    for _, row in profile_df.iterrows():
        cluster = int(row["cluster"])
        scored = []
        for column, label, minimum_gap in candidate_profiles:
            raw_gap = float(row[column] - global_means[column])
            if raw_gap >= minimum_gap:
                scored.append((raw_gap / minimum_gap, raw_gap, column, label))
        if not scored:
            names[cluster] = "serveurs standard"
            drivers[cluster] = "aucun ecart dominant"
        else:
            _, raw_gap, driver, label = sorted(scored, reverse=True)[0]
            names[cluster] = label
            drivers[cluster] = f"{driver} (+{raw_gap:.2f} vs moyenne clusters)"
    return names, drivers


def save_pca_plot(feature_df: pd.DataFrame, labels: list[int], output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    X = feature_df.drop(columns=["server_id"])
    X_processed = build_preprocessor(feature_df).fit_transform(X)
    coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_processed)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="tab10", alpha=0.75, s=16)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.title("Segmentation des serveurs - projection PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(figures_dir / "server_segments_pca.png", dpi=160)
    plt.close()


def main() -> None:
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(output_dir)
    logger.info("Training final GaussianMixture server segmentation model")

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    feature_df = build_server_features(df)
    X = feature_df.drop(columns=["server_id"])

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_df)),
            ("model", GaussianMixture(n_components=N_CLUSTERS, covariance_type="diag", random_state=RANDOM_STATE, n_init=2, max_iter=150)),
        ]
    )
    labels = pipeline.fit_predict(X)
    X_processed = pipeline.named_steps["preprocessor"].transform(X)

    assignments = feature_df[["server_id"]].copy()
    assignments["cluster"] = labels
    enriched = feature_df.copy()
    enriched["cluster"] = labels
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

    metrics = {
        "silhouette": float(silhouette_score(X_processed, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_processed, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_processed, labels)),
    }
    logger.info("Metrics: %s", metrics)
    logger.info("Profiles:\n%s", profiles.to_string(index=False))

    assignments.to_csv(output_dir / "server_cluster_assignments.csv", index=False)
    profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)
    save_pca_plot(feature_df, labels, output_dir)

    run_model_path = output_dir / "server_segmentation_gaussian_mixture.pkl"
    joblib.dump(pipeline, run_model_path)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_PATH)

    metadata = {
        "name": "server_segmentation_gaussian_mixture",
        "problem": "server segmentation clustering",
        "artifact_path": str(ARTIFACT_PATH.relative_to(ROOT_DIR)),
        "saved_with": "joblib",
        "model_type": "GaussianMixture",
        "covariance_type": "diag",
        "n_clusters": N_CLUSTERS,
        "selection_metric": "silhouette",
        **metrics,
        "profiles": profiles.to_dict(orient="records"),
        "source_run": str(output_dir.relative_to(ROOT_DIR)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved model to %s", ARTIFACT_PATH.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
