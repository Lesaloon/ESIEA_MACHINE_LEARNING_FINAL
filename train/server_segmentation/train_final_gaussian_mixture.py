from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_ROOT = ROOT_DIR / "train" / "server_segmentation" / "runs"
ARTIFACT_PATH = ROOT_DIR / "models" / "artifacts" / "server_segmentation_gaussian_mixture.pkl"
METADATA_PATH = ROOT_DIR / "models" / "metadata" / "server_segmentation_gaussian_mixture.json"
RANDOM_STATE = 42

SERVERS_PATH = RAW_DIR / "servers.csv"
USAGE_PATH = RAW_DIR / "daily_server_usage.csv"
INCIDENTS_PATH = RAW_DIR / "incidents.csv"

CATEGORICAL_COLUMNS = ["server_type", "region", "os_family"]
USAGE_BASE_COLUMNS = [
    "cpu_util_pct",
    "ram_util_pct",
    "disk_util_pct",
    "net_in_gb",
    "net_out_gb",
    "temperature_c",
    "backup_success",
    "utilization_pressure",
    "thermal_cpu_pressure",
    "network_total_gb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final raw-data server segmentation model.")
    parser.add_argument("--servers-path", type=Path, default=SERVERS_PATH)
    parser.add_argument("--usage-path", type=Path, default=USAGE_PATH)
    parser.add_argument("--incidents-path", type=Path, default=INCIDENTS_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


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


def load_raw_data(
    servers_path: Path,
    usage_path: Path,
    incidents_path: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading raw datasets")
    servers = pd.read_csv(servers_path)
    usage = pd.read_csv(usage_path)
    incidents = pd.read_csv(incidents_path)

    usage["date"] = pd.to_datetime(usage["date"], errors="raise")
    incidents["date"] = pd.to_datetime(incidents["date"], errors="raise")

    logger.info("servers.csv shape: %s", servers.shape)
    logger.info("daily_server_usage.csv shape: %s", usage.shape)
    logger.info("incidents.csv shape: %s", incidents.shape)
    return servers, usage.sort_values(["server_id", "date"]), incidents.sort_values(["server_id", "date"])


def aggregate_usage_features(usage: pd.DataFrame) -> pd.DataFrame:
    usage = usage.copy()
    usage["utilization_pressure"] = (
        usage["cpu_util_pct"] + usage["ram_util_pct"] + usage["disk_util_pct"]
    ) / 3
    usage["thermal_cpu_pressure"] = usage["temperature_c"] * usage["cpu_util_pct"] / 100
    usage["network_total_gb"] = usage["net_in_gb"] + usage["net_out_gb"]

    grouped = usage.groupby("server_id", sort=False)
    aggregated = grouped.agg(
        cpu_util_pct_mean=("cpu_util_pct", "mean"),
        cpu_util_pct_max=("cpu_util_pct", "max"),
        cpu_util_pct_std=("cpu_util_pct", "std"),
        ram_util_pct_mean=("ram_util_pct", "mean"),
        ram_util_pct_max=("ram_util_pct", "max"),
        ram_util_pct_std=("ram_util_pct", "std"),
        disk_util_pct_mean=("disk_util_pct", "mean"),
        disk_util_pct_max=("disk_util_pct", "max"),
        disk_util_pct_std=("disk_util_pct", "std"),
        net_in_gb_mean=("net_in_gb", "mean"),
        net_in_gb_sum=("net_in_gb", "sum"),
        net_out_gb_mean=("net_out_gb", "mean"),
        net_out_gb_sum=("net_out_gb", "sum"),
        temperature_c_mean=("temperature_c", "mean"),
        temperature_c_max=("temperature_c", "max"),
        temperature_c_std=("temperature_c", "std"),
        backup_success_mean=("backup_success", "mean"),
        backup_success_min=("backup_success", "min"),
        utilization_pressure_mean=("utilization_pressure", "mean"),
        thermal_cpu_pressure_mean=("thermal_cpu_pressure", "mean"),
        thermal_cpu_pressure_max=("thermal_cpu_pressure", "max"),
        network_total_gb_mean=("network_total_gb", "mean"),
        network_total_gb_sum=("network_total_gb", "sum"),
        observation_count=("date", "size"),
    )

    for column in [
        "cpu_util_pct",
        "ram_util_pct",
        "disk_util_pct",
        "temperature_c",
        "net_in_gb",
        "net_out_gb",
        "utilization_pressure",
        "thermal_cpu_pressure",
        "network_total_gb",
    ]:
        aggregated[f"{column}_p90"] = grouped[column].quantile(0.90)

    recent7 = grouped.tail(7).groupby("server_id", sort=False).agg(
        cpu_util_pct_recent7_mean=("cpu_util_pct", "mean"),
        ram_util_pct_recent7_mean=("ram_util_pct", "mean"),
        disk_util_pct_recent7_mean=("disk_util_pct", "mean"),
        temperature_c_recent7_mean=("temperature_c", "mean"),
        net_in_gb_recent7_mean=("net_in_gb", "mean"),
        net_out_gb_recent7_mean=("net_out_gb", "mean"),
        utilization_pressure_recent7_mean=("utilization_pressure", "mean"),
        thermal_cpu_pressure_recent7_mean=("thermal_cpu_pressure", "mean"),
        network_total_gb_recent7_mean=("network_total_gb", "mean"),
    )

    rates = grouped.agg(
        high_cpu_day_rate=("cpu_util_pct", lambda values: float((values >= 85).mean())),
        high_ram_day_rate=("ram_util_pct", lambda values: float((values >= 85).mean())),
        high_temperature_day_rate=("temperature_c", lambda values: float((values >= 70).mean())),
        backup_failure_rate=("backup_success", lambda values: float((1 - values).mean())),
    )

    return aggregated.join(recent7).join(rates).reset_index()


def aggregate_incident_features(
    incidents: pd.DataFrame,
    reference_date: pd.Timestamp,
    all_server_ids: pd.Series,
) -> pd.DataFrame:
    if incidents.empty:
        incident_df = pd.DataFrame({"server_id": all_server_ids})
        incident_df["incident_count"] = 0
        incident_df["incident_day_count"] = 0
        incident_df["incident_duration_minutes_mean"] = 0.0
        incident_df["incident_duration_minutes_max"] = 0.0
        incident_df["incident_duration_minutes_sum"] = 0.0
        incident_df["customer_visible_rate"] = 0.0
        incident_df["sla_breach_rate"] = 0.0
        incident_df["root_cause_known_rate"] = 0.0
        incident_df["critical_incident_count"] = 0
        incident_df["high_severity_incident_rate"] = 0.0
        incident_df["disk_incident_count"] = 0
        incident_df["network_incident_count"] = 0
        incident_df["hypervisor_incident_count"] = 0
        incident_df["recent30_incident_count"] = 0
        incident_df["recent30_high_severity_count"] = 0
        incident_df["days_since_last_incident"] = 999
        return incident_df

    incidents = incidents.copy()
    incidents["is_high_severity"] = incidents["severity"].isin(["high", "critical"]).astype(int)
    incidents["is_critical"] = (incidents["severity"] == "critical").astype(int)
    incidents["is_disk"] = (incidents["component"] == "disk").astype(int)
    incidents["is_network"] = (incidents["component"] == "network").astype(int)
    incidents["is_hypervisor"] = (incidents["component"] == "hypervisor").astype(int)
    incidents["is_recent30"] = (incidents["date"] >= reference_date - pd.Timedelta(days=29)).astype(int)
    incidents["recent30_high_severity"] = incidents["is_recent30"] * incidents["is_high_severity"]

    aggregated = incidents.groupby("server_id", sort=False).agg(
        incident_count=("incident_id", "size"),
        incident_day_count=("date", "nunique"),
        incident_duration_minutes_mean=("duration_minutes", "mean"),
        incident_duration_minutes_max=("duration_minutes", "max"),
        incident_duration_minutes_sum=("duration_minutes", "sum"),
        customer_visible_rate=("customer_visible", "mean"),
        sla_breach_rate=("sla_breach", "mean"),
        root_cause_known_rate=("root_cause_known", "mean"),
        critical_incident_count=("is_critical", "sum"),
        high_severity_incident_rate=("is_high_severity", "mean"),
        disk_incident_count=("is_disk", "sum"),
        network_incident_count=("is_network", "sum"),
        hypervisor_incident_count=("is_hypervisor", "sum"),
        recent30_incident_count=("is_recent30", "sum"),
        recent30_high_severity_count=("recent30_high_severity", "sum"),
        last_incident_date=("date", "max"),
    )
    aggregated["days_since_last_incident"] = (
        reference_date - aggregated["last_incident_date"]
    ).dt.days.astype(float)
    aggregated = aggregated.drop(columns=["last_incident_date"])

    incident_df = pd.DataFrame({"server_id": all_server_ids}).merge(
        aggregated.reset_index(), on="server_id", how="left"
    )
    incident_df = incident_df.fillna(
        {
            "incident_count": 0,
            "incident_day_count": 0,
            "incident_duration_minutes_mean": 0.0,
            "incident_duration_minutes_max": 0.0,
            "incident_duration_minutes_sum": 0.0,
            "customer_visible_rate": 0.0,
            "sla_breach_rate": 0.0,
            "root_cause_known_rate": 0.0,
            "critical_incident_count": 0,
            "high_severity_incident_rate": 0.0,
            "disk_incident_count": 0,
            "network_incident_count": 0,
            "hypervisor_incident_count": 0,
            "recent30_incident_count": 0,
            "recent30_high_severity_count": 0,
            "days_since_last_incident": 999,
        }
    )
    return incident_df


def build_server_feature_table(
    servers: pd.DataFrame,
    usage: pd.DataFrame,
    incidents: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    logger.info("Building server-level segmentation dataset from raw files")
    servers = servers.drop(columns=["customer_id"], errors="ignore").copy()
    usage_features = aggregate_usage_features(usage)
    incident_features = aggregate_incident_features(incidents, usage["date"].max(), servers["server_id"])

    feature_df = servers.merge(usage_features, on="server_id", how="left", validate="one_to_one")
    feature_df = feature_df.merge(incident_features, on="server_id", how="left", validate="one_to_one")

    feature_df["incident_rate"] = np.divide(
        feature_df["incident_count"],
        feature_df["observation_count"],
        out=np.zeros(len(feature_df), dtype=float),
        where=feature_df["observation_count"].fillna(0).to_numpy() > 0,
    )
    feature_df["recent30_incident_rate"] = np.divide(
        feature_df["recent30_incident_count"],
        np.minimum(feature_df["observation_count"].fillna(0), 30),
        out=np.zeros(len(feature_df), dtype=float),
        where=np.minimum(feature_df["observation_count"].fillna(0), 30).to_numpy() > 0,
    )
    feature_df["recent_usage_shift"] = feature_df["cpu_util_pct_recent7_mean"] - feature_df["cpu_util_pct_mean"]
    feature_df["recent_temperature_shift"] = feature_df["temperature_c_recent7_mean"] - feature_df["temperature_c_mean"]
    feature_df["recent_network_shift"] = feature_df["network_total_gb_recent7_mean"] - feature_df["network_total_gb_mean"]

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    logger.info("Server-level feature table shape: %s", feature_df.shape)
    return feature_df


def split_dataset(
    feature_df: pd.DataFrame,
    test_size: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(feature_df, test_size=test_size, random_state=RANDOM_STATE)
    logger.info("Train servers: %s | Test servers: %s", len(train_df), len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def silhouette_scorer(estimator: Pipeline, X: pd.DataFrame, y: Any = None) -> float:
    transformed = estimator[:-1].transform(X)
    labels = estimator.predict(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        return -1.0
    return float(silhouette_score(transformed, labels))


def clustering_metrics(estimator: Pipeline, X: pd.DataFrame) -> dict[str, float]:
    transformed = estimator[:-1].transform(X)
    labels = estimator.predict(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        return {
            "n_clusters": float(len(unique_labels)),
            "silhouette": -1.0,
            "davies_bouldin": float("inf"),
            "calinski_harabasz": 0.0,
        }

    metrics = {
        "n_clusters": float(len(unique_labels)),
        "silhouette": float(silhouette_score(transformed, labels)),
        "davies_bouldin": float(davies_bouldin_score(transformed, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(transformed, labels)),
    }
    model = estimator.named_steps["model"]
    if hasattr(model, "bic"):
        metrics["bic"] = float(model.bic(transformed))
    if hasattr(model, "aic"):
        metrics["aic"] = float(model.aic(transformed))
    return metrics


def model_spaces() -> dict[str, tuple[object, dict[str, object]]]:
    reducers = ["passthrough", PCA(n_components=0.90, random_state=RANDOM_STATE), PCA(n_components=0.95, random_state=RANDOM_STATE)]
    return {
        "gaussian_mixture": (
            GaussianMixture(random_state=RANDOM_STATE),
            {
                "reducer": reducers,
                "model__n_components": randint(2, 8),
                "model__covariance_type": ["full", "diag", "tied"],
                "model__reg_covar": loguniform(1e-6, 1e-2),
                "model__n_init": randint(2, 10),
                "model__max_iter": randint(100, 400),
            },
        ),
        "kmeans": (
            KMeans(random_state=RANDOM_STATE),
            {
                "reducer": reducers,
                "model__n_clusters": randint(2, 8),
                "model__n_init": randint(10, 40),
                "model__max_iter": randint(200, 500),
                "model__init": ["k-means++", "random"],
            },
        ),
    }


def run_searches(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    args: argparse.Namespace,
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[list[dict[str, object]], Pipeline, str, dict[str, object]]:
    cv = KFold(n_splits=args.cv_splits, shuffle=True, random_state=RANDOM_STATE)
    dummy_y = np.zeros(len(X_train))
    results: list[dict[str, object]] = []
    best_estimator: Pipeline | None = None
    best_model_name = ""
    best_params: dict[str, object] = {}
    best_score = -np.inf

    for model_name, (model, param_distributions) in model_spaces().items():
        logger.info("Starting RandomizedSearchCV for %s", model_name)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("variance_filter", VarianceThreshold()),
                ("reducer", "passthrough"),
                ("model", model),
            ]
        )
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=args.n_iter,
            scoring=silhouette_scorer,
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=args.n_jobs,
            verbose=2,
            refit=True,
        )
        start = perf_counter()
        search.fit(X_train, dummy_y)
        elapsed = perf_counter() - start
        logger.info("Finished %s in %.1fs", model_name, elapsed)
        logger.info("%s best CV silhouette: %.4f", model_name, search.best_score_)
        logger.info("%s best params: %s", model_name, search.best_params_)

        pd.DataFrame(search.cv_results_).to_csv(run_dir / f"cv_results_{model_name}.csv", index=False)

        train_metrics = clustering_metrics(search.best_estimator_, X_train)
        test_metrics = clustering_metrics(search.best_estimator_, X_test)
        result = {
            "model": model_name,
            "best_cv_silhouette": float(search.best_score_),
            "training_seconds": float(elapsed),
            "best_params": make_jsonable(search.best_params_),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        results.append(result)

        if test_metrics["silhouette"] > best_score:
            best_score = test_metrics["silhouette"]
            best_estimator = search.best_estimator_
            best_model_name = model_name
            best_params = result["best_params"] if isinstance(result["best_params"], dict) else {}

    if best_estimator is None:
        raise RuntimeError("No segmentation model trained successfully")
    return results, best_estimator, best_model_name, best_params


def assign_profile_names(profile_df: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
    candidate_profiles = [
        ("incident_rate", "serveurs incidents frequents", 0.02),
        ("utilization_pressure_mean", "serveurs forte charge compute", 4.0),
        ("temperature_c_mean", "serveurs sous stress thermique", 2.0),
        ("disk_util_pct_mean", "serveurs stockage sollicite", 5.0),
        ("network_total_gb_mean", "serveurs trafic eleve", 20.0),
        ("backup_failure_rate", "serveurs sauvegarde fragile", 0.02),
    ]
    candidate_columns = [column for column, _, _ in candidate_profiles]
    global_means = profile_df[candidate_columns].mean()

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


def build_cluster_outputs(
    estimator: Pipeline,
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = feature_df.drop(columns=["server_id"])
    labels = estimator.predict(X)

    assignments = feature_df[["server_id"]].copy()
    assignments["cluster"] = labels

    enriched = feature_df.copy()
    enriched["cluster"] = labels
    profiles = enriched.groupby("cluster").agg(
        server_count=("server_id", "size"),
        incident_rate=("incident_rate", "mean"),
        recent30_incident_rate=("recent30_incident_rate", "mean"),
        utilization_pressure_mean=("utilization_pressure_mean", "mean"),
        cpu_util_pct_mean=("cpu_util_pct_mean", "mean"),
        ram_util_pct_mean=("ram_util_pct_mean", "mean"),
        disk_util_pct_mean=("disk_util_pct_mean", "mean"),
        network_total_gb_mean=("network_total_gb_mean", "mean"),
        temperature_c_mean=("temperature_c_mean", "mean"),
        backup_failure_rate=("backup_failure_rate", "mean"),
        incident_duration_minutes_mean=("incident_duration_minutes_mean", "mean"),
        days_since_last_incident=("days_since_last_incident", "mean"),
        observation_count=("observation_count", "mean"),
    ).reset_index()
    profile_names, profile_drivers = assign_profile_names(profiles)
    profiles["profile_name"] = profiles["cluster"].map(profile_names)
    profiles["profile_driver"] = profiles["cluster"].map(profile_drivers)
    assignments["profile_name"] = assignments["cluster"].map(profile_names)
    return assignments, profiles


def save_pca_plot(feature_df: pd.DataFrame, estimator: Pipeline, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    X = feature_df.drop(columns=["server_id"])
    transformed = estimator[:-1].transform(X)
    labels = estimator.predict(X)
    coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(transformed)
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
    args = parse_args()
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(output_dir)
    logger.info("Training final raw-data server segmentation model")
    logger.info("Arguments: %s", vars(args))

    servers, usage, incidents = load_raw_data(args.servers_path, args.usage_path, args.incidents_path, logger)
    feature_df = build_server_feature_table(servers, usage, incidents, logger)
    feature_df.to_csv(output_dir / "server_feature_table.csv", index=False)

    train_df, test_df = split_dataset(feature_df, args.test_size, logger)
    X_train = train_df.drop(columns=["server_id"])
    X_test = test_df.drop(columns=["server_id"])

    results, best_estimator, best_model_name, best_params = run_searches(X_train, X_test, args, output_dir, logger)
    metrics_df = pd.DataFrame(results).sort_values("test_silhouette", ascending=False)
    metrics_df.to_csv(output_dir / "benchmark_metrics.csv", index=False)

    best_pipeline = Pipeline(best_estimator.steps)
    best_pipeline.fit(feature_df.drop(columns=["server_id"]), np.zeros(len(feature_df)))

    assignments, profiles = build_cluster_outputs(best_pipeline, feature_df)
    assignments.to_csv(output_dir / "server_cluster_assignments.csv", index=False)
    profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)
    save_pca_plot(feature_df, best_pipeline, output_dir)

    run_model_path = output_dir / f"server_segmentation_{best_model_name}.pkl"
    joblib.dump(best_pipeline, run_model_path)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, ARTIFACT_PATH)

    full_metrics = clustering_metrics(best_pipeline, feature_df.drop(columns=["server_id"]))
    metadata = {
        "name": "server_segmentation_model",
        "problem": "server segmentation clustering",
        "artifact_path": str(ARTIFACT_PATH.relative_to(ROOT_DIR)),
        "saved_with": "joblib",
        "model": best_model_name,
        "model_type": best_pipeline.named_steps["model"].__class__.__name__,
        "selection_metric": "test_silhouette",
        "test_size": args.test_size,
        "n_iter": args.n_iter,
        "cv_splits": args.cv_splits,
        "n_servers": int(len(feature_df)),
        "feature_columns": feature_df.drop(columns=["server_id"]).columns.tolist(),
        "data_sources": {
            "servers": str(args.servers_path.relative_to(ROOT_DIR)),
            "daily_server_usage": str(args.usage_path.relative_to(ROOT_DIR)),
            "incidents": str(args.incidents_path.relative_to(ROOT_DIR)),
        },
        "best_params": best_params,
        "search_results": results,
        "profiles": profiles.to_dict(orient="records"),
        "source_run": str(output_dir.relative_to(ROOT_DIR)),
        **full_metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Selected best model: %s", best_model_name)
    logger.info("Saved model to %s", ARTIFACT_PATH.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
