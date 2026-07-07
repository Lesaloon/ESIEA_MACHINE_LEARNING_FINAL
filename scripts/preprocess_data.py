from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

ML_DATASET_PATH = RAW_DIR / "ml_training_dataset.csv"
REGION_METRICS_PATH = RAW_DIR / "technical_region_metrics.csv"
CUSTOMERS_PATH = RAW_DIR / "customers.csv"
SERVERS_PATH = RAW_DIR / "servers.csv"
DAILY_USAGE_PATH = RAW_DIR / "daily_server_usage.csv"
INCIDENTS_PATH = RAW_DIR / "incidents.csv"

TARGET_INCIDENT = "incident_next_7d"
TARGET_SUPPORT = "support_tickets"
TARGET_ANOMALY = "overload_anomaly"
TARGET_SUPPORT_NEXT_DAY = "support_tickets_next_1d"

REGION_METRIC_COLUMNS = [
    "scheduled_maintenance",
    "avg_rack_temperature_c",
    "power_usage_mw",
    "network_latency_ms",
    "support_tickets",
    "capacity_used_pct",
]


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    start_date = df["date"].min()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["days_since_start"] = (df["date"] - start_date).dt.days
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def missing_values_summary(df: pd.DataFrame) -> dict[str, int]:
    missing = df.isna().sum()
    return {column: int(count) for column, count in missing.items() if count > 0}


def quality_report(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": missing_values_summary(df),
        "date_min": str(pd.to_datetime(df["date"]).min().date()),
        "date_max": str(pd.to_datetime(df["date"]).max().date()),
        "n_servers": int(df["server_id"].nunique()),
        "n_customers": int(df["customer_id"].nunique()),
        "n_regions": int(df["region"].nunique()),
        "incident_next_7d_rate": float(df[TARGET_INCIDENT].mean()),
        "overload_anomaly_rate": float(df[TARGET_ANOMALY].mean()),
    }


def build_server_day_base(
    usage: pd.DataFrame,
    servers: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    base = usage.copy()
    base["date"] = pd.to_datetime(base["date"], errors="raise")
    server_columns = [
        "server_id",
        "customer_id",
        "server_type",
        "region",
        "os_family",
        "cpu_cores",
        "ram_gb",
        "disk_tb",
        "age_days",
        "has_gpu",
        "is_managed",
    ]
    customer_columns = [
        "customer_id",
        "segment",
        "country",
        "contract_months",
        "support_plan",
        "tenure_days",
        "monthly_spend_eur",
    ]
    base = base.merge(servers[server_columns], on="server_id", how="left", validate="many_to_one")
    base = base.merge(customers[customer_columns], on="customer_id", how="left", validate="many_to_one")
    return base


def add_prior_region_metrics(server_day: pd.DataFrame, region_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = region_metrics.copy()
    metrics["date"] = pd.to_datetime(metrics["date"], errors="raise")
    metrics = metrics.sort_values(["region", "date"])
    shifted = metrics[["date", "region"]].copy()
    for column in REGION_METRIC_COLUMNS:
        shifted[f"{column}_prev_day"] = metrics.groupby("region")[column].shift(1)
    return server_day.merge(shifted, on=["date", "region"], how="left", validate="many_to_one")


def add_current_region_metrics(server_day: pd.DataFrame, region_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = region_metrics.copy()
    metrics["date"] = pd.to_datetime(metrics["date"], errors="raise")
    return server_day.merge(metrics, on=["date", "region"], how="left", validate="many_to_one")


def add_incident_target(server_day: pd.DataFrame, incidents: pd.DataFrame) -> pd.DataFrame:
    incident_dates = incidents[["server_id", "date"]].drop_duplicates().copy()
    incident_dates["date"] = pd.to_datetime(incident_dates["date"], errors="raise")

    future_windows = []
    for offset in range(1, 8):
        shifted = incident_dates.copy()
        shifted["date"] = shifted["date"] - pd.Timedelta(days=offset)
        future_windows.append(shifted)
    target_keys = pd.concat(future_windows, ignore_index=True).drop_duplicates()
    target_keys[TARGET_INCIDENT] = 1

    result = server_day.merge(target_keys, on=["server_id", "date"], how="left")
    result[TARGET_INCIDENT] = result[TARGET_INCIDENT].fillna(0).astype(int)
    return result


def add_overload_anomaly_label(server_day: pd.DataFrame) -> pd.DataFrame:
    result = server_day.copy()
    utilization_pressure = (
        result["cpu_util_pct"] + result["ram_util_pct"] + result["disk_util_pct"]
    ) / 3
    result[TARGET_ANOMALY] = (
        (result["cpu_util_pct"] >= 90)
        | (result["ram_util_pct"] >= 92)
        | ((utilization_pressure >= 80) & (result["temperature_c"] >= 70))
    ).astype(int)
    return result


def build_incident_dataset(
    usage: pd.DataFrame,
    servers: pd.DataFrame,
    customers: pd.DataFrame,
    region_metrics: pd.DataFrame,
    incidents: pd.DataFrame,
) -> pd.DataFrame:
    incident_df = build_server_day_base(usage, servers, customers)
    incident_df = add_prior_region_metrics(incident_df, region_metrics)
    incident_df = add_incident_target(incident_df, incidents)
    incident_df = add_date_features(incident_df)
    incident_df = incident_df.sort_values(["server_id", "date"]).reset_index(drop=True)
    return incident_df.drop(columns=["customer_id"])


def build_support_dataset(region_metrics: pd.DataFrame) -> pd.DataFrame:
    support_df = add_date_features(region_metrics)
    support_df = support_df.sort_values(["region", "date"]).reset_index(drop=True)
    support_df[TARGET_SUPPORT_NEXT_DAY] = support_df.groupby("region")[TARGET_SUPPORT].shift(-1)
    support_df = support_df.dropna(subset=[TARGET_SUPPORT_NEXT_DAY]).copy()
    support_df[TARGET_SUPPORT_NEXT_DAY] = support_df[TARGET_SUPPORT_NEXT_DAY].astype(int)
    return support_df


def build_unsupervised_dataset(
    usage: pd.DataFrame,
    servers: pd.DataFrame,
    customers: pd.DataFrame,
    region_metrics: pd.DataFrame,
) -> pd.DataFrame:
    unsupervised_df = build_server_day_base(usage, servers, customers)
    unsupervised_df = add_current_region_metrics(unsupervised_df, region_metrics)
    unsupervised_df = add_date_features(unsupervised_df)
    unsupervised_df = unsupervised_df.sort_values(["server_id", "date"]).reset_index(drop=True)
    return unsupervised_df.drop(columns=["customer_id"])


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ml_df = pd.read_csv(ML_DATASET_PATH)
    region_metrics_df = pd.read_csv(REGION_METRICS_PATH)
    customers_df = pd.read_csv(CUSTOMERS_PATH)
    servers_df = pd.read_csv(SERVERS_PATH)
    usage_df = pd.read_csv(DAILY_USAGE_PATH)
    incidents_df = pd.read_csv(INCIDENTS_PATH)

    raw_ml_report = quality_report(ml_df)

    incident_df = build_incident_dataset(usage_df, servers_df, customers_df, region_metrics_df, incidents_df)
    support_df = build_support_dataset(region_metrics_df)
    unsupervised_df = build_unsupervised_dataset(usage_df, servers_df, customers_df, region_metrics_df)

    incident_path = PROCESSED_DIR / "incident_dataset.csv"
    support_path = PROCESSED_DIR / "support_dataset.csv"
    unsupervised_path = PROCESSED_DIR / "unsupervised_dataset.csv"
    report_path = PROCESSED_DIR / "preprocessing_report.json"

    incident_df.to_csv(incident_path, index=False)
    support_df.to_csv(support_path, index=False)
    unsupervised_df.to_csv(unsupervised_path, index=False)

    report = {
        "source_files": {
            "ml_training_dataset": str(ML_DATASET_PATH.relative_to(ROOT_DIR)),
            "technical_region_metrics": str(REGION_METRICS_PATH.relative_to(ROOT_DIR)),
            "daily_server_usage": str(DAILY_USAGE_PATH.relative_to(ROOT_DIR)),
            "servers": str(SERVERS_PATH.relative_to(ROOT_DIR)),
            "customers": str(CUSTOMERS_PATH.relative_to(ROOT_DIR)),
            "incidents": str(INCIDENTS_PATH.relative_to(ROOT_DIR)),
        },
        "raw_ml_dataset": raw_ml_report,
        "processed_outputs": {
            "incident_dataset": {
                "path": str(incident_path.relative_to(ROOT_DIR)),
                "rows": int(len(incident_df)),
                "columns": int(len(incident_df.columns)),
                "target": TARGET_INCIDENT,
            },
            "support_dataset": {
                "path": str(support_path.relative_to(ROOT_DIR)),
                "rows": int(len(support_df)),
                "columns": int(len(support_df.columns)),
                "target": TARGET_SUPPORT_NEXT_DAY,
            },
            "unsupervised_dataset": {
                "path": str(unsupervised_path.relative_to(ROOT_DIR)),
                "rows": int(len(unsupervised_df)),
                "columns": int(len(unsupervised_df.columns)),
                "target": None,
            },
        },
        "preprocessing_choices": {
            "date_features": ["day_of_week", "day_of_month", "days_since_start"],
            "incident_target": "Rebuilt from incidents.csv as server incident in days +1 through +7.",
            "incident_region_metrics": "Region metrics are shifted by one day and suffixed with _prev_day to avoid same-day leakage.",
            "support_target": "support_tickets_next_1d predicts next-day region tickets.",
            "removed_identifier_columns": ["customer_id"],
            "kept_traceability_columns": ["date", "server_id"],
            "missing_value_strategy": "No missing values detected in raw files; no imputation applied.",
        },
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Created {incident_path.relative_to(ROOT_DIR)}: {incident_df.shape}")
    print(f"Created {support_path.relative_to(ROOT_DIR)}: {support_df.shape}")
    print(f"Created {unsupervised_path.relative_to(ROOT_DIR)}: {unsupervised_df.shape}")
    print(f"Created {report_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
