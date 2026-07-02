from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

ML_DATASET_PATH = RAW_DIR / "ml_training_dataset.csv"
REGION_METRICS_PATH = RAW_DIR / "technical_region_metrics.csv"

TARGET_INCIDENT = "incident_next_7d"
TARGET_SUPPORT = "support_tickets"
TARGET_ANOMALY = "overload_anomaly"


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


def build_incident_dataset(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ["customer_id", TARGET_ANOMALY]
    return df.drop(columns=columns_to_drop)


def build_support_dataset(region_metrics: pd.DataFrame) -> pd.DataFrame:
    support_df = add_date_features(region_metrics)
    support_df = support_df.sort_values(["region", "date"]).reset_index(drop=True)
    return support_df


def build_unsupervised_dataset(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ["customer_id", TARGET_INCIDENT, TARGET_SUPPORT, TARGET_ANOMALY]
    return df.drop(columns=columns_to_drop)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ml_df = pd.read_csv(ML_DATASET_PATH)
    region_metrics_df = pd.read_csv(REGION_METRICS_PATH)

    raw_ml_report = quality_report(ml_df)

    ml_df = add_date_features(ml_df)
    ml_df = ml_df.sort_values(["server_id", "date"]).reset_index(drop=True)

    incident_df = build_incident_dataset(ml_df)
    support_df = build_support_dataset(region_metrics_df)
    unsupervised_df = build_unsupervised_dataset(ml_df)

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
                "target": TARGET_SUPPORT,
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
