from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

INCIDENT_DATASET_PATH = PROCESSED_DIR / "incident_dataset.csv"
SUPPORT_DATASET_PATH = PROCESSED_DIR / "support_dataset.csv"
UNSUPERVISED_DATASET_PATH = PROCESSED_DIR / "unsupervised_dataset.csv"

RANDOM_STATE = 42
SCATTER_SAMPLE_SIZE = 5000


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "string"]).columns.tolist()


def plot_numeric_histograms(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    for column in numeric_columns(df):
        plt.figure(figsize=(9, 5))
        sns.histplot(df[column], kde=True, bins=40)
        plt.title(f"Distribution de {column} - {dataset_name}")
        plt.xlabel(column)
        plt.ylabel("Nombre d'observations")
        save_current_figure(output_dir / dataset_name / "histograms" / f"{column}.png")


def plot_categorical_counts(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    ignored = {"date", "server_id"}
    for column in categorical_columns(df):
        if column in ignored or df[column].nunique() > 30:
            continue

        plt.figure(figsize=(10, 5))
        order = df[column].value_counts().index
        sns.countplot(data=df, x=column, order=order)
        plt.title(f"Répartition de {column} - {dataset_name}")
        plt.xlabel(column)
        plt.ylabel("Nombre d'observations")
        plt.xticks(rotation=35, ha="right")
        save_current_figure(output_dir / dataset_name / "categorical_counts" / f"{column}.png")


def plot_correlation_heatmap(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    columns = numeric_columns(df)
    if len(columns) < 2:
        return

    corr = df[columns].corr()
    plt.figure(figsize=(18, 14))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=False, linewidths=0.2)
    plt.title(f"Matrice de corrélation - {dataset_name}")
    save_current_figure(output_dir / dataset_name / "correlation_heatmap.png")


def plot_target_distribution(df: pd.DataFrame, target: str, dataset_name: str, output_dir: Path) -> None:
    if target not in df.columns:
        return

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=target)
    plt.title(f"Distribution de la cible {target} - {dataset_name}")
    plt.xlabel(target)
    plt.ylabel("Nombre d'observations")
    save_current_figure(output_dir / dataset_name / f"target_{target}_distribution.png")


def plot_boxplots_by_target(
    df: pd.DataFrame,
    target: str,
    dataset_name: str,
    output_dir: Path,
    columns: list[str],
) -> None:
    if target not in df.columns:
        return

    for column in columns:
        if column not in df.columns:
            continue

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=target, y=column)
        plt.title(f"{column} selon {target} - {dataset_name}")
        plt.xlabel(target)
        plt.ylabel(column)
        save_current_figure(output_dir / dataset_name / "boxplots_by_target" / f"{column}_by_{target}.png")


def plot_scatter_pairs(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    pairs: list[tuple[str, str]],
    hue: str | None = None,
) -> None:
    sample = df.sample(min(len(df), SCATTER_SAMPLE_SIZE), random_state=RANDOM_STATE)

    for x_column, y_column in pairs:
        if x_column not in sample.columns or y_column not in sample.columns:
            continue

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=sample,
            x=x_column,
            y=y_column,
            hue=hue if hue in sample.columns else None,
            alpha=0.45,
            s=18,
        )
        plt.title(f"Scatter plot {x_column} vs {y_column} - {dataset_name}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        save_current_figure(output_dir / dataset_name / "scatter_plots" / f"{x_column}_vs_{y_column}.png")


def plot_time_series(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    if "date" not in df.columns:
        return

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    metrics = [
        "cpu_util_pct",
        "ram_util_pct",
        "disk_util_pct",
        "temperature_c",
        "network_latency_ms",
        "support_tickets",
        "incident_next_7d",
        "capacity_used_pct",
    ]
    available_metrics = [column for column in metrics if column in data.columns]
    if not available_metrics:
        return

    daily = data.groupby("date", as_index=False)[available_metrics].mean()
    for column in available_metrics:
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=daily, x="date", y=column)
        plt.title(f"Évolution temporelle de {column} - {dataset_name}")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.xticks(rotation=35, ha="right")
        save_current_figure(output_dir / dataset_name / "time_series" / f"{column}_over_time.png")


def generate_incident_visualizations(output_dir: Path) -> None:
    df = pd.read_csv(INCIDENT_DATASET_PATH)
    dataset_name = "incident_dataset"

    plot_numeric_histograms(df, dataset_name, output_dir)
    plot_categorical_counts(df, dataset_name, output_dir)
    plot_correlation_heatmap(df, dataset_name, output_dir)
    plot_target_distribution(df, "incident_next_7d", dataset_name, output_dir)
    plot_boxplots_by_target(
        df,
        "incident_next_7d",
        dataset_name,
        output_dir,
        [
            "cpu_util_pct",
            "ram_util_pct",
            "disk_util_pct",
            "temperature_c",
            "network_latency_ms",
            "support_tickets",
            "capacity_used_pct",
        ],
    )
    plot_scatter_pairs(
        df,
        dataset_name,
        output_dir,
        [
            ("cpu_util_pct", "temperature_c"),
            ("ram_util_pct", "disk_util_pct"),
            ("network_latency_ms", "support_tickets"),
            ("capacity_used_pct", "power_usage_mw"),
            ("net_in_gb", "net_out_gb"),
        ],
        hue="incident_next_7d",
    )
    plot_time_series(df, dataset_name, output_dir)


def generate_support_visualizations(output_dir: Path) -> None:
    df = pd.read_csv(SUPPORT_DATASET_PATH)
    dataset_name = "support_dataset"

    plot_numeric_histograms(df, dataset_name, output_dir)
    plot_categorical_counts(df, dataset_name, output_dir)
    plot_correlation_heatmap(df, dataset_name, output_dir)
    plot_scatter_pairs(
        df,
        dataset_name,
        output_dir,
        [
            ("network_latency_ms", "support_tickets"),
            ("capacity_used_pct", "support_tickets"),
            ("avg_rack_temperature_c", "support_tickets"),
            ("power_usage_mw", "support_tickets"),
        ],
        hue="region",
    )
    plot_time_series(df, dataset_name, output_dir)


def generate_unsupervised_visualizations(output_dir: Path) -> None:
    df = pd.read_csv(UNSUPERVISED_DATASET_PATH)
    dataset_name = "unsupervised_dataset"

    plot_numeric_histograms(df, dataset_name, output_dir)
    plot_categorical_counts(df, dataset_name, output_dir)
    plot_correlation_heatmap(df, dataset_name, output_dir)
    plot_scatter_pairs(
        df,
        dataset_name,
        output_dir,
        [
            ("cpu_util_pct", "ram_util_pct"),
            ("cpu_util_pct", "temperature_c"),
            ("disk_util_pct", "capacity_used_pct"),
            ("net_in_gb", "net_out_gb"),
            ("age_days", "monthly_spend_eur"),
        ],
        hue="server_type",
    )
    plot_time_series(df, dataset_name, output_dir)


def ensure_processed_datasets_exist() -> None:
    missing_paths = [
        path
        for path in [INCIDENT_DATASET_PATH, SUPPORT_DATASET_PATH, UNSUPERVISED_DATASET_PATH]
        if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path.relative_to(ROOT_DIR)) for path in missing_paths)
        raise FileNotFoundError(
            f"Missing processed datasets: {missing}. Run `python scripts/preprocess_data.py` first."
        )


def main() -> None:
    ensure_processed_datasets_exist()
    sns.set_theme(style="whitegrid", context="notebook")

    generate_incident_visualizations(FIGURES_DIR)
    generate_support_visualizations(FIGURES_DIR)
    generate_unsupervised_visualizations(FIGURES_DIR)

    png_count = len(list(FIGURES_DIR.rglob("*.png")))
    print(f"Created {png_count} PNG visualizations in {FIGURES_DIR.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
