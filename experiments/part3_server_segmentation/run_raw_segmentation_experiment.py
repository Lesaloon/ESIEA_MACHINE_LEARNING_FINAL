from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train.server_segmentation.train_final_gaussian_mixture import (
    build_cluster_outputs,
    build_server_feature_table,
    clustering_metrics,
    load_raw_data,
    run_searches,
    save_pca_plot,
    setup_logger,
    split_dataset,
)


EXPERIMENT_DIR = ROOT_DIR / "experiments" / "part3_server_segmentation"
DEFAULT_SERVERS_PATH = ROOT_DIR / "data" / "raw" / "servers.csv"
DEFAULT_USAGE_PATH = ROOT_DIR / "data" / "raw" / "daily_server_usage.csv"
DEFAULT_INCIDENTS_PATH = ROOT_DIR / "data" / "raw" / "incidents.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw-data server segmentation experiment.")
    parser.add_argument("--servers-path", type=Path, default=DEFAULT_SERVERS_PATH)
    parser.add_argument("--usage-path", type=Path, default=DEFAULT_USAGE_PATH)
    parser.add_argument("--incidents-path", type=Path, default=DEFAULT_INCIDENTS_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = EXPERIMENT_DIR / "raw_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(run_dir)
    logger.info("Raw server segmentation experiment started")
    logger.info("Arguments: %s", vars(args))

    servers, usage, incidents = load_raw_data(args.servers_path, args.usage_path, args.incidents_path, logger)
    feature_df = build_server_feature_table(servers, usage, incidents, logger)
    feature_df.to_csv(run_dir / "server_feature_table.csv", index=False)

    train_df, test_df = split_dataset(feature_df, args.test_size, logger)
    results, best_estimator, best_model_name, best_params = run_searches(
        train_df.drop(columns=["server_id"]),
        test_df.drop(columns=["server_id"]),
        args,
        run_dir,
        logger,
    )

    metrics_df = pd.DataFrame(results).sort_values("test_silhouette", ascending=False)
    metrics_df.to_csv(run_dir / "benchmark_metrics.csv", index=False)

    full_pipeline = best_estimator
    full_pipeline.fit(feature_df.drop(columns=["server_id"]), [0] * len(feature_df))
    assignments, profiles = build_cluster_outputs(full_pipeline, feature_df)
    assignments.to_csv(run_dir / "server_cluster_assignments.csv", index=False)
    profiles.to_csv(run_dir / "cluster_profiles.csv", index=False)
    save_pca_plot(feature_df, full_pipeline, run_dir)

    model_path = run_dir / f"server_segmentation_{best_model_name}.pkl"
    joblib.dump(full_pipeline, model_path)

    summary = {
        "problem_type": "unsupervised_server_segmentation",
        "data_sources": {
            "servers": str(args.servers_path.relative_to(ROOT_DIR)),
            "daily_server_usage": str(args.usage_path.relative_to(ROOT_DIR)),
            "incidents": str(args.incidents_path.relative_to(ROOT_DIR)),
        },
        "selection_metric": "test_silhouette",
        "best_model": best_model_name,
        "best_params": best_params,
        "feature_rows": int(len(feature_df)),
        "feature_columns": int(feature_df.shape[1] - 1),
        "full_metrics": clustering_metrics(full_pipeline, feature_df.drop(columns=["server_id"])),
        "model_path": str(model_path.relative_to(ROOT_DIR)),
        "profiles_path": str((run_dir / "cluster_profiles.csv").relative_to(ROOT_DIR)),
        "assignments_path": str((run_dir / "server_cluster_assignments.csv").relative_to(ROOT_DIR)),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Best model: %s", best_model_name)
    logger.info("Saved model to %s", model_path.relative_to(ROOT_DIR))
    logger.info("Raw server segmentation experiment finished")


if __name__ == "__main__":
    main()
