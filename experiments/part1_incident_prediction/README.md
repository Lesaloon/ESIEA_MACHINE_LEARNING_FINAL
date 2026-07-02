# Part 1 - Incident Prediction Experiments

Goal: benchmark binary classification models for `incident_next_7d`.

The script uses:

- temporal train/test split: train before `2026-03-01`, test from `2026-03-01`
- `TimeSeriesSplit` inside `RandomizedSearchCV`
- `average_precision` as main tuning metric because the positive class is rare
- final metrics on the temporal test set

## Environment

Create the venv with the requested Python 3.13 interpreter:

```bash
uv venv --python /home/lesaloon/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/bin/python3.13 .venv
uv pip install --python .venv/bin/python -r experiments/part1_incident_prediction/requirements.txt
```

## Run

Default benchmark:

```bash
.venv/bin/python experiments/part1_incident_prediction/run_incident_experiments.py
```

Fast smoke test:

```bash
.venv/bin/python experiments/part1_incident_prediction/run_incident_experiments.py --n-iter 1 --max-train-rows 3000
```

More exhaustive benchmark:

```bash
.venv/bin/python experiments/part1_incident_prediction/run_incident_experiments.py --n-iter 20 --cv-splits 4
```

Second experiment with historical features, `SelectKBest`/`PCA`, threshold metrics and top-K business metrics:

```bash
.venv/bin/python experiments/part1_incident_prediction/run_incident_experiments_v2.py --n-iter 10 --cv-splits 3
```

## Outputs

Each run writes to `experiments/part1_incident_prediction/runs/<timestamp>/`:

- `experiment.log`
- `benchmark_metrics.csv`
- `run_summary.json`
- `cv_results_<model>.csv`
- `classification_reports/<model>.json`
- `figures/confusion_matrix_<model>.png`
- `best_incident_model.joblib`

The v2 script writes to `experiments/part1_incident_prediction/runs_v2/<timestamp>/` and adds:

- `benchmark_metrics_v2.csv`
- `run_summary_v2.json`
- `topk_metrics_<model>.csv`
- `best_incident_model_v2.joblib`
