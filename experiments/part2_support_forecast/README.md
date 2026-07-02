# Part 2 - Support Tickets Forecast

Goal: predict `support_tickets` for each region and day.

Run:

```bash
.venv/bin/python experiments/part2_support_forecast/run_support_experiments.py
```

Fast smoke test:

```bash
.venv/bin/python experiments/part2_support_forecast/run_support_experiments.py --n-iter 1
```

Optional run with server region/day aggregates from `incident_dataset.csv`:

```bash
.venv/bin/python experiments/part2_support_forecast/run_support_experiments.py --use-server-aggregates
```

Outputs are written to:

```bash
experiments/part2_support_forecast/runs/<timestamp>/
```

The best model artifact is saved as:

```bash
best_support_model.pkl
```
