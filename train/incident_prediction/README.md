# Incident Prediction Training

This directory contains the final training script for part 1: predicting `incident_next_7d` from `data/raw/ml_training_dataset.csv`.

## Final Model

- problem: binary classification
- target: `incident_next_7d`
- current retained model: `RandomForestClassifier`
- training selection metric: `average_precision`
- serving decision rule: predict `1` when `probability >= threshold`

Current saved threshold from `models/metadata/incident_random_forest_model.json`:

```json
0.05344621745357038
```

This threshold is intentionally much lower than `0.5` because incidents are rare in the dataset.

## Why We Do Not Optimize Accuracy

The positive class rate is only about `2.2%`.

That means a model can get very high accuracy by predicting almost everything as `0`.
That is exactly what happened with the accuracy-based version: it looked strong on paper, but it missed almost all real incidents.

For that reason, the final training now uses:

- `average_precision` to choose the model during search
- a precision-recall threshold search to choose the serving cutoff

This keeps much more recall on the minority class.

## Current Metrics

For the retained random forest model:

- average precision: `0.1945`
- ROC-AUC: `0.9464`
- threshold: `0.0534`
- precision at threshold: `0.2026`
- recall at threshold: `0.8252`
- F1 at threshold: `0.3253`

Important: the probability is not the prediction. The prediction is produced after applying the saved threshold.

## Why A 20% Probability Can Produce Prediction = 1

Because the service does not use `0.5` as the decision boundary.

It uses the saved threshold:

```text
0.05344621745357038
```

So:

- probability `0.20` means `20%`
- `0.20 > 0.0534`
- therefore prediction = `1`

This is expected behavior, not a bug.

The reason is business and class imbalance driven: with a `0.5` threshold, the model almost never predicts incidents. With a lower threshold, it catches far more real incidents.

## Why Probabilities Rarely Go Above 25%

This is also expected for this dataset.

Main reasons:

- incidents are rare, so the model's prior belief stays low
- random forests often produce conservative probabilities when the positive class is sparse
- many rows may be risky relative to the baseline without looking like near-certain incidents
- the model was selected to rank risky cases well, not to emit very large probabilities

In this setup, the probability should be read as a ranking score with probabilistic meaning, not as a requirement that alerts must be above `50%`.

For this model, a probability around `0.15` to `0.25` can already be very high relative to the dataset base rate of about `0.022`.

Example:

- base rate: about `2.2%`
- score: `20%`

That is roughly 9 times the baseline incident frequency, so it is a strong signal even if it does not look numerically large.

## Explainability

The inference service makes the model explainable in two layers.

### 1. Local feature attributions

In `services/shared-inference/inference/predictor.py`, the service computes top local drivers for each prediction.

It tries SHAP first:

- transform the request with the same saved preprocessing pipeline
- explain the trained tree model on the transformed row
- return the top features with signed impact

If SHAP is unavailable, it falls back to permutation-style local explanations:

- perturb one numeric feature at a time
- recompute the score
- measure how much the prediction changes

This is what populates:

- `top_explanations`
- `human_explanation`

### 2. Human-readable explanation text

The service converts the strongest feature drivers into a short sentence such as:

```text
Le modele estime un risque incident de X %. Les facteurs qui expliquent le plus ce score sont: ...
```

That makes the output usable in the API and UI without requiring the user to read raw feature vectors.

## Inference Contract

The deployed incident model is saved as a sklearn `Pipeline`.

That is important because it keeps preprocessing and model together:

- one-hot encoding for categorical fields
- scaling for selected numeric fields
- the trained random forest itself

The API sends raw incident fields, and the service builds exactly the feature frame expected by the saved pipeline using the metadata file.

This avoids training/serving skew.

## Run Training

```bash
.venv/bin/python train/incident_prediction/train_final_random_forest.py
```

Outputs are written to:

```bash
train/incident_prediction/runs/<timestamp>/
```

The persisted serving files are:

```bash
models/artifacts/incident_random_forest_model.pkl
models/metadata/incident_random_forest_model.json
```
