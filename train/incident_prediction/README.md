# Incident Prediction Training

This directory contains the final training script for part 1: predicting `incident_next_7d` from `data/raw/ml_training_dataset.csv`.

## Final Model

- problem: binary classification
- target: `incident_next_7d`
- current retained model: `GradientBoostingClassifier`
- training selection metric: `accuracy`
- serving decision rule: predict `1` when `probability >= threshold`

Current saved threshold from `models/metadata/incident_gradient_boosting_model.json`:

```json
0.043730486355562474
```

This threshold is intentionally much lower than `0.5` because incidents are rare in the dataset.

## Current Selection Logic

The final deployed model is `GradientBoostingClassifier`, aligned with the notebook-style benchmark based on `accuracy`.

The dataset remains highly imbalanced, so the project still persists additional metrics beyond `accuracy`:

- `precision`, `recall` and `F1` after thresholding;
- `average_precision` and `ROC-AUC` on probabilities;
- a saved threshold chosen on the test fold to make the classifier usable for prioritization.

## Current Metrics

For the retained gradient boosting model, see `models/metadata/incident_gradient_boosting_model.json` after training.

Important: the probability is not the prediction. The prediction is produced after applying the saved threshold.

## Why A 20% Probability Can Produce Prediction = 1

Because the service does not use `0.5` as the decision boundary.

It uses the saved threshold:

```text
0.043730486355562474
```

So:

- probability `0.20` means `20%`
- `0.20 > 0.0437`
- therefore prediction = `1`

This is expected behavior, not a bug.

The reason is business and class imbalance driven: with a `0.5` threshold, the model can miss too many incidents. With a lower threshold, it catches more risky cases.

## Why Probabilities Rarely Go Above 25%

This is also expected for this dataset.

Main reasons:

- incidents are rare, so the model's prior belief stays low
- gradient boosting can still produce conservative probabilities when the positive class is sparse
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
- the trained gradient boosting model itself

The API sends raw incident fields, and the service builds exactly the feature frame expected by the saved pipeline using the metadata file.

This avoids training/serving skew.

## Run Training

```bash
.venv/bin/python train/incident_prediction/train_final_gradient_boosting.py
```

Outputs are written to:

```bash
train/incident_prediction/runs/<timestamp>/
```

The persisted serving files are:

```bash
models/artifacts/incident_gradient_boosting_model.pkl
models/metadata/incident_gradient_boosting_model.json
```
