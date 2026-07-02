# Incident Prediction Training

Final training script for the retained `RandomForestClassifier` model.

Run:

```bash
.venv/bin/python train/incident_prediction/train_final_random_forest.py
```

Outputs are written to:

```bash
train/incident_prediction/runs/<timestamp>/
```

Main artifact:

```bash
final_random_forest_model.pkl
```

The `.pkl` file is saved with `joblib.dump`.
