# Partie 4 - Détection d'anomalies

Objectif : détecter automatiquement les comportements anormaux pouvant précéder une panne.

Le script compare plusieurs modèles non supervisés. La colonne `overload_anomaly` est conservée uniquement pour l'évaluation, jamais comme variable d'entrée.

Modèles testés :

- `IsolationForest`
- `LocalOutlierFactor` en mode novelty
- `SGDOneClassSVM`

## Lancer l'expérience

```bash
.venv/bin/python experiments/part4_anomaly_detection/run_anomaly_experiments.py
```

Options utiles :

```bash
.venv/bin/python experiments/part4_anomaly_detection/run_anomaly_experiments.py --max-fit-rows 30000
```

## Sorties

Les sorties sont générées dans :

```bash
experiments/part4_anomaly_detection/runs/<timestamp>/
```

Fichiers principaux :

- `model_metrics.csv` : comparaison des modèles.
- `predictions_test_<model>.csv` : scores d'anomalie sur le test temporel.
- `best_anomaly_<model>.pkl` : meilleur pipeline sauvegardé avec `joblib.dump`.
- `figures/anomaly_score_distribution.png` : distribution des scores normal/anomalie.
