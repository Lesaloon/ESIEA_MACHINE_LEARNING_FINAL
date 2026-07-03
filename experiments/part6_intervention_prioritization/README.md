# Partie 6 - Priorisation intelligente des interventions

Objectif : chaque matin, sélectionner les `50` serveurs où une intervention préventive apporte le plus de valeur.

Le score de priorité combine :

```text
priority_score = incident_probability * business_value
```

La valeur métier est une approximation basée sur le revenu mensuel, le support plan, le type de serveur et la pression de capacité.

## Lancer l'expérience

```bash
.venv/bin/python experiments/part6_intervention_prioritization/run_prioritization_experiments.py --top-k 50
```

## Sorties

- `model_metrics.csv` : comparaison des modèles.
- `predictions_test_<model>.csv` : probabilités d'incident sur le test temporel.
- `best_prioritization_<model>.pkl` : meilleur pipeline sauvegardé avec `joblib.dump`.
- `run_summary.json` : résumé du meilleur modèle.
