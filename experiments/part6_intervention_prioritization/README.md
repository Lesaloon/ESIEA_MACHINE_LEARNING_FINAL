# Partie 6 - Priorisation intelligente des interventions

Objectif : chaque matin, sélectionner les `50` serveurs où une intervention préventive apporte le plus de valeur.

Cette partie utilise des règles métier avec les modèles déjà existants. Aucun modèle supplémentaire n'est entraîné pour la priorisation.

Le score de priorité combine :

```text
priority_score = incident_probability * business_value
```

La valeur métier est une approximation basée sur le revenu mensuel, le support plan, le type de serveur et la pression de capacité. Les signaux anomalie/segmentation peuvent être affichés comme contexte, mais la règle retenue ne les multiplie pas dans le score final.

## Lancer l'expérience

```bash
.venv/bin/python experiments/part6_intervention_prioritization/run_prioritization_experiments.py --top-k 50
```

## Sorties

- `rule_metrics.csv` : comparaison des règles métier.
- `prioritization_scores.csv` : scores calculés sur le test temporel.
- `run_summary.json` : résumé du meilleur modèle.
