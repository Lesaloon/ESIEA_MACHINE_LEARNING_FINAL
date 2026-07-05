# Choix de la règle métier - Priorisation des interventions préventives

La Partie 6 est traitée comme un problème de ranking opérationnel avec des règles métier réutilisant les modèles déjà existants.

## Score utilisé

```text
priority_score = incident_probability * business_value
```

- `incident_probability` vient d'un modèle de classification.
- `business_value` approxime la valeur métier d'une intervention évitant un incident.
- Les modèles d'anomalie et de segmentation restent affichés comme contexte, mais ne multiplient pas le score final car le benchmark montre que cela dégrade la capture de valeur.

## Règles testées

- `incident_only`
- `incident_value`
- `incident_value_anomaly`
- `incident_value_anomaly_temperature`

## Résultats

Run retenu : `experiments/part6_intervention_prioritization/runs/20260705_165305/`

- meilleure règle : `incident_value`
- métrique de sélection : `value_capture_rate` sur le top 50 par jour
- formule : `priority_score = incident_probability * business_value`
- `value_capture_rate` : `0.5902`
- incidents capturés : `57 / 190`
- `capture_rate` : `0.3000`
- `precision_at_k` : `0.0713`

Comparaison :

- `incident_value` : meilleure capture de valeur.
- `incident_value_anomaly` : résultat identique sur ce run.
- `incident_value_anomaly_temperature` : légèrement moins bon.
- `incident_only` : capture plus d'incidents bruts, mais beaucoup moins de valeur métier.

Nous retenons donc une règle métier simple basée sur le modèle incident déjà existant. Elle est plus explicable et plus performante en valeur capturée que le modèle dédié entraîné précédemment.

## Limite

La valeur métier est une approximation. En production, elle devrait être ajustée avec les coûts réels d'incident, les SLA, la criticité client et la disponibilité des équipes.
