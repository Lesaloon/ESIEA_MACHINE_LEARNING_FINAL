# Choix du modèle - Priorisation des interventions préventives

La Partie 6 est traitée comme un problème de ranking opérationnel.

## Score utilisé

```text
priority_score = incident_probability * business_value
```

- `incident_probability` vient d'un modèle de classification.
- `business_value` approxime la valeur métier d'une intervention évitant un incident.

## Modèles testés

- `RandomForestClassifier`
- `ExtraTreesClassifier`
- `HistGradientBoostingClassifier`

## Résultats

Run retenu : `experiments/part6_intervention_prioritization/runs/20260703_151435/`

- meilleur modèle : `HistGradientBoostingClassifier`
- métrique de sélection : `value_capture_rate` sur le top 50 par jour
- `value_capture_rate` : `0.5292`
- incidents capturés : `47 / 190`
- `capture_rate` : `0.2474`
- `precision_at_k` : `0.0588`
- `average_precision` : `0.0753`
- `roc_auc` : `0.7735`

Comparaison :

- `HistGradientBoostingClassifier` : meilleure capture de valeur et meilleur taux de capture d'incidents.
- `RandomForestClassifier` : proche en `average_precision`, mais moins bon sur le top 50 métier.
- `ExtraTreesClassifier` : moins bon en valeur capturée.

Nous retenons donc `HistGradientBoostingClassifier`, car la Partie 6 doit optimiser la valeur opérationnelle des 50 interventions possibles, pas seulement la métrique probabiliste globale.

## Limite

La valeur métier est une approximation. En production, elle devrait être ajustée avec les coûts réels d'incident, les SLA, la criticité client et la disponibilité des équipes.
