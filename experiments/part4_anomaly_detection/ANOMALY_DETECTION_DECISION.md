# Choix du modèle - Détection d'anomalies

Nous comparons plusieurs modèles non supervisés pour détecter `overload_anomaly`.

## Pourquoi cette approche

- La Partie 4 demande une détection automatique de comportements anormaux.
- Les modèles sont entraînés sans utiliser la cible `overload_anomaly` comme feature.
- La cible sert uniquement à évaluer les scores produits.
- Le seuil de décision est fixé avec le taux d'anomalie observé sur le train temporel.

## Modèles testés

- `IsolationForest`
- `LocalOutlierFactor`
- `SGDOneClassSVM`

## Résultats

Run retenu : `experiments/part4_anomaly_detection/runs/20260703_143907/`

- meilleur modèle : `LocalOutlierFactor`
- métrique de sélection : `average_precision`
- `average_precision` : `0.0719`
- `roc_auc` : `0.8497`
- précision : `0.0313`
- rappel : `0.1852`
- `F1` : `0.0536`
- taux d'anomalies prédites : `0.0201`

Comparaison :

- `LocalOutlierFactor` : meilleur `average_precision` et meilleur rappel.
- `IsolationForest` : meilleure précision et `F1`, mais rappel plus faible.
- `SGDOneClassSVM` : mauvais classement sur ce dataset.

Nous retenons `LocalOutlierFactor` car la Partie 4 vise à détecter des comportements anormaux pouvant précéder une panne. Dans ce contexte, manquer trop d'anomalies est plus risqué que générer quelques alertes supplémentaires.

Fichiers à analyser :

- `model_metrics.csv`
- `run_summary.json`
- `figures/anomaly_score_distribution.png`

## Limite

La détection d'anomalies est sensible au choix du seuil. Ici, le seuil utilise un taux d'anomalie estimé sur le train, ce qui rend le résultat exploitable mais dépendant de la qualité de l'étiquette historique.

Le taux d'anomalie est très faible (`0.28%` sur tout le dataset), donc les métriques de précision restent naturellement basses. Le modèle doit être utilisé comme outil de priorisation ou d'alerte, pas comme décision automatique isolée.
