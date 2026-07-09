# Choix du modèle - Prédiction des incidents

Nous retenons un `GradientBoostingClassifier` pour prédire `incident_next_7d`.

## Pourquoi GradientBoosting

- Il fonctionne bien sur des données tabulaires avec variables numériques et catégorielles encodées.
- Il gère correctement les relations non linéaires entre charge CPU, RAM, température, région, support, etc.
- Il donne une tres bonne accuracy sur le split de validation utilise dans le projet.
- Il reste compatible avec une explication locale de type SHAP.

## Résultats retenus

Modele retenu pour le deploiement:

- modele : `GradientBoostingClassifier`
- metrique de selection principale : `accuracy`
- metriques complementaires : `precision`, `recall`, `F1`, `average_precision`, `ROC-AUC`

## Intérêt métier

Le modele est surtout utile pour classer les serveurs par niveau de risque, pas seulement pour une prediction binaire brute.

Les chiffres deployes exacts sont ceux ecrits dans `models/metadata/incident_gradient_boosting_model.json` apres entrainement.

## Limite

La cible est rare, environ `2%` d'incidents. Meme si l'accuracy sert a choisir le modele final pour rester aligne avec le notebook, il faut encore lire `Recall`, `Precision`, `F1` et `Average Precision` avant d'utiliser le modele en production.
