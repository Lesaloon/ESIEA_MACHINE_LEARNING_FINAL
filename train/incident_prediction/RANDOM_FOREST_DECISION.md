# Choix du modèle - Prédiction des incidents

Nous retenons un `RandomForestClassifier` pour prédire `incident_next_7d`.

## Pourquoi RandomForest

- Il fonctionne bien sur des données tabulaires avec variables numériques et catégorielles encodées.
- Il gère correctement les relations non linéaires entre charge CPU, RAM, température, région, support, etc.
- Il est robuste au bruit et aux outliers.
- Il donne de meilleures performances que les autres modèles testés sur notre métrique principale.

## Résultats retenus

Meilleur run v2 :

- modèle : `RandomForestClassifier`
- sélection de variables : `SelectKBest(k=70)`
- `Average Precision` : `0.1456`
- `ROC-AUC` : `0.8217`
- `F1` au meilleur seuil : `0.2589`
- `Precision` au meilleur seuil : `0.2500`
- `Recall` au meilleur seuil : `0.2684`

## Intérêt métier

Le modèle est surtout utile pour classer les serveurs par niveau de risque.

Sur le jeu de test, en prenant les 50 serveurs les plus risqués par jour :

- incidents capturés : `104 / 190`
- taux de capture : `54.7%`
- précision top 50 : `13.0%`

## Limite

La cible est rare, environ `2%` d'incidents. Le modèle ne doit donc pas être évalué avec l'accuracy seule. Les métriques importantes sont `Average Precision`, `Recall`, `Precision` et les métriques top-K.
