# Choix du modèle - Prévision des tickets support

Nous retenons `ExtraTreesRegressor` pour prédire `support_tickets`.

## Pourquoi ce modèle

- Il fonctionne bien sur les données tabulaires.
- Il gère les relations non linéaires entre région, latence, capacité et historique récent.
- Il obtient le meilleur `MAE` parmi les modèles classiques testés sans `PoissonRegressor`.

## Résultats

Run retenu : `experiments/part2_support_forecast/runs/20260702_154507/`

- meilleur modèle : `ExtraTreesRegressor`
- meilleurs paramètres : `max_depth=5`, `max_features=0.6`, `min_samples_leaf=7`, `min_samples_split=12`, `n_estimators=207`
- `MAE` : `1.6636`
- `RMSE` : `2.0496`
- `R2` : `0.0233`

Baseline moyenne :

- `MAE` : `1.7123`

## Limite

Le gain reste faible. Le dataset contient seulement `525` lignes, avec `75` jours et `7` régions. Pour améliorer fortement ce modèle, il faudrait plus d'historique et davantage de signaux métier sur la charge support.
