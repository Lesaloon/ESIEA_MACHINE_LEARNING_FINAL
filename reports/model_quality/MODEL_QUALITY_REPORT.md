# Qualite Des Modeles

## Incident Prediction

Modele deploye : `GradientBoostingClassifier`

Source : `models/metadata/incident_gradient_boosting_model.json`

Metriques principales :

- `accuracy` : `0.9780`
- `cv_best_accuracy` : `0.9780`
- `average_precision` : `0.1379`
- `roc_auc` : `0.8316`
- seuil retenu : `0.0437`
- `precision` au seuil retenu : `0.1767`
- `recall` au seuil retenu : `0.3957`
- `f1` au seuil retenu : `0.2443`

Lecture :

- Le chiffre d'`accuracy` est eleve et cohérent avec le notebook.
- En revanche, au seuil par defaut du classifieur, le modele ne detecte pratiquement aucun incident (`recall = 0.0`).
- Le seuil optimise corrige partiellement ce probleme et rend le modele exploitable pour la priorisation.

Verdict :

Le modele est correct pour classer des serveurs par risque, mais il n'est pas fort en detection pure. Il est acceptable pour un tableau de priorisation, pas pour une decision automatique stricte.

## Support Forecast

Modele deploye : `ExtraTreesRegressor`

Source : `models/metadata/support_extra_trees_model.json`

Metriques principales :

- `mae` : `1.6641`
- `rmse` : `2.0606`
- `r2` : `-0.0267`
- `safe_mape` : `0.6487`

Lecture :

- Le `MAE` reste moderement eleve par rapport a une cible en faible volume.
- Le `R2` negatif signifie que le modele fait moins bien qu'une baseline moyenne sur la variance expliquee.
- Le modele peut donner un ordre de grandeur, mais pas une prevision robuste.

Verdict :

Le modele n'est pas bon au sens predictif strict. Il peut servir de repere grossier, mais pas de base fiable pour du pilotage fin.

## Server Segmentation

Modele deploye : `KMeans`

Source : `models/metadata/server_segmentation_kmeans.json`

Metriques principales du modele retenu :

- `best_cv_silhouette` : `0.1644`
- `test_silhouette` : `0.1783`
- `test_davies_bouldin` : `1.6404`
- `test_calinski_harabasz` : `73.8594`
- `n_clusters` : `5`

Lecture :

- Le `KMeans` retenu est meilleur que `GaussianMixture` sur le split de comparaison.
- Le score de silhouette reste faible: les groupes existent, mais leur separation n'est pas tres nette.
- Les profils restent interpretable metierement, surtout pour `serveurs stockage sollicite` et `serveurs sous stress thermique`.

Verdict :

Le modele est moyen mais utile. Il est suffisamment bon pour construire des profils operationnels, pas pour affirmer une segmentation tres stable ou tres tranchee.

## Anomaly Detection

Modele deploye : `LocalOutlierFactor`

Source : `models/metadata/anomaly_local_outlier_factor.json`

Metriques principales :

- `average_precision` : `0.1066`
- `roc_auc` : `0.8798`
- `precision` : `0.0625`
- `recall` : `0.4259`
- `f1` : `0.1090`
- taux reel d'anomalie test : `0.0034`
- taux predit d'anomalie : `0.0232`

Lecture :

- Le modele retrouve une partie interessante des anomalies rares.
- Sa precision est faible, donc il produit beaucoup de faux positifs.
- Ce comportement est coherent avec un usage d'alerte precoce plutot que de validation definitive.

Verdict :

Le modele est correct pour de la surveillance sensible et de la priorisation d'alertes. Il n'est pas assez precis pour une automatisation sans verification humaine.

## Conclusion Par Usage

- Incident : utilisable pour prioriser, pas assez fort pour un cut-off binaire autonome.
- Support : faible, a ameliorer avant tout usage decisionnel important.
- Segmentation : utile pour profiler et contextualiser, separation moderee.
- Anomalie : utile pour surveiller large, a filtrer ensuite avec une revue humaine ou une regle metier.
