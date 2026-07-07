# Choix du modèle - Segmentation des serveurs

Depuis le rebuild du `2026-07-07`, nous retenons `KMeans` comme modele de segmentation deploye.

## Pourquoi ce modèle

- La Partie 3 est un problème non supervisé : il n'y a pas de variable cible.
- La segmentation est maintenant reconstruite depuis les donnees brutes `servers.csv`, `daily_server_usage.csv` et `incidents.csv`.
- Plusieurs modèles sont comparés via `RandomizedSearchCV` pour ne pas figer l'algorithme a l'avance.
- Les variables numeriques sont imputees puis standardisees, les variables categorielles sont encodees, et une reduction `PCA` peut etre selectionnee pendant la recherche.
- Le modèle final est choisi sur le `test_silhouette` apres tuning.

Modeles testes : `KMeans`, `GaussianMixture`.

## Résultats

Run retenu pour l'artefact deploye : `train/server_segmentation/runs/20260707_224440/`

- meilleur modele : `KMeans`
- meilleurs hyperparametres : `n_clusters=5`, `init=random`, `n_init=39`, `max_iter=457`, `PCA(n_components=0.9)`
- meilleur CV silhouette : `0.1644`
- silhouette holdout : `0.1783`
- Davies-Bouldin holdout : `1.6404`
- Calinski-Harabasz holdout : `73.8594`
- silhouette full fit : `0.1625`
- serveurs segmentés : `2200`

Benchmark retenu :

- `KMeans`, silhouette test `0.1783`
- `GaussianMixture`, silhouette test `0.1222`

Profils obtenus :

- cluster `0` : `229` serveurs, profil `serveurs stockage sollicite`, driver `disk_util_pct_mean (+13.71 vs moyenne clusters)`
- cluster `1` : `46` serveurs, profil `serveurs standard`, driver `aucun ecart dominant`
- cluster `2` : `766` serveurs, profil `serveurs sous stress thermique`, driver `temperature_c_mean (+3.65 vs moyenne clusters)`
- cluster `3` : `175` serveurs, profil `serveurs standard`, driver `aucun ecart dominant`
- cluster `4` : `984` serveurs, profil `serveurs standard`, driver `aucun ecart dominant`

Le rebuild introduit aussi des variables d'incidents agregees (`incident_count`, `incident_rate`, `recent30_incident_rate`, severite, duree, recence). Elles n'ont pas force un profil nomme dedie sur ce run, mais elles contribuent a la separation entre groupes a faible historique d'incidents et groupes plus exposes.

Fichiers à utiliser pour l'analyse métier :

- `benchmark_metrics.csv`
- `server_feature_table.csv`
- `cluster_profiles.csv`
- `server_cluster_assignments.csv`
- `figures/server_segments_pca.png`

## Limite

La segmentation depend des variables disponibles et de l'agregation par serveur. Elle sert a identifier des profils globaux, pas a detecter directement une panne ponctuelle. Le schema d'inference segmentation a aussi ete simplifie le `2026-07-07` pour retirer les variables client et se caler sur les seules donnees serveur / usage / incidents.
