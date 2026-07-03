# Choix du modèle - Segmentation des serveurs

Nous retenons `GaussianMixture` avec covariance diagonale pour segmenter les serveurs.

## Pourquoi ce modèle

- La Partie 3 est un problème non supervisé : il n'y a pas de variable cible.
- Plusieurs modèles sont comparés pour ne pas supposer que `KMeans` est toujours le meilleur choix.
- Les variables numériques sont standardisées et les variables catégorielles sont encodées avant clustering.
- Le modèle final et le nombre de clusters sont choisis avec le score de silhouette.

Modèles testés : `KMeans`, `GaussianMixture`, `Birch`, `AgglomerativeClustering`.

## Résultats

Run retenu : `experiments/part3_server_segmentation/runs/20260703_141532/`

- meilleur modèle : `GaussianMixture`, covariance `diag`
- nombre de clusters : `2`
- score de silhouette : `0.1225`
- Davies-Bouldin : `2.5606`
- Calinski-Harabasz : `179.4024`
- serveurs segmentés : `2200`

Top 5 du benchmark :

- `GaussianMixture(diag)`, `k=2`, silhouette `0.1225`
- `GaussianMixture(diag)`, `k=3`, silhouette `0.1006`
- `KMeans`, `k=2`, silhouette `0.0981`
- `GaussianMixture(tied)`, `k=2`, silhouette `0.0939`
- `Birch`, `k=2`, silhouette `0.0869`

Profils obtenus après correction du nommage automatique :

- cluster `0` : `1843` serveurs, profil `serveurs standard`, driver `aucun ecart dominant`
- cluster `1` : `357` serveurs, profil `serveurs temperature elevee`, driver `temperature_c_mean (+2.36 vs moyenne clusters)`

Le profil `latence elevee` n'est pas retenu car les différences de latence entre clusters sont trop faibles pour être interprétées métier. Le profil `stockage sollicite` apparaissait avec `k=3`, mais le score de silhouette indique que la séparation la plus robuste est plutôt une segmentation en deux familles : serveurs standards et serveurs à température élevée.

Fichiers à utiliser pour l'analyse métier :

- `model_metrics.csv`
- `cluster_profiles.csv`
- `server_cluster_assignments.csv`
- `figures/server_segments_pca.png`

## Limite

La segmentation dépend des variables disponibles et de l'agrégation par serveur. Elle sert à identifier des profils globaux, pas à détecter directement une panne ponctuelle.
