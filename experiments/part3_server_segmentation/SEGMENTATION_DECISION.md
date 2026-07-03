# Choix du modèle - Segmentation des serveurs

Nous retenons `GaussianMixture` avec covariance diagonale pour segmenter les serveurs.

## Pourquoi ce modèle

- La Partie 3 est un problème non supervisé : il n'y a pas de variable cible.
- Plusieurs modèles sont comparés pour ne pas supposer que `KMeans` est toujours le meilleur choix.
- Les variables numériques sont standardisées et les variables catégorielles sont encodées avant clustering.
- Le modèle final et le nombre de clusters sont choisis avec le score de silhouette.

Modèles testés : `KMeans`, `GaussianMixture`, `Birch`, `AgglomerativeClustering`.

## Résultats

Run retenu : `experiments/part3_server_segmentation/runs/20260703_134459/`

- meilleur modèle : `GaussianMixture`, covariance `diag`
- nombre de clusters : `3`
- score de silhouette : `0.1006`
- Davies-Bouldin : `2.5423`
- Calinski-Harabasz : `183.6313`
- serveurs segmentés : `2200`

Top 5 du benchmark :

- `GaussianMixture(diag)`, `k=3`, silhouette `0.1006`
- `GaussianMixture(diag)`, `k=4`, silhouette `0.0826`
- `KMeans`, `k=3`, silhouette `0.0824`
- `Birch`, `k=3`, silhouette `0.0770`
- `AgglomerativeClustering`, `k=3`, silhouette `0.0770`

Profils obtenus après correction du nommage automatique :

- cluster `0` : `240` serveurs, profil `serveurs stockage sollicite`, driver `disk_util_pct_mean (+12.03 vs moyenne clusters)`
- cluster `1` : `357` serveurs, profil `serveurs temperature elevee`, driver `temperature_c_mean (+4.52 vs moyenne clusters)`
- cluster `2` : `1603` serveurs, profil `serveurs standard`, driver `aucun ecart dominant`

Le profil `latence elevee` n'est pas retenu car les différences de latence entre clusters sont trop faibles pour être interprétées métier.

Fichiers à utiliser pour l'analyse métier :

- `model_metrics.csv`
- `cluster_profiles.csv`
- `server_cluster_assignments.csv`
- `figures/server_segments_pca.png`

## Limite

La segmentation dépend des variables disponibles et de l'agrégation par serveur. Elle sert à identifier des profils globaux, pas à détecter directement une panne ponctuelle.
