# Partie 3 - Segmentation des serveurs

Objectif : identifier des profils de serveurs sans utiliser de cible supervisée.

Le script agrège les observations journalières par `server_id`, compare plusieurs algorithmes de clustering et plusieurs valeurs de `k`, puis sauvegarde les clusters, les profils et le modèle final.

Modèles testés :

- `KMeans`
- `GaussianMixture` covariance `diag`
- `GaussianMixture` covariance `tied`
- `Birch`
- `AgglomerativeClustering`

## Lancer l'expérience

```bash
.venv/bin/python experiments/part3_server_segmentation/run_segmentation_experiments.py --min-clusters 3 --max-clusters 8
```

## Sorties

Les sorties sont générées dans :

```bash
experiments/part3_server_segmentation/runs/<timestamp>/
```

Fichiers principaux :

- `model_metrics.csv` : comparaison des modèles et valeurs de `k`.
- `cluster_profiles.csv` : profil métier de chaque segment.
- `server_cluster_assignments.csv` : cluster attribué à chaque serveur.
- `server_segmentation_<model>.pkl` : pipeline final sauvegardé avec `joblib.dump`.
- `figures/server_segments_pca.png` : visualisation PCA des segments.
