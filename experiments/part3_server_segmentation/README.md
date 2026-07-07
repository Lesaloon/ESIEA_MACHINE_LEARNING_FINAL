# Partie 3 - Segmentation des serveurs

Objectif : identifier des profils de serveurs sans utiliser de cible supervisée.

Le workflow a ete mis a jour le `2026-07-07` pour repartir des fichiers bruts et suivre la meme logique de pipeline que les nouveaux experiments incident : preprocessing scikit-learn complet, recherche aleatoire et evaluation sur holdout.

Jeux de donnees utilises :

- `data/raw/servers.csv`
- `data/raw/daily_server_usage.csv`
- `data/raw/incidents.csv`

Modeles testes :

- `KMeans`
- `GaussianMixture`

Features construites :

- signaux moyens / max / std / p90 de charge et temperature ;
- recence sur les 7 derniers points ;
- taux de jours fortement charges ;
- historique d'incidents agrege par serveur ;
- variables derivees comme `incident_rate`, `recent30_incident_rate`, `recent_usage_shift`.

## Lancer l'expérience

```bash
.venv/bin/python experiments/part3_server_segmentation/run_raw_segmentation_experiment.py --n-iter 8 --cv-splits 3
```

## Sorties

Les sorties sont générées dans :

```bash
experiments/part3_server_segmentation/runs/<timestamp>/
```

Fichiers principaux :

- `benchmark_metrics.csv` : comparaison des modeles et meilleurs hyperparametres tunes.
- `server_feature_table.csv` : table finale une ligne par serveur.
- `cluster_profiles.csv` : profil métier de chaque segment.
- `server_cluster_assignments.csv` : cluster attribué à chaque serveur.
- `server_segmentation_<model>.pkl` : pipeline final sauvegardé avec `joblib.dump`.
- `figures/server_segments_pca.png` : visualisation PCA des segments.

Run brut de reference actuel : `experiments/part3_server_segmentation/raw_runs/20260707_224359/`
