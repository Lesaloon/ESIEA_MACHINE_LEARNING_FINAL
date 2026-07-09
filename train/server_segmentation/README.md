# Training final - Partie 3 segmentation serveurs

Ce dossier contient le training final de la segmentation reconstruite a partir des donnees brutes :

- `data/raw/servers.csv`
- `data/raw/daily_server_usage.csv`
- `data/raw/incidents.csv`

Le script :

- agrege les observations au niveau `server_id` ;
- construit des features de charge, variabilite, recence et historique d'incidents ;
- applique un pipeline de preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `VarianceThreshold`) ;
- teste `GaussianMixture` et `KMeans` avec `RandomizedSearchCV` ;
- retient le meilleur modele selon le `test_silhouette`.

Commande :

```bash
.venv/bin/python train/server_segmentation/train_final_gaussian_mixture.py --n-iter 8 --cv-splits 3
```

Sorties principales :

- `models/artifacts/server_segmentation_kmeans.pkl`
- `models/metadata/server_segmentation_kmeans.json`
- `train/server_segmentation/runs/<timestamp>/`

Run de reference actuel : `train/server_segmentation/runs/20260707_224440/`

Resultat retenu sur ce run :

- meilleur modele : `KMeans`
- nombre de clusters : `5`
- meilleur CV silhouette : `0.1644`
- silhouette test : `0.1783`
