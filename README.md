# ML Project

Projet ML conteneurise pour prioriser les operations preventives sur une infrastructure cloud.

## Structure

- `data/` : donnees brutes, transformees et predictions generees.
- `models/` : registre, artefacts et metadonnees des modeles.
- `services/frontend/` : frontend statique servi par Nginx.
- `services/api-gateway/` : API publique qui route vers les services d'inference internes.
- `services/incident-inference-service/` : microservice d'inference du modele incident.
- `services/support-inference-service/` : microservice d'inference du modele support.
- `services/segmentation-inference-service/` : microservice d'inference du modele segmentation.
- `services/anomaly-inference-service/` : microservice d'inference du modele anomalie.
- `services/shared-inference/` : code commun de schemas, feature engineering et chargement des artefacts.
- `services/model-storage/` : service ou conteneur dedie au stockage des modeles.
- `shared/` : configuration, constantes, logger et utilitaires communs.
- `notebooks/` : notebooks d'exploration et d'experimentation.
- `tests/` : tests automatises.
- `scripts/` : scripts d'execution courants.

## Demarrage

```bash
docker compose up --build
```

Services Docker principaux :

- `frontend` : Nginx public sur `http://localhost:8000` par defaut.
- `api-gateway` : API publique sur `http://localhost:8001` par defaut.
- `incident-inference-service` : API interne du modele incident.
- `support-inference-service` : API interne du modele support.
- `segmentation-inference-service` : API interne du modele segmentation.
- `anomaly-inference-service` : API interne du modele anomalie.

Le frontend Nginx proxifie les appels API vers le gateway. Le gateway appelle les services d'inference et agrege leurs resultats, notamment pour la priorisation des interventions.

## Interpretabilite

Les resultats affiches par le frontend incluent directement une explication locale:

- incident : SHAP sur le `RandomForestClassifier` ;
- support : SHAP sur le `ExtraTreesRegressor` ;
- segmentation : interpretation par profil de cluster ;
- anomalie : perturbation locale vers les valeurs normales de reference.

Le rapport Partie 7 est disponible dans `reports/interpretability/INTERPRETABILITY_REPORT.md`.

Pour personnaliser la configuration locale, copier `.env.example` vers `.env`.

## Notes

Les artefacts de modeles deployes sont dans `models/artifacts/` et leurs metadonnees dans `models/metadata/`.
