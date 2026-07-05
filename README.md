# ML Project

Projet ML conteneurise pour prioriser les operations preventives sur une infrastructure cloud.

## Structure

- `data/` : donnees brutes, transformees et predictions generees.
- `models/` : registre, artefacts et metadonnees des modeles.
- `services/inference-service/` : API gateway et frontend. Le gateway appelle les services modele internes.
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

- `inference-service` : gateway public + frontend sur le port `8000`.
- `incident-model-service` : API interne du modele incident.
- `support-model-service` : API interne du modele support.
- `segmentation-model-service` : API interne du modele segmentation.
- `anomaly-model-service` : API interne du modele anomalie.

Le frontend appelle uniquement le gateway. Le gateway appelle les services modele et agrege leurs resultats, notamment pour la priorisation des interventions.

Pour personnaliser la configuration locale, copier `.env.example` vers `.env`.

## Notes

Les artefacts de modeles deployes sont dans `models/artifacts/` et leurs metadonnees dans `models/metadata/`.
