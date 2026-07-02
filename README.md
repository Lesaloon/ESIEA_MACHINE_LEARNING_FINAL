# ML Project

Architecture generique pour un projet de machine learning conteneurise.

## Structure

- `data/` : donnees brutes, transformees et predictions generees.
- `models/` : registre, artefacts et metadonnees des modeles.
- `services/inference-service/` : service d'inference generique, a adapter ou dupliquer par modele.
- `services/model-storage/` : service ou conteneur dedie au stockage des modeles.
- `shared/` : configuration, constantes, logger et utilitaires communs.
- `notebooks/` : notebooks d'exploration et d'experimentation.
- `tests/` : tests automatises.
- `scripts/` : scripts d'execution courants.

## Demarrage

```bash
docker compose up --build
```

Pour personnaliser la configuration locale, copier `.env.example` vers `.env`.

## Notes

Cette base ne contient pas d'implementation specifique a un dataset, un modele ou un cas metier.
