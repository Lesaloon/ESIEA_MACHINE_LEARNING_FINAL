# Architecture de mise en production

Ce schema represente l'architecture Docker actuelle du projet. Le frontend est servi par Nginx, les appels API passent par un gateway FastAPI, puis chaque modele est expose par son propre microservice d'inference isole.

```mermaid
flowchart LR
    user[Utilisateur navigateur]

    subgraph public[Zone publique]
        frontend[Frontend Nginx<br/>Service: frontend<br/>Port public: 8000]
        gateway[API Gateway FastAPI<br/>Service: api-gateway<br/>Port public: 8001]
    end

    subgraph private[Reseau Docker interne ml-network]
        incident[Incident inference API<br/>Service: incident-inference-service<br/>Modele: GradientBoostingClassifier]
        support[Support forecast API<br/>Service: support-inference-service<br/>Modele: ExtraTreesRegressor]
        segmentation[Segmentation inference API<br/>Service: segmentation-inference-service<br/>Modele: KMeans]
        anomaly[Anomaly inference API<br/>Service: anomaly-inference-service<br/>Modele: LocalOutlierFactor]
        storage[Model storage<br/>Service: model-storage]
    end

    subgraph artifacts[Volumes et artefacts]
        models[(./models<br/>artifacts + metadata)]
        incidentModel[incident_gradient_boosting_model.pkl]
        supportModel[support_extra_trees_model.pkl]
        segmentationModel[server_segmentation_kmeans.pkl]
        anomalyModel[anomaly_local_outlier_factor.pkl]
    end

    user -->|HTTP GET /| frontend
    user -->|HTTP API direct optionnel| gateway

    frontend -->|Proxy /predict| gateway
    frontend -->|Proxy /predict-support| gateway
    frontend -->|Proxy /predict-segmentation| gateway
    frontend -->|Proxy /predict-anomaly| gateway
    frontend -->|Proxy /prioritize-interventions| gateway

    gateway -->|POST /predict| incident
    gateway -->|POST /predict-support| support
    gateway -->|POST /predict-segmentation| segmentation
    gateway -->|POST /predict-anomaly| anomaly

    gateway -->|Priorisation: incident probability| incident
    gateway -->|Priorisation: contexte segment| segmentation
    gateway -->|Priorisation: contexte anomalie| anomaly

    incident -->|Lecture modele + metadata| models
    support -->|Lecture modele + metadata| models
    segmentation -->|Lecture modele + metadata| models
    anomaly -->|Lecture modele + metadata| models
    storage -->|Initialise /models| models

    models --> incidentModel
    models --> supportModel
    models --> segmentationModel
    models --> anomalyModel
```

## Responsabilites des services

| Service | Role | Exposition |
| --- | --- | --- |
| `frontend` | Sert l'interface HTML et proxifie les routes API vers le gateway | Public, `localhost:8000` |
| `api-gateway` | Point d'entree API, routage et agregation des resultats | Public optionnel, `localhost:8001` |
| `incident-inference-service` | Prediction du risque incident et explication SHAP | Interne Docker |
| `support-inference-service` | Prevision des tickets support et explication SHAP | Interne Docker |
| `segmentation-inference-service` | Attribution d'un profil serveur | Interne Docker |
| `anomaly-inference-service` | Detection d'anomalie et explication locale | Interne Docker |
| `model-storage` | Conteneur de stockage/initialisation du repertoire `/models` | Interne Docker |

## Flux principaux

1. L'utilisateur ouvre `http://localhost:8000`.
2. Nginx sert le frontend statique.
3. Les formulaires du frontend appellent des routes relatives comme `/predict` ou `/predict-anomaly`.
4. Nginx proxifie ces appels vers `api-gateway:8000` dans le reseau Docker.
5. Le gateway appelle le microservice d'inference concerne.
6. Chaque microservice charge uniquement son artefact depuis le volume `./models:/models`.
7. Le resultat revient au frontend avec la prediction et son explication.

## Flux de priorisation

La route `/prioritize-interventions` est agregee par le gateway:

- appel du modele incident pour obtenir `incident_probability`;
- appel du modele anomalie pour obtenir le contexte d'anomalie;
- appel du modele segmentation pour obtenir le profil serveur;
- calcul metier final: `priority_score = incident_probability * business_value`.

Cette route ne repose pas sur un modele dedie supplementaire: elle reutilise les modeles existants et une regle metier explicable.
