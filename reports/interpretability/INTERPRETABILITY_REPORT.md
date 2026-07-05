# Partie 7 - Interpretabilite

Ce projet fournit maintenant une explication directement avec chaque resultat de modele dans le frontend. L'utilisateur ne lance pas une action separee: la prediction et son interpretation sont affichees ensemble.

## Synthese des methodes

| Modele | Type | Methode d'explication | Justification |
| --- | --- | --- | --- |
| Incident | `RandomForestClassifier` | SHAP local, `TreeExplainer` | Modele d'arbres compatible avec SHAP. Les valeurs SHAP indiquent les variables qui augmentent ou diminuent la probabilite d'incident. |
| Support | `ExtraTreesRegressor` | SHAP local, `TreeExplainer` | Modele d'arbres compatible avec SHAP. Les valeurs SHAP indiquent les variables qui augmentent ou diminuent la prevision de tickets. |
| Segmentation | `GaussianMixture` | Profil de cluster | SHAP n'est pas adapte naturellement a ce clustering non supervise. L'interpretation repose sur le profil appris du cluster et son facteur distinctif. |
| Anomalie | `LocalOutlierFactor` | Perturbation locale vers valeurs normales | SHAP n'est pas naturellement adapte a LOF. L'explication mesure comment le score d'anomalie baisse quand une variable est remplacee par sa reference normale. |

## Explications locales dans le frontend

Chaque carte de resultat affiche:

- la prediction principale;
- une phrase d'explication humaine;
- les principaux facteurs explicatifs;
- la methode utilisee (`shap`, `cluster_profile` ou `reference_perturbation`).

Pour les modeles incident et support, les impacts positifs augmentent la prediction du modele et les impacts negatifs la diminuent.

## Incident

Le modele incident est un `RandomForestClassifier`. Pour chaque prediction, le service calcule les valeurs SHAP sur la ligne demandee apres preprocessing et selection de variables.

Interpretation:

- `impact > 0`: la variable pousse le modele vers un risque incident plus eleve;
- `impact < 0`: la variable reduit le risque estime;
- les variables sont triees par impact absolu.

Limite: les variables affichees peuvent etre des variables transformees par le pipeline, par exemple des colonnes one-hot encodees pour les variables categorielles.

## Support

Le modele support est un `ExtraTreesRegressor`. SHAP explique la prevision du nombre de tickets support.

Interpretation:

- `impact > 0`: la variable augmente la prevision de tickets;
- `impact < 0`: la variable diminue la prevision;
- le resultat conserve aussi la marge d'erreur MAE pour rappeler l'incertitude du modele.

## Segmentation

Le modele de segmentation est un `GaussianMixture`. Comme il s'agit d'un modele non supervise, l'explication repose sur les profils de clusters sauvegardes dans `models/metadata/server_segmentation_gaussian_mixture.json`.

Le frontend affiche:

- le cluster predit;
- le nom du profil;
- le facteur distinctif du profil;
- la probabilite de cluster quand elle est disponible.

## Anomalie

Le modele anomalie est un `LocalOutlierFactor`. L'explication locale compare chaque signal a une valeur normale de reference issue du train.

Pour chaque variable candidate:

1. le score d'anomalie initial est calcule;
2. la variable est remplacee par sa reference normale;
3. l'impact correspond a la baisse du score d'anomalie;
4. les plus forts impacts sont affiches.

Cette methode est plus adaptee que SHAP pour ce cas, car LOF repose sur la densite locale et non sur une fonction d'arbre directement explicable.

## Limites generales

Les explications sont locales: elles expliquent une prediction precise, pas tout le comportement du modele. Elles doivent etre lues comme une aide a la decision operationnelle, pas comme une preuve causale. Les resultats dependent aussi du preprocessing et des approximations utilisees a l'inference en ligne pour les variables historiques.
