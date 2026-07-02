# Projet Machine Learning Avancé

## Compétences évaluées

1. Choisir un modèle un modèle ML en fonction du problème
2. Créer un modèle ML de manière professionnelle
3. Utiliser un modèle ML de manière professionnelle
4. Déterminer une architecture adéquate pour un SI utilisant des modèles ML

# Système Intelligent d’Exploitation d’une Infrastructure Cloud

## Contexte

Vous êtes une équipe Data Science travaillant pour un fournisseur de services cloud comparable à OVHcloud.

L'entreprise exploite plusieurs milliers de serveurs répartis dans plusieurs datacenters. Les équipes d'exploitation souhaitent améliorer la qualité de service en anticipant les incidents, en détectant les comportements anormaux et en priorisant les actions de maintenance.

Pour cela, vous devez concevoir un système décisionnel basé sur plusieurs modèles de Machine Learning.

## Objectif métier

Construire un système capable d'attribuer à chaque serveur un score global de risque opérationnel permettant aux équipes techniques de prioriser leurs interventions.

## Partie 1 : Modèle de prédiction des incidents

Prédire `incident_next_7d` : indique si un serveur connaîtra au moins un incident dans les 7 jours suivant l'observation étudiée.

## Partie 2 : Prévision de charge support

Prédire `support_tickets` : représente le nombre de tickets support ouverts par les clients pour une région donnée et une journée donnée.

## Partie 3 : Segmentation des serveurs

Identifier différents profils de serveurs.

Exemples : serveurs fortement sollicités ; serveurs instables ; etc.

## Partie 4 : Détection d'anomalies

Détecter automatiquement les comportements anormaux pouvant précéder une panne.

## Partie 5 : Construction du score de risque global

Vous devez implémenter un score de risque global adéquate aux problématiques métier.

## Partie 6 : Priorisation intelligente des interventions préventives

Problématique : Chaque matin, l'équipe d'exploitation peut intervenir sur un maximum de 50 serveurs.

Vous devez construire un système capable de répondre à la question suivante : Quels sont les 50 serveurs sur lesquels une intervention préventive apportera le plus de valeur à l'entreprise ?

## Partie 7 : Interprétabilité

Vous devez fournir des explications globales et locales pour les modèles compatibles.

## Partie 8 : Architecture

Vous devez proposer une architecture de mise en production. Un schéma d'architecture est attendu.

## Rendus

Archive zip nommé `ml-nom-prenom.zip` d’un répertoire nommé `ml-nom-prenom` contentant à envoyer avant le **jeudi 09 juillet 2026 à 20h** par mail à <ludivine.crepin@ext-esiea.fr> :

- Les scripts python du pipeline ML
- Une `README.md` technique
- L’explication en français de l’interprétabilité des modèles
- L’explication en français des choix finaux des modèles
- L’architecture du SI choisie

Soutenance en tête à tête sur le projet le **vendredi 10 juillet 2026 de 13h30 à 17h15**.