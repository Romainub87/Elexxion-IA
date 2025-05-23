# Elexxion-IA

## Description du projet

Elexxion-IA est un projet d’intelligence artificielle qui analyse et prédit des données socio-économiques, environnementales et électorales françaises. Il utilise des modèles de machine learning pour anticiper l’évolution de variables comme les points de bourse, le taux de chômage, la pollution atmosphérique, la criminalité et les résultats électoraux.

## Fonctionnalités principales

- **Prédictions** : Anticipe les points de bourse, le taux de chômage, la population, les jours de pics de pollution et les infractions pour les années à venir.
- **Prédiction électorale** : Prédit l’orientation politique du gagnant de la présidentielle sur plusieurs années.
- **Analyse des corrélations** : Identifie les relations entre différentes variables socio-économiques.
- **API Flask** : Fournit une interface RESTful pour entraîner les modèles et obtenir des prédictions.

## Structure du projet

- `main.py` : API Flask pour l’entraînement et la prédiction.
- `utils.py` : Fonctions utilitaires pour la récupération et le traitement des données.
- `requirements.txt` : Dépendances Python du projet.
- `model.h5` et `model_classif.h5` : Modèles de machine learning entraînés.
- `README.md` : Documentation du projet.

## Prérequis

- Python 3.10 ou supérieur
- Google Cloud SDK configuré pour accéder à BigQuery
- Les bibliothèques listées dans `requirements.txt`

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/Romainub87/Elexxion-IA.git
   cd Elexxion-IA
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   
## Utilisation

1. Lancez l'API Flask :
   ```bash
   python main.py
   ```
   
## Endpoints principaux
- **GET /** : Documentation de l’API.
- **GET /predict?annee=?** : Prédit les indicateurs socio-économiques pour les 3 prochaines années à partir de l'année en paramètre (par défaut 2025).  
- **GET /predict/securite?annee=?** : Prédit le nombre d’infractions par type pour les 3 prochaines années à partir de l'année en paramètre (par défaut 2025).  
- **GET /predict/election?annee=?** : Prédit l’orientation politique du gagnant de la présidentielle pour les 3 prochaines années à partir de l'année en paramètre (par défaut 2025).