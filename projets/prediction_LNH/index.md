### Méthodes d'analyse de données (STT-7335) — Hiver 2026

[← Retour au portfolio](../../)

---

## Contexte

Projet d'équipe réalisé dans le cadre du cours STT-7335 (Université Laval), évaluant si le volume (quantité brute de statistiques) ou l'efficacité (production par match) des performances avant repêchage est le plus déterminant pour expliquer le rang de sélection d'un joueur dans la LNH (`DRAFT_OVERALL`).

**Ma contribution :** conception et implémentation du pipeline de préparation de données avancé pour l'apprentissage automatique (`process_data.py`) et développement complet de la modélisation par forêts aléatoires (`Foret_aleatoire.py`), incluant l'ingénierie de variables spécifiques, le traitement de la colinéarité, l'optimisation des hyperparamètres et l'analyse de performance. Le reste du rapport présente les travaux de l'équipe sur l'analyse descriptive, l'ACP et la régression linéaire.

## Construction et préparation du jeu de données

Le jeu de données global résulte de la fusion et de l'enrichissement de sources publiques Kaggle (*Elite Prospects* et *NHL Draft 1963–2022*) :
- **Fusion multi-sources :** Appariement par nom et année de naissance recalculée (`year - age`) pour récupérer les rangs de sélection manquants, suivi d'une jointure par `PLAYER_ID` avec l'historique saisonnier détaillé.
- **Nettoyage & Agrégation :** Élimination des doublons ambigus, imputation par médiane des valeurs manquantes d'efficacité, gestion des divisions par zéro et agrégation au niveau joueur (3 583 joueurs, 34 variables).

## Ce que j'ai développé

### 1. Ingénierie des données pour le Machine Learning (`process_data.py`)
Génération d'un jeu de données enrichi spécifiquement pour la modélisation non linéaire :
- **Facteurs d'équivalence de ligue (`NHLe_PPG`) :** Pondération de la production selon la difficulté relative des circuits (KHL à 0,80, SHL à 0,58, ligues juniors canadiennes de 0,26 à 0,32, etc.) d'après le modèle de P. Bacon.
- **Maturité et profils :** Calcul de l'âge exact continu au moment du repêchage (`EXACT_AGE_AT_DRAFT`), des ratios de profil offensif (`PLAYMAKER_RATIO`, `SNIPER_RATIO`) et de la dynamique d'amélioration (`MOMENTUM_PPG`).
- **Filtrage ciblé :** Restriction aux choix des deux premières rondes (N = 1 232) pour éliminer le bruit aléatoire des rondes tardives, et sélection des joueurs ayant disputé au moins 5 matchs.

### 2. Modèle de forêt aléatoire (`Foret_aleatoire.py`)
- **Traitement de la colinéarité :** Analyse matricielle pour éliminer les redondances strictes (conservation de `LAST_P_PER_GP` et retrait des métriques redondantes de points/buts/passes).
- **Ajustement & Recherche sur grille :** Optimisation via Grid Search (`n_estimators`, `max_depth`, `min_samples_leaf`, `criterion`) avec séparation 70/30 (862 entraînement / 370 test) et encodage one-hot des variables catégorielles (`LAST_LEAGUE`, `PRIMARY_POS`, `NATIONALITY`).
- **Évaluation :** Analyse des résidus, calcul du MAE, RMSE, R² et précision par fenêtres de tolérance.

## Résultats clés (Random Forest)

- **Performance globale :** R² = 0,22 avec une erreur absolue moyenne (**MAE**) de **13,36 rangs** et un **RMSE** de **16,21**.
- **Précision par intervalles :** 13,51 % des sélections prédites à ±3 rangs, 42,43 % à ±10 rangs et 62,43 % à ±16 rangs.
- **Variables les plus déterminantes :** L'importance des variables confirme que la qualité et le contexte priment, avec en tête `NHLe_PPG`, `LAST_P_PER_GP` et `EXACT_AGE_AT_DRAFT`.

## Conclusion

La modélisation démontre que l'efficacité pondérée par la difficulté de la ligue et l'âge exact au repêchage sont nettement plus prédictifs que le volume de points cumulé. Le plafonnement du pouvoir prédictif (R² = 0,22) met en lumière le caractère conservateur du modèle et les limites des données statistiques publiques, qui ne capturent pas les attributs intangibles (vision de jeu, éthique de travail, entrevues) décisifs pour les équipes.

---

**Outils et bibliothèques :** Python · Scikit-Learn · Pandas · NumPy · Seaborn / Matplotlib

[Voir le rapport complet (PDF)](Rapport_LNH.pdf) &nbsp;|&nbsp; [Voir le document de construction du dataset (PDF)](construction_dataset.pdf) &nbsp;|&nbsp; [Voir le code sur GitHub](https://github.com/BenoitDaigleCLG/Portfolio/tree/main/projets/prediction_LNH/code)
