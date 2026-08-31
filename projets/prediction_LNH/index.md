# Prédiction du rang de repêchage — LNH
### Méthodes d'analyse de données — Hiver 2026

[← Retour au portfolio](../../)

---

## Contexte

Projet réalisé dans le cadre du cours STT-7335 (Université Laval), visant à évaluer et modéliser l'impact du volume versus l'efficacité des statistiques pré-repêchage sur le rang de sélection d'un joueur dans la LNH (`DRAFT_OVERALL`). L'étude s'appuie sur une base de 3 560 joueurs combinant historiques de repêchage, informations individuelles et statistiques de ligues d'origine.

## Ce qui a été développé

### 1. Analyse en composantes principales (ACP) & EDA
- Analyse exploratoire et réduction de dimensionnalité sur 11 variables quantitatives (standardisées).
- Identification de 3 composantes principales (76 % de la variance cumulée) : **PC1** (efficacité offensive), **PC2** (gabarit physique) et **PC3** (progression et âge).
- Démonstration statistique (ANOVA, $F=74{,}31$, $p < 10^{-31}$) que seule la composante d'efficacité offensive sépare significativement les joueurs du Top 32 des rondes suivantes.

### 2. Modélisation par régression linéaire
- Évaluation comparative entre le volume pur (`TOTAL_P`) et l'efficacité (`MEAN_PPG`, `LAST_PPG`, `MAX_PPG`).
- Ajout du nombre de parties jouées (`TOTAL_GP`) comme variable de contrôle pour isoler l'effet du temps de jeu.
- Le modèle complet démontre la supériorité de l'efficacité récente (`LAST_PPG`, coefficient de $-31{,}56$), confirmant que les dépisteurs valorisent la production par match récente plutôt que le volume cumulé brut.

### 3. Pipeline Random Forest (`process_data.py` & `Foret_aleatoire`)
- Ingénierie de variables avancées : équivalence de niveau de ligue (`NHLe_PPG` basé sur les facteurs de conversion de Patrick Bacon), maturité précise (`EXACT_AGE_AT_DRAFT`), ratio buts/passes (`PLAYMAKER_RATIO`) et momentum de progression.
- Nettoyage de la colinéarité, encodage one-hot des variables catégorielles et optimisation des hyperparamètres par recherche sur grille (Grid Search) sur les rondes 1 et 2 ($N=1\,232$).

## Résultats clés

- **Pouvoir prédictif du Random Forest :** $R^2 = 0{,}22$, avec une erreur absolue moyenne (**MAE**) de **13,36 rangs** et un **RMSE** de **16,21**.
- **Précision par intervalles :** 13,51 % des sélections prédites à $\pm 3$ rangs près et 62,43 % à $\pm 16$ rangs.
- **Facteurs déterminants :** Les trois variables les plus influentes sont `NHLe_PPG` (qualité/difficulté de la ligue), `EXACT_AGE_AT_DRAFT` (âge précis au repêchage) et `LAST_P_PER_GP` (production récente).

## Conclusion

L'analyse confirme que la qualité et le contexte de la production (efficacité ajustée à la ligue et maturité) priment largement sur le volume brut accumulé. Le plafonnement du $R^2$ à 0,22 illustre toutefois le caractère non déterministe du repêchage et le bruit inhérent aux choix avancés, où les aspects qualitatifs intangibles (vision de jeu, éthique de travail, entrevues) ne peuvent être capturés par les seules statistiques publiques.

---

**Technologies :** Python · Scikit-Learn · Pandas · NumPy · Seaborn / Matplotlib

[Voir le rapport complet (PDF)](Rapport_LNH.pdf) &nbsp;|&nbsp; [Voir le code sur GitHub](code/)
