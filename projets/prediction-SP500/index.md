---
title: "Benoit Daigle"
---

# Prédiction des rendements du S&P 500
*Analyse de données (STT-2200) — Automne 2025 (Revue Été 2026)*

[← Retour au portfolio](../../)

---

## Contexte

Projet réalisé dans le cadre du cours en analyse de données, visant à évaluer la prédictibilité du signe des rendements à 21 jours (horizon ~1 mois) du S&P 500. L'objectif initial était de concevoir une stratégie de synchronisation de marché (*market timing*) permettant de surperformer une détention passive (*Buy & Hold*) en désinvestissant lors des phases baissières anticipées.

## Données et prétraitement (R)

Le pipeline initial s'appuie sur 20 ans d'historique quotidien combinant indicateurs techniques et données macroéconomiques :
- **Indicateurs techniques & volatilité (TTR / quantmod) :** RSI (14j), Momentum (10j), Average True Range (ATR 14j), écart-type roulant (20j), On-Balance Volume (OBV), Chaikin Money Flow (CMF 20j), Money Flow Index (MFI 14j), CCI (20j) et ratio prix/moyennes mobiles (SMA 20/200).
- **Données macroéconomiques (FRED / Yahoo Finance) :** Niveau de clôture du VIX, cours de l'or et du pétrole WTI, ainsi que les variations à 1 jour et 1 mois du spread de taux (10 ans – 2 ans) et du spread de crédit IG–GOV (BBB vs obligations souveraines).
- **Réduction de dimension (FactoMineR) :** Analyse en composantes principales (ACP) pour identifier les facteurs dominants et élimination des variables colinéaires (ex. ATR vs volatilité roulante, OBV vs prix de l'or) pour aboutir à un ensemble optimisé de 10 indicateurs clés.

## Modélisation et validation (Python)

- **Algorithme :** Entraînement d'un classificateur non linéaire XGBoost avec régularisation stricte (faible profondeur, sous-échantillonnage, `reg_alpha` et `reg_lambda`).
- **Validation Walk-Forward :** Évaluation temporelle séquentielle simulant un investissement en conditions réelles avec ré-entraînement quotidien sur fenêtre glissante de 250 jours.
- **Stratégie de réévaluation :** Backtest par blocs de 21 jours comparant un capital initial de 10 000 $ investi en *Buy & Hold* contre la stratégie active guidée par les prédictions.

## Résultats initiaux

- **Exactitude directionnelle :** Le modèle affichait une précision globale de ~72 %, surpassant la proportion historique naturelle de rendements positifs de la période (~67 %).
- **Surperformance apparente :** La courbe de richesse cumulée montrait un net découplage en faveur du modèle par rapport à l'indice de référence, grâce à une évitement apparent des baisses majeures.

---

## Rétrospective critique (Revue Été 2026) : Détection d'un *Look-Ahead Bias*

Durant l'été 2026, j'ai repris ce projet afin d'approfondir les résultats et auditer la robustesse du code. Cette analyse a révélé un biais méthodologique critique dans le pipeline d'entraînement : un **biais d'anticipation (*look-ahead bias*) causé par le chevauchement des cibles (*overlapping target labels*)**.

### Mécanisme du biais identifié :
1. **Construction de la cible :** La cible calculée à l'instant *t* est le rendement cumulé futur R(t → t+21).
2. **Fenêtre glissante pas-à-pas :** Lorsque le modèle est entraîné sur une fenêtre se terminant à *t-1* pour prédire la direction à l'instant *t*, les dernières lignes du jeu d'entraînement contiennent les cibles R(t-20 → t+1), R(t-19 → t+2), ..., R(t-1 → t+20).
3. **Fuite d'information (*Data Leakage*) :** Le modèle apprenait à partir d'étiquettes qui incorporaient déjà l'évolution des prix jusqu'à *t+20*. Bien que les variables explicatives (*features*) soient passées, la cible d'entraînement contenait de l'information sur le futur immédiat de la date de test *t*, faussant artificiellement la précision du *walk-forward* et les rendements du backtest.

Cette révision m'a permis de documenter l'importance cruciale de purger les données d'entraînement (*purging & embargoing*) lors de l'utilisation de cibles multi-horizons en séries temporelles financières.

---

**Outils et bibliothèques :** R (quantmod, TTR, FactoMineR, data.table) · Python (XGBoost, Scikit-Learn, Pandas, NumPy, Matplotlib)

[Voir le poster du projet (PDF)](Poster_projet.pdf) &nbsp;|&nbsp; [Voir le code sur GitHub](https://github.com/BenoitDaigleCLG/Portfolio/tree/main/projets/prediction-SP500/code)
