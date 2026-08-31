# Détection de tumeurs cérébrales
### Réseau de neurones profonds — Hiver 2026

[← Retour au portfolio](../../)

---

## Contexte

Projet réalisé en équipe de 5 dans le cadre du cours GLO-7030 (Université Laval), explorant différentes modifications architecturales de la famille YOLO pour la détection et la localisation de tumeurs cérébrales sur IRM (Gliome, Méningiome, Tumeur pituitaire, Sans tumeur).

**Ma contribution :** conception et implémentation de l'architecture avec mécanisme d'attention **PSA (Position-Sensitive Attention)**, incluant le script d'entraînement et le pipeline de génération des résultats. Le reste du rapport présente les contributions de l'équipe sur les autres axes explorés (CBAM, architecture haute résolution HR-D).

## Ce que j'ai développé

### 1. Architecture custom (`architecture_custom.py`)
Génération dynamique d'un fichier de configuration YOLO11 modifié, intégrant une couche d'attention **PSA** insérée stratégiquement après la branche dédiée aux petits détails (contours fins), avant la tête de détection. L'entraînement s'est fait à résolution réduite (512×512) pour compenser la surcharge calculatoire du mécanisme d'attention.

### 2. Script d'entraînement (`Main_2.py`)
Pipeline complet configurant automatiquement le dataset (chemins RoboFlow), chargeant les poids pré-entraînés YOLO11 comme point de départ, puis lançant l'entraînement avec les hyperparamètres du projet (AdamW, seed fixe à 42, early stopping, mosaic augmentation désactivé en fin d'entraînement).

### 3. Génération des résultats (`generer_resultats.py`)
Script d'évaluation calculant les métriques clés (Précision, Rappel, F1-Score, mAP@50) sur les jeux de validation et de test, globalement et par classe pathologique, exportées dans un rapport texte structuré avec Pandas.

## Résultats clés (PSA)

Le module PSA a amélioré toutes les métriques simultanément par rapport au modèle de base, avec la plus forte hausse observée sur la classe **Gliome** (la plus difficile à délimiter en raison de ses contours diffus). Détails complets dans le rapport ci-dessous.

## Conclusion

L'attention légère (PSA) s'est révélée l'optimisation la plus stable du projet, améliorant la détection de textures diffuses sans alourdir significativement le modèle — contrairement aux architectures plus complexes testées par l'équipe (CBAM massif, haute résolution), qui imposaient des compromis plus marqués entre sensibilité et précision.

---

**Technologies :** Python · PyTorch · Ultralytics YOLO11 · Pandas · Google Colab

[Voir le rapport complet (PDF)](rapport-tumeurs-cerebrales.pdf) &nbsp;|&nbsp; [Voir le code sur GitHub](code/)
