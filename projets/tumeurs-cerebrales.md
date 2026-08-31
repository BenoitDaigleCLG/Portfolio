# Détection de tumeurs cérébrales
### Réseau de neurones profonds — Hiver 2026

[← Retour au portfolio](../)

---

## Contexte

Projet réalisé en équipe de 5 dans le cadre du cours GLO-7030 (Université Laval), explorant différentes modifications architecturales de la famille YOLO (v8 et v11) pour la détection et la localisation de tumeurs cérébrales sur IRM. Deux axes ont été étudiés : l'intégration de mécanismes d'attention (PSA, C2PSA, CBAM) et l'ajout d'une tête de détection haute résolution pour mieux capter les micro-lésions.

En imagerie médicale, un faux négatif — une tumeur non détectée — peut retarder un diagnostic vital. L'enjeu principal était donc de comprendre comment ces modifications affectent le compromis entre sensibilité (rappel) et précision de délimitation spatiale.

## Données

**Brain Tumor Detection Dataset** (Kaggle) : 3 903 images IRM annotées, réparties en 4 classes — Gliome, Méningiome, Tumeur pituitaire, Sans tumeur (63% entraînement / 28% validation / 10% test).

*Note : les expériences sur YOLOv8n ont utilisé un second dataset (Roboflow, 1 956 images, 5 classes) en raison de contraintes de format.*

## Méthodologie

- **Axe 1 — Attention :** intégration des modules **PSA** (Position-Sensitive Attention) sur YOLOv11n et **CBAM** (Convolutional Block Attention Module) sur YOLOv8n et YOLOv11s, positionnés stratégiquement pour affiner la détection des contours flous (notamment les Gliomes).
- **Axe 2 — Haute résolution :** conception d'une architecture expérimentale **YOLOv11-HR-D**, ajoutant une 4ème tête de détection (stride 4, grille 160×160) pour les micro-lésions, combinée à des connexions résiduelles profondes (C3k2).
- Entraînement sur GPU Tesla T4 (Google Colab), 50 époques, optimiseur AdamW, augmentation de données (RandAugment, Mosaic).

## Résultats

| Modèle | Précision | Rappel | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| YOLOv11n baseline | 0.920 | 0.892 | 0.946 | 0.739 |
| YOLOv11s + P2 Head + C2PSA | **0.945** | **0.912** | **0.957** | 0.752 |
| YOLOv11-HR-D (P2 + C3k2) | 0.898 | 0.888 | 0.940 | **0.741** |

**Points clés :**
- Le module **PSA** améliore toutes les métriques simultanément, avec la plus forte hausse sur la classe Gliome (+0,9% en mAP@50) — la classe la plus difficile à délimiter en raison de ses contours diffus.
- L'architecture **HR-D** est la seule à améliorer le mAP@50-95 (précision géométrique stricte), au prix d'une légère baisse du rappel — un compromis clinique entre outil de *triage* rapide et outil d'assistance chirurgicale de précision.
- Le CBAM appliqué de façon massive (YOLOv11s) a dégradé le rappel de 1,4 point, suggérant qu'un excès de modules d'attention peut nuire à l'extraction de caractéristiques.

## Conclusion

L'attention légère (PSA) s'est révélée l'optimisation la plus stable, améliorant la détection sans alourdir le modèle. Les architectures plus complexes (HR-D, CBAM massif) imposent des compromis cliniques clairs entre sensibilité et précision de délimitation. Un entraînement plus long (150 époques + early-stopping) serait nécessaire pour permettre aux architectures haute-résolution de converger pleinement.

---

**Technologies :** Python · PyTorch · Ultralytics YOLO · Google Colab

[Voir le rapport complet (PDF)](#) &nbsp;|&nbsp; [Voir le code sur GitHub](#)
