import os
from ultralytics import YOLO
import numpy as np


CHEMIN_DATASET_YAML = 'data/data.yaml'
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
DOSSIER_SAUVEGARDE = 'mes_modeles_irm'

# Noms des classes — ordre identique à data.yaml
NOMS_CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']




def generer_yaml_modif1_p2(chemin_sortie):
    """Modification 1 : Ajout de la 4ème tête P2 uniquement."""
    yaml_content = """# Modif 1 : 4ème tête de détection P2 uniquement
nc: 4
scales:
  n: [0.33, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]       # 1-P2/4
  - [-1, 2, C3k2, [128, False]]      # 2  (False = standard)
  - [-1, 1, Conv, [256, 3, 2]]       # 3-P3/8
  - [-1, 2, C3k2, [256, False]]      # 4  (False = standard)
  - [-1, 1, Conv, [512, 3, 2]]       # 5-P4/16
  - [-1, 2, C3k2, [512, False]]      # 6  (False = standard)
  - [-1, 1, Conv, [1024, 3, 2]]      # 7-P5/32
  - [-1, 2, C3k2, [1024, False]]     # 8  (False = standard)
  - [-1, 1, SPPF, [1024, 5]]         # 9
  - [-1, 2, C2PSA, [1024]]           # 10 (gardé car présent dans yolo11n standard)

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 11
  - [[-1, 6], 1, Concat, [1]]                   # 12 concat P4
  - [-1, 2, C3k2, [512, False]]                 # 13

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 14
  - [[-1, 4], 1, Concat, [1]]                   # 15 concat P3
  - [-1, 2, C3k2, [256, False]]                 # 16

  # --- TÊTE P2 (160x160) ---
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 17
  - [[-1, 2], 1, Concat, [1]]                   # 18
  - [-1, 2, C3k2, [128, False]]                 # 19

  - [-1, 1, Conv, [128, 3, 2]]                  # 20
  - [[-1, 16], 1, Concat, [1]]                  # 21
  - [-1, 2, C3k2, [256, False]]                 # 22

  - [-1, 1, Conv, [256, 3, 2]]                  # 23
  - [[-1, 13], 1, Concat, [1]]                  # 24
  - [-1, 2, C3k2, [512, False]]                 # 25

  - [-1, 1, Conv, [512, 3, 2]]                  # 26
  - [[-1, 10], 1, Concat, [1]]                  # 27
  - [-1, 2, C3k2, [1024, False]]                # 28

  - [[19, 22, 25, 28], 1, Detect, [nc]]         # Detect(P2, P3, P4, P5)
"""
    with open(chemin_sortie, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"[INFO] Architecture Modif1 (P2) créee : {chemin_sortie}")


def generer_yaml_modif2_c2psa(chemin_sortie):
    """Modification 2 : C2PSA uniquement, 3 têtes standard."""
    yaml_content = """# Modif 2 : Bloc C2PSA uniquement (attention), 3 têtes standard
nc: 4
scales:
  n: [0.33, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]       # 1-P2/4
  - [-1, 2, C3k2, [128, False]]      # 2
  - [-1, 1, Conv, [256, 3, 2]]       # 3-P3/8
  - [-1, 2, C3k2, [256, False]]      # 4
  - [-1, 1, Conv, [512, 3, 2]]       # 5-P4/16
  - [-1, 2, C3k2, [512, False]]      # 6
  - [-1, 1, Conv, [1024, 3, 2]]      # 7-P5/32
  - [-1, 2, C3k2, [1024, False]]     # 8
  - [-1, 1, SPPF, [1024, 5]]         # 9
  - [-1, 2, C2PSA, [1024]]           # 10 MODIFICATION : bloc attention

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 11
  - [[-1, 6], 1, Concat, [1]]                   # 12
  - [-1, 2, C3k2, [512, False]]                 # 13

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 14
  - [[-1, 4], 1, Concat, [1]]                   # 15
  - [-1, 2, C3k2, [256, False]]                 # 16

  - [-1, 1, Conv, [256, 3, 2]]                  # 17
  - [[-1, 6], 1, Concat, [1]]                   # 18
  - [-1, 2, C3k2, [512, False]]                 # 19

  - [-1, 1, Conv, [512, 3, 2]]                  # 20
  - [[-1, 10], 1, Concat, [1]]                  # 21
  - [-1, 2, C3k2, [1024, False]]                # 22

  - [[16, 19, 22], 1, Detect, [nc]]             # Detect(P3, P4, P5)
"""
    with open(chemin_sortie, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"[INFO] Architecture Modif2 (C2PSA) creee : {chemin_sortie}")


def generer_yaml_modif3_c3k2true(chemin_sortie):
    """Modification 3 : C3k2 True uniquement, 3 têtes standard."""
    yaml_content = """# Modif 3 : C3k2 True uniquement (backbone plus profond), 3 têtes standard
nc: 4
scales:
  n: [0.33, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]       # 1-P2/4
  - [-1, 2, C3k2, [128, False]]      # 2
  - [-1, 1, Conv, [256, 3, 2]]       # 3-P3/8
  - [-1, 2, C3k2, [256, False]]      # 4
  - [-1, 1, Conv, [512, 3, 2]]       # 5-P4/16
  - [-1, 2, C3k2, [512, True]]       # 6  MODIFICATION
  - [-1, 1, Conv, [1024, 3, 2]]      # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]      # 8  MODIFICATION
  - [-1, 1, SPPF, [1024, 5]]         # 9
  - [-1, 2, C2PSA, [1024]]           # 10 (garde car present dans yolo11n standard)

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 11
  - [[-1, 6], 1, Concat, [1]]                   # 12
  - [-1, 2, C3k2, [512, False]]                 # 13

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 14
  - [[-1, 4], 1, Concat, [1]]                   # 15
  - [-1, 2, C3k2, [256, False]]                 # 16

  - [-1, 1, Conv, [256, 3, 2]]                  # 17
  - [[-1, 6], 1, Concat, [1]]                   # 18
  - [-1, 2, C3k2, [512, False]]                 # 19

  - [-1, 1, Conv, [512, 3, 2]]                  # 20
  - [[-1, 10], 1, Concat, [1]]                  # 21
  - [-1, 2, C3k2, [1024, False]]                # 22

  - [[16, 19, 22], 1, Detect, [nc]]             # Detect(P3, P4, P5)
"""
    with open(chemin_sortie, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"[INFO] Architecture Modif3 (C3k2 True) creee : {chemin_sortie}")


def generer_yaml_modif4_complet(chemin_sortie):
    """Modification 4 : Tout combiné — P2 + C2PSA + C3k2 True."""
    yaml_content = """# Modif 4 : Tout combiné — P2 + C2PSA + C3k2 True
nc: 4
scales:
  n: [0.33, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]       # 1-P2/4
  - [-1, 2, C3k2, [128, False]]      # 2
  - [-1, 1, Conv, [256, 3, 2]]       # 3-P3/8
  - [-1, 2, C3k2, [256, False]]      # 4
  - [-1, 1, Conv, [512, 3, 2]]       # 5-P4/16
  - [-1, 2, C3k2, [512, True]]       # 6  C3k2 True
  - [-1, 1, Conv, [1024, 3, 2]]      # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]      # 8  C3k2 True
  - [-1, 1, SPPF, [1024, 5]]         # 9
  - [-1, 2, C2PSA, [1024]]           # 10 C2PSA

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 11
  - [[-1, 6], 1, Concat, [1]]                   # 12
  - [-1, 2, C3k2, [512, False]]                 # 13

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 14
  - [[-1, 4], 1, Concat, [1]]                   # 15
  - [-1, 2, C3k2, [256, False]]                 # 16

  # --- TÊTE P2 (160x160) ---
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 17
  - [[-1, 2], 1, Concat, [1]]                   # 18
  - [-1, 2, C3k2, [128, False]]                 # 19

  - [-1, 1, Conv, [128, 3, 2]]                  # 20
  - [[-1, 16], 1, Concat, [1]]                  # 21
  - [-1, 2, C3k2, [256, False]]                 # 22

  - [-1, 1, Conv, [256, 3, 2]]                  # 23
  - [[-1, 13], 1, Concat, [1]]                  # 24
  - [-1, 2, C3k2, [512, False]]                 # 25

  - [-1, 1, Conv, [512, 3, 2]]                  # 26
  - [[-1, 10], 1, Concat, [1]]                  # 27
  - [-1, 2, C3k2, [1024, False]]                # 28

  - [[19, 22, 25, 28], 1, Detect, [nc]]         # Detect(P2, P3, P4, P5)
"""
    with open(chemin_sortie, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"[INFO] Architecture Modif4 (Tout combine) creee : {chemin_sortie}")


# ==========================================


def extraire_metriques(metrics_obj):
    """Extrait et calcule les métriques requises."""
    p = metrics_obj.box.mp
    r = metrics_obj.box.mr
    map50 = metrics_obj.box.map50
    f1 = 2 * (p * r) / (p + r + 1e-16)
    ap50_par_classe = metrics_obj.box.ap50
    return p, r, f1, map50, ap50_par_classe



def run_pipeline():
    print("=" * 60)
    print(" PIPELINE ABLATION STUDY — 5 MODÈLES")
    print("=" * 60)

    # Définition des 5 modèles
    modeles = [
        {
            'nom': 'baseline',
            'yaml': 'yolo11n.yaml',
            'label': 'Baseline (standard)',
            'generer': None,
        },
        {
            'nom': 'modif1_p2',
            'yaml': 'yolo11_modif1_p2.yaml',
            'label': 'Modif 1 — 4ème tête P2',
            'generer': generer_yaml_modif1_p2,
        },
        {
            'nom': 'modif2_c2psa',
            'yaml': 'yolo11_modif2_c2psa.yaml',
            'label': 'Modif 2 — C2PSA (attention)',
            'generer': generer_yaml_modif2_c2psa,
        },
        {
            'nom': 'modif3_c3k2true',
            'yaml': 'yolo11_modif3_c3k2true.yaml',
            'label': 'Modif 3 — C3k2 True',
            'generer': generer_yaml_modif3_c3k2true,
        },
        {
            'nom': 'modif4_complet',
            'yaml': 'yolo11_modif4_complet.yaml',
            'label': 'Modif 4 — Tout combiné',
            'generer': generer_yaml_modif4_complet,
        },
    ]

    # ENTRAÎNEMENT DE TOUS LES MODÈLES
    for i, modele in enumerate(modeles):
        print(f"\n{'='*60}")
        print(f" ENTRAÎNEMENT {i+1}/5 : {modele['label']}")
        print(f"{'='*60}")

        # Générer le YAML si nécessaire
        if modele['generer'] is not None:
            modele['generer'](modele['yaml'])

        # Entraînement
        model = YOLO(modele['yaml'])
        model.train(
            data=CHEMIN_DATASET_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=0,
            project=DOSSIER_SAUVEGARDE,
            name=modele['nom'],
            save=True,
            plots=True
        )

    # ÉVALUATION DE TOUS LES MODÈLES
    print(f"\n{'='*60}")
    print(" ÉVALUATION FINALE DE TOUS LES MODÈLES")
    print(f"{'='*60}")

    resultats = []
    for modele in modeles:
        poids = os.path.join(DOSSIER_SAUVEGARDE, modele['nom'], 'weights', 'best.pt')
        print(f"\nValidation : {modele['label']}...")
        m = YOLO(poids)
        val = m.val(data=CHEMIN_DATASET_YAML, split='val')
        p, r, f1, map50, ap_classes = extraire_metriques(val)
        resultats.append({
            'label': modele['label'],
            'p': p, 'r': r, 'f1': f1,
            'map50': map50,
            'ap_classes': ap_classes
        })

    # RAPPORT FINAL GLOBAL
    print("\n" + "=" * 75)
    print(" RAPPORT DE COMPARAISON FINAL — ABLATION STUDY")
    print("=" * 75)
    print(f"{'Modèle':<30} | {'Précision':>9} | {'Recall':>6} | {'F1':>6} | {'mAP@50':>6}")
    print("-" * 75)
    for res in resultats:
        print(f"{res['label']:<30} | {res['p']:>9.4f} | {res['r']:>6.4f} | {res['f1']:>6.4f} | {res['map50']:>6.4f}")
    print("=" * 75)

    # RAPPORT PAR CLASSE
    print("\n--- SCORES mAP@50 PAR CLASSE ---")
    header = f"{'Classe':<15}"
    for res in resultats:
        header += f" | {res['label'][:18]:>18}"
    print(header)
    print("-" * (15 + 22 * len(resultats)))

    for i, nom_classe in enumerate(NOMS_CLASSES):
        ligne = f"{nom_classe:<15}"
        for res in resultats:
            score = res['ap_classes'][i] if i < len(res['ap_classes']) else 0.0
            ligne += f" | {score:>18.4f}"
        print(ligne)

    print("\n" + "=" * 75)
    print(" MATRICES DE CONFUSION :")
    for modele in modeles:
        path = os.path.join(DOSSIER_SAUVEGARDE, modele['nom'], 'confusion_matrix.png')
        print(f"- {modele['label']:<30} : {path}")
    print("=" * 75)
    print("Fin du processus. Tous les résultats sont dans :", DOSSIER_SAUVEGARDE)


if __name__ == "__main__":
    run_pipeline()
