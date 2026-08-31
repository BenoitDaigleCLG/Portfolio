import os
import pandas as pd
from ultralytics import YOLO

if __name__ == '__main__':

    dataset_path = 'Donnees_RoboFlow'
    yaml_path = os.path.join(dataset_path, 'data.yaml')

    run_name = 'train'
    chemin_poids = os.path.join('runs', 'detect', run_name, 'weights', 'best.pt')
    model = YOLO(chemin_poids)

    # Métriques sur données de validation
    metrics_val = model.val(data=yaml_path, split='val', verbose=False, conf=0.1)

    precision_val = metrics_val.box.mp
    recall_val = metrics_val.box.mr
    map50_val = metrics_val.box.map50
    f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)

    # Métriques sur données de test
    metrics_test = model.val(data=yaml_path, split='test', verbose=False, conf=0.1)

    precision_test = metrics_test.box.mp
    recall_test = metrics_test.box.mr
    map50_test = metrics_test.box.map50
    f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)

    # Métriques sur validation et test
    df_global = pd.DataFrame({
        "Dataset": ["Validation Set", "Test Set"],
        "F1-score": [f"{f1_val * 100:.1f}%", f"{f1_test * 100:.1f}%"],
        "Precision": [f"{precision_val * 100:.1f}%", f"{precision_test * 100:.1f}%"],
        "Recall": [f"{recall_val * 100:.1f}%", f"{recall_test * 100:.1f}%"],
        "mAP@50": [f"{map50_val * 100:.1f}%", f"{map50_test * 100:.1f}%"]
    })

    noms_classes = model.names
    classes_val = metrics_val.box.ap_class_index
    classes_test = metrics_test.box.ap_class_index

    # map50 sur toutes les classes
    lignes_classes = [{
        "Classe": "All (Global)",
        "Validation Set mAP@50": f"{map50_val * 100:.1f}%",
        "Test Set mAP@50": f"{map50_test * 100:.1f}%"
    }]

    # On passe sur toutes les classes
    for classe in range(len(noms_classes)):
        nom = noms_classes[classe]

        # Pour chaque classe, on va chercher map50 de validation
        if classe in classes_val:
            pos_val = list(classes_val).index(classe)
            score_val = f"{metrics_val.box.ap50[pos_val] * 100:.1f}%"

        # Pour chaque classe, on va chercher map50 de test
        if classe in classes_test:
            pos_test = list(classes_test).index(classe)
            score_test = f"{metrics_test.box.ap50[pos_test] * 100:.1f}%"

        # On ajoute à lignes_classe pour une avoir notre liste de résultats sur toutes les classes
        lignes_classes.append({
            "Classe": nom,
            "Validation Set mAP@50": score_val,
            "Test Set mAP@50": score_test
        })

    df_classes = pd.DataFrame(lignes_classes)

    # On affiche
    resultats = (
        "\nRÉSULTATS GLOBAUX \n" +
        df_global.to_string(index=False) +
        "\n\n RÉSULTATS PAR CLASSE \n" +
        df_classes.to_string(index=False)
    )


    nom_fichier = "resultats_metrics.txt"
    with open(nom_fichier, "w", encoding="utf-8") as f:
        f.write(resultats)
