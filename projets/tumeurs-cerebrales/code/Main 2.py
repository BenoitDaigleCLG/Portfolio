import os
import yaml
from ultralytics import YOLO
from architecture_custom import create_architecture

def configurer_dataset_yaml(dataset_path):

    #On entre dans le dossier DonneesRoboFlow et on lit le data.yaml
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    #On s'en va renommer le path avec le bon chemin des images
    chemin_absolu = os.path.abspath(dataset_path)
    data['path'] = chemin_absolu

    #On va mettre les bons paths pour les donnees
    data['train'] = 'train/images'
    data['val'] = 'valid/images'
    data['test'] = 'test/images'

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)

    return yaml_path


def main():
    create_architecture()

    dataset_path = "Donnees_RoboFlow"

    yaml_path = configurer_dataset_yaml(dataset_path)

    #On utilise l'architecture custom, mais avec les poids de départ du modèle pré-entrainé
    model = YOLO('custom_yolo11s.yaml')
    model.load('yolo11s.pt')

    #Lancer l'entraînement
    print("\nDébut de l'entraînement")
    results = model.train(
        data=yaml_path,
        epochs=50,
        patience=20,
        imgsz=640,
        batch=8,
        workers=2,
        seed=42,
        optimizer='AdamW',
        lr0=0.001,
        close_mosaic=15,
        device=0
    )
    print("\nEntraînement terminé")

if __name__ == '__main__':
    main()
