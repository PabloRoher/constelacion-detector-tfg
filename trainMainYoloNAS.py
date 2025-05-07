import time
from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)

# Lista de clases del dataset
CLASSES = ['aquila', 'bootes', 'canis_major', 'canis_minor', 'cassiopeia', 'cygnus',
           'gemini', 'leo', 'lyra', 'moon', 'orion', 'pleiades', 'sagittarius',
           'scorpius', 'taurus', 'ursa_major']

def main():

    # Cargamos el dataloader de entrenamiento en formato YOLO
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": "C:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.yoloNAS",
            "images_dir": "train/images",
            "labels_dir": "train/labels",
            "classes": CLASSES
        },
        dataloader_params={
            "batch_size": 8,
            "num_workers": 2
        }
    )

    # Cargamos el dataloader de validación en formato YOLO
    val_data = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": "C:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.yoloNAS",
            "images_dir": "valid/images",
            "labels_dir": "valid/labels",
            "classes": CLASSES
        },
        dataloader_params={
            "batch_size": 8,
            "num_workers": 2
        }
    )

    model = models.get(
        "yolo_nas_s",
        num_classes=len(CLASSES),

        # Solo necesario si vas a reanudar un entrenamiento. Si se empieza desde cero, se puede omitir esta línea o poner checkpoint_path=None
        checkpoint_path="checkpoints/mi_yolo_nas_local/RUN_20250505_101316_646756/ckpt_latest.pth" # Cargamos los pesos desde un checkpoint
    )

    trainer = Trainer(
        experiment_name="mi_yolo_nas_local", # Define la carpeta donde guarda los checkpoints
        ckpt_root_dir="checkpoints"
    )

    trainer.train(
        model=model,
        training_params={
            "max_epochs": 51,
            "lr_mode": "cosine",
            "initial_lr": 1e-4,
            "batch_accumulate": 1,
            "loss": "PPYoloELoss", # Función de pérdida usada por YOLO-NAS
            "criterion_params": {
                "num_classes": len(CLASSES) # Pasamos el número de clases al criterio de pérdida
            },
            "metric_to_watch": "PPYoloELoss/loss", # Métrica que el entrenador usa para guardar el mejor modelo
            "greater_metric_to_watch_is_better": False, # Queremos minimizar la pérdida
            "average_best_models": False, # No promediar los mejores modelos (Poner True parece tener incompatibilidades en esta version de super gradients)

            # Solo necesario si vas a reanudar un entrenamiento. Si se empieza desde cero omite estas dos lineas
            "resume": True,
            "resume_path": "checkpoints/mi_yolo_nas_local/RUN_20250505_101316_646756/ckpt_latest.pth"
        },
        train_loader=train_data,
        valid_loader=val_data
    )

if __name__ == "__main__":
    main()