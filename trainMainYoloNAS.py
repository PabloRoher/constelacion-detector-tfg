import time
import logging
from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)

# Configuración del logger
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Lista de clases del dataset
CLASSES = ['aquila', 'bootes', 'canis_major', 'canis_minor', 'cassiopeia', 'cygnus',
           'gemini', 'leo', 'lyra', 'moon', 'orion', 'pleiades', 'sagittarius',
           'scorpius', 'taurus', 'ursa_major']

def main():
    # Verificar si la lista de clases está vacía
    assert CLASSES, "La lista de clases está vacía."

    # Cargamos el dataloader de entrenamiento en formato YOLO
    try:
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
    except Exception as e:
        logging.warning(f"Error al cargar el dataloader de entrenamiento: {e}")
        train_data = None

    # Cargamos el dataloader de validación en formato YOLO
    try:
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
    except Exception as e:
        logging.warning(f"Error al cargar el dataloader de validación: {e}")
        val_data = None
    
    try:
        model = models.get(
            "yolo_nas_s",
            num_classes=len(CLASSES),

            # Solo necesario si vas a reanudar un entrenamiento. Si se empieza desde cero, se puede omitir esta línea o poner checkpoint_path=None
            checkpoint_path="checkpoints/mi_yolo_nas_local/RUN_20250505_101316_646756/ckpt_latest.pth" # Cargamos los pesos desde un checkpoint
        )
    except Exception as e:
        logging.warning(f"Error al cargar el modelo YOLO-NAS: {e}")
        model = None

    # Verificar que los objetos necesarios no sean None
    assert train_data is not None, "El dataloader de entrenamiento no se cargó correctamente."
    assert val_data is not None, "El dataloader de validación no se cargó correctamente."
    assert model is not None, "El modelo no se cargó correctamente."

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