import time
import logging
import csv
from datetime import datetime
from super_gradients.training import models, Trainer
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

# Configuración del logger
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Lista de clases del dataset
CLASSES = ['aquila', 'bootes', 'canis_major', 'canis_minor', 'cassiopeia', 'cygnus',
           'gemini', 'leo', 'lyra', 'moon', 'orion', 'pleiades', 'sagittarius',
           'scorpius', 'taurus', 'ursa_major']

def main():
    assert CLASSES, "La lista de clases está vacía."

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": "C:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.yoloNAS",
            "images_dir": "train/images",
            "labels_dir": "train/labels",
            "classes": CLASSES
        },
        dataloader_params={"batch_size": 8, "num_workers": 2}
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": "C:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.yoloNAS",
            "images_dir": "valid/images",
            "labels_dir": "valid/labels",
            "classes": CLASSES
        },
        dataloader_params={"batch_size": 8, "num_workers": 2}
    )

    model = models.get(
        "yolo_nas_s",
        num_classes=len(CLASSES),
        pretrained_weights="coco",
    )

    assert train_data is not None, "El dataloader de entrenamiento no se cargó correctamente."
    assert val_data is not None, "El dataloader de validación no se cargó correctamente."
    assert model is not None, "El modelo no se cargó correctamente."

    pp_callback = PPYoloEPostPredictionCallback(
        score_threshold=0.05,
        nms_top_k=300,
        max_predictions=100,
        nms_threshold=0.6
    )

    valid_metrics = [
        DetectionMetrics_050(
            score_thres=0.3,
            top_k_predictions=300,
            num_cls=len(CLASSES),
            normalize_targets=True,
            post_prediction_callback=pp_callback
        ),
        DetectionMetrics_050_095(
            score_thres=0.3,
            top_k_predictions=300,
            num_cls=len(CLASSES),
            normalize_targets=True,
            post_prediction_callback=pp_callback
        )
    ]

    trainer = Trainer(
        experiment_name="mi_yolo_nas_local",
        ckpt_root_dir="checkpoints"
    )

    start_time = time.time()

    training_result = trainer.train(
        model=model,
        training_params={
            "max_epochs": 20,
            "lr_mode": "cosine",
            "initial_lr": 1e-4,
            "batch_accumulate": 1,
            "loss": "PPYoloELoss", # Función de pérdida usada por YOLO-NAS
            "criterion_params": {
                "num_classes": len(CLASSES)
            },
            "metric_to_watch": "PPYoloELoss/loss",
            "greater_metric_to_watch_is_better": True,
            "average_best_models": False
        },
        train_loader=train_data,
        valid_loader=val_data
    )

    training_time = round(time.time() - start_time, 2)
    logging.info(f"Tiempo total de entrenamiento: {training_time} segundos")

    test_results = trainer.test(
        model=model,
        test_loader=val_data,
        test_metrics_list=valid_metrics
    )

    precision = test_results.get('Precision@0.50', None)
    recall = test_results.get('Recall@0.50', None)
    map50 = test_results.get('mAP@0.50', None)
    map50_95 = test_results.get('mAP@0.50:0.95', None)

    csv_filename = "checkpoints/mi_yolo_nas_local/resultados_entrenamiento.csv"
    modelo_nombre = "YOLO-NAS-S-ajustado"

    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer.writerow([
            run_id,
            modelo_nombre,
            precision,
            recall,
            map50,
            map50_95,
            training_time
        ])

    logging.info(f"Resultados guardados en {csv_filename}")

if __name__ == "__main__":
    main()