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
from super_gradients.training.transforms import DetectionHSV, DetectionRandomAffine
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.losses import PPYoloELoss

import torch

class DebugPPYoloELoss(PPYoloELoss):
    def forward(self, predictions, targets):
        if targets.numel() > 0:
            target_class_ids = targets[:, 1]
            print(f"DEBUG: IDs de clase del dataset (raw) - Max: {target_class_ids.max().item()}, Min: {target_class_ids.min().item()}")
            print(f"DEBUG: IDs de clase del dataset (raw) - Únicos: {torch.unique(target_class_ids)}")
            print(f"DEBUG: Número de clases configurado para la pérdida (len(CLASSES)): {self.num_classes}")

        out = super().forward(predictions, targets)
        return out

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TUS CLASES
CLASSES = ['aquila', 'bootes', 'canis_major', 'canis_minor', 'cassiopeia', 'cygnus', 'gemini', 'leo', 'lyra', 'moon', 'orion', 'pleiades', 'sagittarius', 'scorpius', 'taurus', 'ursa_major']

def main():
    # TUS RUTAS AL DATASET
    train_root_dir = "c:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.yolov11/train"
    val_root_dir = "c:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.yolov11/valid"

    start_time = time.time()

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": train_root_dir,
            "images_dir": "images",
            "labels_dir": "labels",
        },
        dataloader_params={
            "batch_size": 16,
            "num_workers": 4
        }
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": val_root_dir,
            "images_dir": "images",
            "labels_dir": "labels",
        },
        dataloader_params={
            "batch_size": 16,
            "num_workers": 4
        }
    )

    model = models.get(
        "yolo_nas_s",
        num_classes=len(CLASSES),
        pretrained_weights="coco"
    )

    valid_metrics = [
        DetectionMetrics_050(
            num_cls=len(CLASSES), # <<<<< MODIFICADO AQUÍ A 'num_cls'
            post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.1, nms_threshold=0.7, nms_top_k=1000, max_predictions=300)
        ),
        DetectionMetrics_050_095(
            num_cls=len(CLASSES), # <<<<< MODIFICADO AQUÍ A 'num_cls'
            post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.1, nms_threshold=0.7, nms_top_k=1000, max_predictions=300)
        )
    ]

    trainer = Trainer(experiment_name="mi_yolo_nas_local", ckpt_root_dir="checkpoints")

    training_params = {
        "max_epochs": 100,
        "lr_updates": [30, 60, 90],
        "lr_decay_factor": 0.1,
        "initial_lr": 0.01,
        "loss": DebugPPYoloELoss(num_classes=len(CLASSES)),
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0005},
        "criterion_params": {},
        "train_metrics_list": [],
        "valid_metrics_list": valid_metrics,
        "loss_logging_items_names": ["loss", "lbox", "lcls", "lobj"],
        "metric_to_watch": "mAP@0.50:0.95",
        "greater_metric_to_watch_is_better": True,
        "average_best_models": False,
        "ema": True,
        "ema_params": {"decay": 0.9999, "beta_factor": 0.96},
        "early_stopping": False,
        "early_stopping_params": {"metric_name": "mAP@0.50:0.95", "patience": 20, "mode": "max"},
        "save_best_model": True,
        "save_ckpt_epoch_list": [],
        "resume": False,
        "lr_mode": "cosine",
        "batch_accumulate": 1,
    }

    training_result = trainer.train(
        model=model,
        training_params=training_params,
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

    try:
        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                csv_writer.writerow(["Fecha", "Modelo", "Epocas", "Tiempo de Entrenamiento (s)", "Precision@0.50", "Recall@0.50", "mAP@0.50", "mAP@0.50:0.95"])
            
            csv_writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                modelo_nombre,
                training_params["max_epochs"],
                training_time,
                precision,
                recall,
                map50,
                map50_95
            ])
        logging.info(f"Resultados guardados en {csv_filename}")
    except Exception as e:
        logging.error(f"Error al guardar resultados en CSV: {e}")

if __name__ == "__main__":
    main()