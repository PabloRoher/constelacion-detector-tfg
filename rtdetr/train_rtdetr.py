import time
import csv
import os
from ultralytics import YOLO

def main():
    # Modelo a Entrenar
    modelo_nombre = 'rtdetr-l.pt'

    # Path al dataset
    dataset_path = '/home/pablo/datasets/constellation_yolov11/data.yaml'

    # Hiperparámetros de Entrenamiento
    EPOCHS = 50
    IMAGE_SIZE = 640
    BATCH_SIZE = 8

    PROJECT_NAME = 'RTDETR_Constellations'
    csv_filename = "resultados_entrenamiento.csv"

    # Crear el archivo CSV y escribir la cabecera si no existe
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Modelo", "mAP50-95", "mAP50", "mAP75", "Tiempo Entrenamiento (s)"])

    # Cargar el modelo base pre-entrenado
    model = YOLO(modelo_nombre)

    # Medir el tiempo de inicio del entrenamiento
    start_time = time.time()

    model.train(
        data=dataset_path,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
    )

    # Calcular el tiempo total de entrenamiento
    training_time = time.time() - start_time

    print(f"\n--- Entrenamiento para {modelo_nombre} finalizado en {training_time:.2f} segundos. ---")
    print("--- Iniciando validación final en el conjunto 'val' ---")

    # Validar el mejor modelo guardado
    metrics = model.val()

    # Guardar resultados en el CSV
    try:
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                modelo_nombre,
                round(metrics.box.map, 6),
                round(metrics.box.map50, 6),
                round(metrics.box.map75, 6),
                round(training_time, 2)
            ])
        print(f"Resultados de {modelo_nombre} guardados exitosamente en {csv_filename}")
    except Exception as e:
        print(f"Error al guardar los resultados del modelo {modelo_nombre} en el CSV: {e}")

    print(f"\n--- Proceso para {modelo_nombre} completado. ---")


if __name__ == "__main__":
    main()
