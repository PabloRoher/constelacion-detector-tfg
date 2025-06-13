import os
import time
import csv
import logging
from ultralytics import YOLO

# Configuración del logger
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Lista de modelos que quieres entrenar
    modelos = [
        "yolo11n.pt",
        "yolo11m.pt",
    ]

    # CSV donde guardaremos los resultados
    csv_filename = "resultados_entrenamiento.csv"
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Cabecera del CSV
        writer.writerow(["Modelo", "Precision", "Recall", "mAP50", "mAP50-95", "Tiempo Entrenamiento (s)"])

    # Ruta al dataset
    dataset_path = "C:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.yolov11/data.yaml"

    for modelo_nombre in modelos:
        print(f"Entrenando modelo: {modelo_nombre}")

        try:
            # Cargar modelo
            model = YOLO(modelo_nombre)
        except Exception as e:
            logging.warning(f"Error al cargar el modelo {modelo_nombre}: {e}")
            continue

        # Medir tiempo al empezar
        start_time = time.time() 

    
        try:
            # Entrenar el modelo con un dataset personalizado
            train_results = model.train(
                data=dataset_path,  
                epochs=100,
                imgsz=640,
                device="cuda",
                val=True,           # Asegura que haga validación durante el entrenamiento
                save=True,          # Guarda los mejores modelos
                save_period=10,     # Guarda cada 10 epochs
                patience=20,         # Early stopping si no mejora tras 20 epochs
                name=modelo_nombre.replace(".pt", "")  # Crea una carpeta específica para cada modelo
            )
        except Exception as e:
            logging.warning(f"Error durante el entrenamiento del modelo {modelo_nombre}: {e}")
            continue

        try:
            # Evaluar el modelo en el conjunto de validación
            metrics = model.val()
        except Exception as e:
            logging.warning(f"Error durante la validación del modelo {modelo_nombre}: {e}")
            continue

        # Realizar detección de objetos en una imagen
        #results = model("path/to/image.jpg")
        #results[0].show()  # Mostrar los resultados

        # Después de train y val
        training_time = time.time() - start_time

        try:
            model.export(format="onnx")         # Muy compatible con APIs Python, C++, etc.
            model.export(format="torchscript")  # Rápido y portable en entornos PyTorch
            model.export(format="openvino")        # Para Intel o edge devices
        except Exception as e:
            logging.warning(f"Error al exportar el modelo {modelo_nombre}: {e}")

        try:
            # Guardar resultados en el CSV
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    modelo_nombre,
                    metrics.box.map,        # mAP@0.5 (mean average precision a IoU=0.5)
                    metrics.box.map50,      
                    metrics.box.map75,      
                    metrics.box.map,        
                    round(training_time, 2) # Tiempo de entrenamiento en segundos
                ])
        except Exception as e:
            logging.warning(f"Error al guardar los resultados del modelo {modelo_nombre} en el CSV: {e}")

        print(f"Modelo {modelo_nombre} terminado.\n")

if __name__ == "__main__":
    main()