from ultralytics import YOLO
import os
from pathlib import Path

def generate_predictions(model_name, model_path, image_folder, output_base_dir):
    """
    Carga un modelo y realiza predicciones en una carpeta de imágenes,
    guardando los resultados visuales.
    """
    print("="*60)
    print(f"GENERANDO VISUALES PARA EL MODELO: {model_name}")
    print(f"Cargando pesos desde: {model_path}")
    print("="*60)

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo {model_name}. Verifica la ruta: {model_path}")
        print(e)
        return

    # Define un directorio de salida específico para este modelo
    output_dir = os.path.join(output_base_dir, model_name)
    print(f"Las imágenes con las detecciones se guardarán en: {output_dir}")

    # conf=0.5 es el umbral de confianza, solo mostrará detecciones con más del 50% de confianza
    results = model.predict(
        source=image_folder,
        save=True,
        conf=0.5,
        project=output_base_dir, # Directorio base para los resultados
        name=model_name          # Nombre de la carpeta específica del experimento
    )
    
    print(f"\nVisuales para {model_name} generados correctamente.")
    print("="*60 + "\n")


if __name__ == '__main__':

    PROJECT_ROOT = Path(__file__).resolve().parent.parent 

    MODELS_DIR = PROJECT_ROOT / "models"

    models_to_predict = {
        "YOLOv11n": MODELS_DIR / "yolov11_m_final.pt",
        "YOLOv11m": MODELS_DIR / "yolov11_N_final.pt",
        "RT-DETR-L": MODELS_DIR / "rtdetr_final.pt"
    }
    
    # Ruta a la carpeta que contiene las IMÁGENES de prueba
    test_images_folder = 'C:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.yolov11/test/images'

    # Carpeta donde se guardarán todos los resultados visuales
    output_directory = 'prediction_visuals'

    print("INICIO DE LA GENERACIÓN DE VISUALES\n")
    
    for name, path in models_to_predict.items():
        generate_predictions(name, path, test_images_folder, output_directory)
        
    print("GENERACIÓN DE VISUALES FINALIZADA")