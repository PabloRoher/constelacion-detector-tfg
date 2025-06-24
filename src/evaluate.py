from ultralytics import YOLO
from pathlib import Path

def evaluate_model(model_name, model_path, data_config):
    """
    Carga un modelo específico, lo evalúa en el conjunto de prueba
    y muestra sus resultados.
    """
    print("==========================================")
    print(f"EVALUANDO MODELO: {model_name}")
    print(f"Cargando pesos desde: {model_path}")
    print("==========================================")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo {model_name}. Verifica la ruta: {model_path}")
        print(e)
        return

    print(f"Iniciando evaluación de {model_name} en el conjunto de prueba (split='test')...")
    try:
        results = model.val(data=data_config, split='test', name=f'{model_name}_test_evaluation')
    except Exception as e:
        print(f"ERROR: Falló la evaluación para el modelo {model_name}.")
        print(e)
        return

    # Imprime los resultados
    print("\n" + "-"*20 + f" RESULTADOS FINALES PARA {model_name} ")
    print(f"  mAP@0.5:                  {results.box.map50:.4f}")
    print(f"  mAP@[.5:.95]:             {results.box.map:.4f}")
    print(f"  Precisión (Precision):      {results.box.p[0]:.4f}")
    print(f"  Recall:                   {results.box.r[0]:.4f}")
    print(f"  Latencia de Inferencia (ms): {results.speed['inference']:.2f} ms")
    print("\n--- Métricas por clase (AP@0.5) ---")
    
    ap50_per_class = results.box.ap50
    
    # Imprime el AP por cada clase
    for i, name in sorted(results.names.items()):
        # Accedemos al valor correspondiente usando el índice
        print(f"  - {name:<15}: {ap50_per_class[i]:.4f}")
        
    print("\n" + "==========================================" + "\n")


if __name__ == '__main__':

    PROJECT_ROOT = Path(__file__).resolve().parent.parent 
    MODELS_DIR = PROJECT_ROOT / "models"

    # Los modelos a evaluar y sus rutas
    models_to_evaluate = {
        "YOLOv11n": MODELS_DIR / "yolov11_m_final.pt",
        "YOLOv11m": MODELS_DIR / "yolov11_N_final.pt",
        "RT-DETR-L": MODELS_DIR / "rtdetr_final.pt"
    }

    # Ruta al fichero de configuración del dataset
    dataset_yaml = 'C:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.yolov11/data.yaml'

    print("INICIO DEL SCRIPT DE EVALUACIÓN FINAL\n")
    
    for name, path in models_to_evaluate.items():
        evaluate_model(name, path, dataset_yaml)
        
    print("SCRIPT DE EVALUACIÓN FINALIZADO")