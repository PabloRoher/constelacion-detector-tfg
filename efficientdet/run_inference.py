# Este script ejecuta el script official de inference.py de EfficientDet para realizar inferencias en un conjunto de imágenes.
from inference import InferenceDriver

MODEL_NAME = "efficientdet-d0"
CKPT_PATH = "/home/pablo/constelacion-detector-tfg/efficientdet/automl/model_dir" # Path al directorio del modelo entrenado
IMAGE_PATH_PATTERN = "/mnt/c/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.coco/test/*.jpg" #Path a las imágenes de prueba
OUTPUT_DIR = "/mnt/c/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.coco" #Path al directorio donde se guardarán los resultados de la inferencia
MIN_SCORE_THRESH = 0.15 # Umbral mínimo de puntuación para considerar una detección válida

if __name__ == "__main__":
    driver = InferenceDriver(MODEL_NAME, CKPT_PATH)
    driver.inference(
        IMAGE_PATH_PATTERN,
        OUTPUT_DIR,
        min_score_thresh=MIN_SCORE_THRESH
    )
