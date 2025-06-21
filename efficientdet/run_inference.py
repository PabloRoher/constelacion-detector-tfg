import sys
sys.path.append('/home/pablo/constelacion-detector-tfg/efficientdet/automl/efficientdet')

import tensorflow.compat.v1 as tf
from PIL import Image
import numpy as np
import os
import re

# Importamos los módulos necesarios de la librería
from hparams_config import get_detection_config
from inference import build_model, build_inputs, det_post_process, visualize_image_prediction

MODEL_NAME = 'efficientdet-d0'
CHECKPOINT_PATH = 'automl/model_dir/model.ckpt-10256'
INPUT_IMAGE_PATH = '/mnt/c/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.coco/test/2022-12-03-00-00-00-s_png_jpg.rf.caa3e459dc759fde474eeae3257c0bf5.jpg'
OUTPUT_DIR = '/mnt/c/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.coco/resultados_prediccion/'
NUM_CLASES = 16

LABEL_MAP_PATH = '/mnt/c/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.coco/label_map.pbtxt'

def load_label_map(path):
    """Carga el mapa de etiquetas desde un archivo .pbtxt."""
    category_index = {}
    with open(path, 'r') as f:
        content = f.read()
        items = re.findall(r'item\s?{\s*id:\s*(\d+)\s*name:\s*"(.*?)"', content, re.DOTALL)
        for item_id, name in items:
            category_index[int(item_id)] = name
    print(f"Mapa de etiquetas cargado: {category_index}")
    return category_index

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Configurando el modelo...")
    params = get_detection_config(MODEL_NAME).as_dict()
    params.update({'num_classes': NUM_CLASES})

    with tf.Graph().as_default():
        # 1. Preparar la entrada
        raw_images, images, scales = build_inputs(
            INPUT_IMAGE_PATH, params['image_size'], params['mean_rgb'], params['stddev_rgb']
        )

        # 2. Construir el modelo y el post-procesamiento
        class_outputs, box_outputs = build_model(MODEL_NAME, images, **params)
        detections_batch = det_post_process(params, class_outputs, box_outputs, scales)

        # 3. Preparar la carga de pesos
        vars_to_restore = {}
        for var in tf.global_variables():
            if 'ExponentialMovingAverage' not in var.name and 'Momentum' not in var.name:
                var_name_in_ckpt = var.name.split(':')[0]
                vars_to_restore[var_name_in_ckpt] = var

        saver = tf.train.Saver(vars_to_restore)

        with tf.Session() as sess:
            # 4. Restaurar pesos y hacer la predicción
            print(f"Restaurando checkpoint: {CHECKPOINT_PATH}")
            saver.restore(sess, CHECKPOINT_PATH)

            print("Realizando la predicción...")
            predictions = sess.run(detections_batch)

            # 5. Visualizar el resultado
            print("Generando imagen con detecciones...")

            label_map = load_label_map(LABEL_MAP_PATH)

            predictions[0][:, 6] += 1

            img = visualize_image_prediction(
                raw_images[0],
                predictions[0],
                label_map=label_map, # Le pasamos un mapa personalizado para nuestro caso
                min_score_thresh=0.2
            )

            output_image_path = os.path.join(OUTPUT_DIR, 'prediccion_test2.jpg')
            Image.fromarray(img).save(output_image_path)

            print(f"\n¡ÉXITO! La imagen ha sido guardada en: {output_image_path}")

if __name__ == '__main__':
    tf.disable_eager_execution()
    main()
