# Basado en main.py del repo oficial EfficientDet (Google Research, Apache 2.0)
# Cambios:
#   - Rutas y parámetros adaptados a dataset de constelaciones
#   - Número de clases fijado a 16
#   - Epochs, batch size y modelo predefinidos
#   - Simplificado para entrenamiento local
# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The main training script."""
import sys
import os
import multiprocessing
import time
import csv
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

# Ruta para importar los módulos de efficientdet
sys.path.append(os.path.join(os.path.dirname(__file__), 'automl', 'efficientdet'))

import dataloader
import det_model_fn
import hparams_config
import utils

FLAGS = flags.FLAGS

# Ruta donde se guardarán las métricas COCO
COCO_CSV_PATH = "/home/pablo/constelacion-detector-tfg/efficientdet/metrics_coco.csv"

def save_coco_metrics_to_csv(csv_path, model_name, metrics, training_time=0):
    # Crear CSV si no existe
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Modelo", "mAP@[.5:.95]", "mAP@0.5", "mAP@0.75", "Tiempo Entrenamiento (s)"
            ])

    mAP   = metrics.get('AP', 0)
    mAP50 = metrics.get('AP50', 0)
    mAP75 = metrics.get('AP75', 0)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            model_name,
            round(mAP, 6), round(mAP50, 6), round(mAP75, 6),
            round(training_time, 2)
        ])

# Definición de flags necesarios
flags.DEFINE_string('train_file_pattern', None, 'Glob for training data files')
flags.DEFINE_string('val_file_pattern', None, 'Glob for validation data files')
flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string('model_name', None, 'Model name')
flags.DEFINE_string('hparams', '', 'Hyperparameters string')
flags.DEFINE_integer('num_epochs', 50, 'Number of training epochs')
flags.DEFINE_integer('train_batch_size', 8, 'Training batch size')
flags.DEFINE_bool('use_fake_data', False, 'Use fake input')
flags.DEFINE_string('strategy', None, 'Training strategy')
flags.DEFINE_integer('eval_samples', 0, 'Number of samples for eval')
flags.DEFINE_integer('eval_batch_size', 1, 'Batch size for eval')
flags.DEFINE_integer('num_examples_per_epoch', 1641, 'Num examples per epoch')
flags.DEFINE_integer('iterations_per_loop', 1000, '')
flags.DEFINE_integer('save_checkpoints_steps', 1000, '')
flags.DEFINE_string('mode', 'train', 'Mode: train, eval o train_and_eval')

def main(_):
    # rutas y parámetros
    TRAIN_RECORD = "/mnt/c/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.coco/tfrecords/train-*-of-*.tfrecord"
    VAL_RECORD   = "/mnt/c/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.coco/tfrecords/valid-*-of-*.tfrecord"
    TEST_RECORD  = "/mnt/c/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.coco/tfrecords/test-*-of-*.tfrecord"
    MODEL_DIR    = "/home/pablo/constelacion-detector-tfg/efficientdet/automl/model_dir"
    MODEL_NAME   = "efficientdet-d0"
    NUM_CLASSES  = 16
    EPOCHS       = 50
    BATCH_SIZE   = 8

    flags.FLAGS.train_file_pattern = TRAIN_RECORD
    flags.FLAGS.val_file_pattern   = VAL_RECORD
    flags.FLAGS.model_dir          = MODEL_DIR
    flags.FLAGS.model_name         = MODEL_NAME
    flags.FLAGS.hparams            = f"num_classes={NUM_CLASSES}"
    flags.FLAGS.num_epochs         = EPOCHS
    flags.FLAGS.train_batch_size   = BATCH_SIZE

    config = hparams_config.get_detection_config(FLAGS.model_name)
    config.override(FLAGS.hparams)
    if FLAGS.num_epochs:
        config.num_epochs = FLAGS.num_epochs
    config.image_size = utils.parse_image_size(config.image_size)
    max_instances_per_image = config.max_instances_per_image

    # pasos por epoch y totales
    if FLAGS.eval_samples:
        eval_steps = int((FLAGS.eval_samples + FLAGS.eval_batch_size - 1) // FLAGS.eval_batch_size)
    else:
        eval_steps = None
    total_examples = int(config.num_epochs * FLAGS.num_examples_per_epoch)
    train_steps = total_examples // FLAGS.train_batch_size

    if not tf.io.gfile.exists(MODEL_DIR):
        tf.io.gfile.makedirs(MODEL_DIR)

    # entrada train/val/test
    train_input_fn = dataloader.InputReader(
        FLAGS.train_file_pattern,
        is_training=True,
        use_fake_data=FLAGS.use_fake_data,
        max_instances_per_image=max_instances_per_image)
    eval_input_fn = dataloader.InputReader(
        FLAGS.val_file_pattern,
        is_training=False,
        use_fake_data=FLAGS.use_fake_data,
        max_instances_per_image=max_instances_per_image)
    test_input_fn = dataloader.InputReader(
        TEST_RECORD,
        is_training=False,
        use_fake_data=FLAGS.use_fake_data,
        max_instances_per_image=max_instances_per_image)

    # instancia el modelo
    model_fn_instance = det_model_fn.get_model_fn(FLAGS.model_name)
    strategy = None
    if FLAGS.strategy == 'gpus':
        strategy = tf.distribute.MirroredStrategy()
    run_config = tf.estimator.RunConfig(
        model_dir=MODEL_DIR,
        train_distribute=strategy,
        log_step_count_steps=FLAGS.iterations_per_loop,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps if hasattr(FLAGS, "save_checkpoints_steps") else 1000,
    )

    def get_estimator(global_batch_size):
        n_shards = getattr(strategy, 'num_replicas_in_sync', 1) if strategy else 1
        params = dict(config.as_dict())
        params['num_shards'] = n_shards
        params['batch_size'] = global_batch_size // n_shards
        params['model_name'] = FLAGS.model_name
        params['num_examples_per_epoch'] = FLAGS.num_examples_per_epoch
        params['val_json_file'] = '/mnt/c/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.coco/valid/_annotations.coco.json'
        return tf.estimator.Estimator(
            model_fn=model_fn_instance, config=run_config, params=params)

    train_est = get_estimator(FLAGS.train_batch_size)
    eval_est  = get_estimator(FLAGS.eval_batch_size if hasattr(FLAGS, "eval_batch_size") else 1)

     # Entrenamiento completo + evaluación final en valid y test
    if FLAGS.mode == 'train':
        start_time = time.time()
        train_est.train(input_fn=train_input_fn, max_steps=train_steps)
        training_time = time.time() - start_time

        # Evalúa en valid y guarda métricas
        eval_results = eval_est.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        save_coco_metrics_to_csv(COCO_CSV_PATH, MODEL_NAME + "_valid", eval_results, training_time)
        # Evalúa en test y guarda métricas
        test_results = eval_est.evaluate(input_fn=test_input_fn)
        save_coco_metrics_to_csv(COCO_CSV_PATH, MODEL_NAME + "_test", test_results, training_time)

    # Solo evaluación (sin entrenamiento)
    elif FLAGS.mode == 'eval':
        eval_results = eval_est.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        save_coco_metrics_to_csv(COCO_CSV_PATH, MODEL_NAME + "_valid", eval_results)
        test_results = eval_est.evaluate(input_fn=test_input_fn)
        save_coco_metrics_to_csv(COCO_CSV_PATH, MODEL_NAME + "_test", test_results)

    else:
        logging.info('Invalid mode: %s', FLAGS.mode)
if __name__ == '__main__':
    app.run(main)