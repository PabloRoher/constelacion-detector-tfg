import tensorflow as tf

# Cambia esta ruta al archivo TFRecord que quieras analizar
tfrecord_path = "C:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.coco/tfrecords/train-00007-of-00008.tfrecord"

# Features esperados
feature_description = {
    "image/height": tf.io.FixedLenFeature([], tf.int64),
    "image/width": tf.io.FixedLenFeature([], tf.int64),
    "image/filename": tf.io.FixedLenFeature([], tf.string),
    "image/source_id": tf.io.FixedLenFeature([], tf.string),
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/format": tf.io.FixedLenFeature([], tf.string),
    "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    "image/object/class/text": tf.io.VarLenFeature(tf.string),
    "image/object/class/label": tf.io.VarLenFeature(tf.int64),
}

def parse_example(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

dataset = tf.data.TFRecordDataset([tfrecord_path])
parsed_dataset = dataset.map(parse_example)

total_images = 0
total_boxes = 0
labels_set = set()
images_with_zero_boxes = 0
box_stats = []

for example in parsed_dataset.take(20):
    total_images += 1
    xmin = tf.sparse.to_dense(example['image/object/bbox/xmin']).numpy()
    xmax = tf.sparse.to_dense(example['image/object/bbox/xmax']).numpy()
    ymin = tf.sparse.to_dense(example['image/object/bbox/ymin']).numpy()
    ymax = tf.sparse.to_dense(example['image/object/bbox/ymax']).numpy()
    labels = tf.sparse.to_dense(example['image/object/class/label']).numpy()
    n_boxes = len(xmin)
    total_boxes += n_boxes
    if n_boxes == 0:
        images_with_zero_boxes += 1
    else:
        box_stats.append((xmin, xmax, ymin, ymax))
        labels_set.update(labels.tolist())
    # Imprime info de la primera imagen
    if total_images == 1:
        print("Primer ejemplo:")
        print("  filename:", example['image/filename'].numpy())
        print("  labels:", labels)
        print("  xmin:", xmin)
        print("  xmax:", xmax)
        print("  ymin:", ymin)
        print("  ymax:", ymax)

# Estadísticas
if box_stats:
    all_xmin = [x for stats in box_stats for x in stats[0]]
    all_xmax = [x for stats in box_stats for x in stats[1]]
    all_ymin = [y for stats in box_stats for y in stats[2]]
    all_ymax = [y for stats in box_stats for y in stats[3]]
else:
    all_xmin = all_xmax = all_ymin = all_ymax = []

print("\nRESUMEN:")
print("Total imágenes analizadas:", total_images)
print("Total bounding boxes:", total_boxes)
print("Imágenes con cero boxes:", images_with_zero_boxes)
print("Clases encontradas:", sorted(labels_set))
print("xmin: min =", min(all_xmin) if all_xmin else None, "max =", max(all_xmin) if all_xmin else None)
print("xmax: min =", min(all_xmax) if all_xmax else None, "max =", max(all_xmax) if all_xmax else None)
print("ymin: min =", min(all_ymin) if all_ymin else None, "max =", max(all_ymin) if all_ymin else None)
print("ymax: min =", min(all_ymax) if all_ymax else None, "max =", max(all_ymax) if all_ymax else None)