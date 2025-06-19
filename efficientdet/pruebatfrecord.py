import os
import glob
import tensorflow as tf

# Path al directorio donde se encuentran los archivos TFRecord
TFRECORDS_DIR = r"C:\Users\pablo\Desktop\TFG\Datasets\Constellation.v1i.coco\tfrecords"

# Path al archivo donde se guardará el resumen
RESUMEN_PATH = r"C:\Users\pablo\Desktop\TFG\Codigo\TFG1\efficientdet\Resumen_tfrecord.txt"

def parse_example(example_proto):
    features = {
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    return tf.io.parse_single_example(example_proto, features)

def resumen_tfrecord(path, num_samples=20):
    resumen = []
    resumen.append(f"### Resumen de: {os.path.basename(path)} ###")
    raw_dataset = tf.data.TFRecordDataset(path)
    parsed_dataset = raw_dataset.map(parse_example)
    total_images = 0
    total_boxes = 0
    clases = set()
    imagenes_con_cero_boxes = 0
    xmin_all, xmax_all, ymin_all, ymax_all = [], [], [], []
    primer_ejemplo = None

    for parsed in parsed_dataset.take(num_samples):
        total_images += 1
        labels = tf.sparse.to_dense(parsed['image/object/class/label']).numpy()
        xmins = tf.sparse.to_dense(parsed['image/object/bbox/xmin']).numpy()
        xmaxs = tf.sparse.to_dense(parsed['image/object/bbox/xmax']).numpy()
        ymins = tf.sparse.to_dense(parsed['image/object/bbox/ymin']).numpy()
        ymaxs = tf.sparse.to_dense(parsed['image/object/bbox/ymax']).numpy()

        n_boxes = len(labels)
        total_boxes += n_boxes
        if n_boxes == 0:
            imagenes_con_cero_boxes += 1
        else:
            clases.update(labels)
            xmin_all.extend(xmins)
            xmax_all.extend(xmaxs)
            ymin_all.extend(ymins)
            ymax_all.extend(ymaxs)

        if primer_ejemplo is None:
            primer_ejemplo = (
                f"Primer ejemplo:\n"
                f"  filename: {parsed['image/filename'].numpy()}\n"
                f"  labels: {labels}\n"
                f"  xmin: {xmins}\n"
                f"  xmax: {xmaxs}\n"
                f"  ymin: {ymins}\n"
                f"  ymax: {ymaxs}\n"
            )

    resumen.append(primer_ejemplo if primer_ejemplo else "No se encontró ningún ejemplo válido.\n")
    resumen.append(f"RESUMEN:\nTotal imágenes analizadas: {total_images}")
    resumen.append(f"Total bounding boxes: {total_boxes}")
    resumen.append(f"Imágenes con cero boxes: {imagenes_con_cero_boxes}")
    resumen.append(f"Clases encontradas: {sorted(clases) if clases else 'Ninguna'}")
    resumen.append(f"xmin: min = {min(xmin_all) if xmin_all else 'N/A'} max = {max(xmin_all) if xmin_all else 'N/A'}")
    resumen.append(f"xmax: min = {min(xmax_all) if xmax_all else 'N/A'} max = {max(xmax_all) if xmax_all else 'N/A'}")
    resumen.append(f"ymin: min = {min(ymin_all) if ymin_all else 'N/A'} max = {max(ymin_all) if ymin_all else 'N/A'}")
    resumen.append(f"ymax: min = {min(ymax_all) if ymax_all else 'N/A'} max = {max(ymax_all) if ymax_all else 'N/A'}")
    resumen.append("-" * 50 + "\n")
    return "\n".join(resumen)

def main():
    resumen_total = []
    tfrecord_files = glob.glob(os.path.join(TFRECORDS_DIR, "*.tfrecord"))

    for tfrec in sorted(tfrecord_files):
        print(f"Procesando {tfrec} ...")
        resumen = resumen_tfrecord(tfrec)
        resumen_total.append(resumen)

    # Guardar el archivo
    with open(RESUMEN_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(resumen_total))

    print(f"Resúmenes guardados en '{RESUMEN_PATH}'.")

if __name__ == "__main__":
    main()