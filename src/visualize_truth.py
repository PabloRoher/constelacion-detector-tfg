import cv2
import os

def draw_ground_truth(image_path, label_path, class_names):
    """
    Dibuja las cajas ground truth en una imagen.
    """
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return
    h, w, _ = image.shape

    # Leer el archivo de etiquetas
    if not os.path.exists(label_path):
        print(f"Advertencia: No se encontró el archivo de etiquetas {label_path}")
        return

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, box_w, box_h = map(float, parts[1:])

            # Des-normalizar las coordenadas a píxeles
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            # Dibujar el rectángulo
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Color verde, grosor 2

            # Escribir el nombre de la clase
            label = class_names.get(class_id, "Desconocido")
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    return image


if __name__ == '__main__':

    class_names = {
        0: 'aquila', 1: 'bootes', 2: 'canis_major', 3: 'canis_minor',
        4: 'cassiopeia', 5: 'cygnus', 6: 'gemini', 7: 'leo', 8: 'lyra',
        9: 'moon', 10: 'orion', 11: 'pleiades', 12: 'sagittarius',
        13: 'scorpius', 14: 'taurus', 15: 'ursa_major'
    }

    # Elige la imagen que quieres visualizar
    image_filename = 'messier_m45_010_png_jpg.rf.44a4f776c715fedfaf3b04ed24279e7c.jpg'
    
    # Rutas a las carpetas de imágenes y etiquetas de prueba
    test_image_dir = 'C:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.yolov11/test/images'
    test_label_dir = 'C:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.yolov11/test/labels'
    
    image_path = os.path.join(test_image_dir, image_filename)
    label_path = os.path.join(test_label_dir, os.path.splitext(image_filename)[0] + '.txt')

    ground_truth_image = draw_ground_truth(image_path, label_path, class_names)

    if ground_truth_image is not None:
        # Guardar la imagen resultante
        output_filename = 'ground_truth_example.png'
        cv2.imwrite(output_filename, ground_truth_image)
        print(f"Imagen de verdad terreno guardada como: {output_filename}")
        
        # Opcional: mostrar la imagen en una ventana
        cv2.imshow('Ground Truth', ground_truth_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()