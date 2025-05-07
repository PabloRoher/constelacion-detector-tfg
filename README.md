# TFG - Detección de Constelaciones usando YOLOv11, YOLO-NAS y Modelo Propio

Este proyecto tiene como objetivo detectar constelaciones en imágenes del cielo utilizando redes de detección de objetos de última generación basadas en **YOLOv11**, **YOLO-NAS** y un **modelo propio personalizado**.

> Nota: En este repositorio se incluye una carpeta `/runs/detect/` que contiene únicamente resultados de entrenamientos de prueba iniciales, como ejemplos de ejecución (principalmente con el modelo YOLOv11n). Los mejores resultados de cada entrenamiento se encuentran organizados en el archivo llamado `resultados_entrenamiento`.

Este proyecto utiliza **tres entornos de trabajo**:

- **YOLOv11 / Ultralytics:** Entorno virtual `venv`
- **YOLO-NAS / Super-Gradients:** Entorno virtual `venv_nas`
- **Modelo Propio:**

### Activar el entorno YOLOv11 (`venv`)

cd ruta/del/proyecto
.\venv\Scripts\activate
python yolo11/trainMainYolov11.py

### Activar el entorno YOLO-NAS (`venv_nas`)

cd ruta/del/proyecto
.\venv_nas\Scripts\activate
python yolo_nas/train_yolo_nas.py

### Entrenar el Modelo Propio (`--`)

## Requisitos

Python 3.10+

### Para YOLOv11: Ultralytics

pip install ultralytics

### Para YOLO-NAS: Super-Gradients

pip install super-gradients==3.7.1

> Nota: Algunas dependencias como onnx y pycocotools pueden requerir que Visual C++ Build Tools esté instalado en el sistema. Se recomienda no utilizar versiones de Python superiores a 3.10, ya que podrían producirse errores de compatibilidad.

Alternativa:

pip install -r requirementsYOLO_NAS.txt

> Nota: El archivo requirementsYOLO_NAS.txt se incluye en este repositorio y permite instalar todas las dependencias necesarias de una sola vez.

### Para el Modelo Propio:


## Resultados y Métricas

Después de completar los entrenamientos:

resultados_entrenamiento.csv → Métricas de los modelos.

El archivo contiene 4 campos: mAP50,mAP75,mAP50-95 y Tiempo de Entrenamiento (s)

## Referencias

- Este proyecto utiliza el framework [Super-Gradients](https://zenodo.org/records/7789328) para el modelo YOLO-NAS, desarrollado por Shay Aharon y colaboradores.
  
- También se hace uso de [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), desarrollado por Glenn Jocher y Jing Qiu, bajo licencia AGPL-3.0.
  
- El dataset base utilizado es el conjunto [Constellation Dataset](https://universe.roboflow.com/ws-qwbuh/constellation-dsphi) disponible en Roboflow Universe, creado por WS (2023).

## Autor
Pablo Antonio Rodriguez Hernandez

Estudiante de Ingeniería Informática.

Trabajo Fin de Grado - 2025.



