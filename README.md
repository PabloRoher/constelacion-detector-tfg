Github: https://github.com/PabloRoher/constelacion-detector-tfg.git

# TFG - Detección de Constelaciones usando YOLOv11, YOLO-NAS, RT-DETR y EfficientDet

# Titulo:

## De Mitos a Modelos: Implementación de Técnicas de Inteligencia Artificial para la Detección y Análisis Automático de Constelaciones

Este proyecto tiene como objetivo entrenar, evaluar y comparar diferentes arquitecturas de detección de objetos 
para la tarea específica de identificar constelaciones en imágenes del cielo. Los modelos principales 
implementados son **YOLOv11**, **YOLO-NAS**, **EfficientDet** y **RT-DETR**.

> Nota: En este repositorio se incluye una carpeta `/runs/detect/` que contiene únicamente resultados de entrenamientos de prueba iniciales, como ejemplos de ejecución (principalmente con el modelo YOLOv11n). Los  mejores resultados de cada entrenamiento se encuentran organizados en el archivo llamado  `resultados_entrenamiento`.

Clonar el Repositorio y Configurar Git LFS

Este proyecto utiliza **Git LFS** para gestionar los archivos de modelos grandes. Es necesario tener Git LFS 
instalado.

# Instala Git LFS (solo es necesario una vez por sistema)
git lfs install

# Clona el repositorio
git clone https://github.com/PabloRoher/constelacion-detector-tfg

# Descarga los archivos grandes de LFS (los modelos)
git lfs pull

Este proyecto utiliza **cuatro entornos de trabajo aislados** mediante entornos virtuales de Python para evitar 
conflictos de dependencias entre los diferentes frameworks.

- **YOLOv11 / Ultralytics:** Entorno virtual `venv` - **YOLO-NAS / Super-Gradients:** Entorno virtual `venv_nas` 
- **EfficientDet (TensorFlow):** Entorno virtual `venv_efficientDet`

### Activar el entorno YOLOv11 (`venv`)

# Navega a la raíz del proyecto
cd ruta/del/proyecto

# Activa el entorno virtual (Windows)
.\venv\Scripts\activate

# Ejecuta el script de entrenamiento
python yolov11/trainMainYolov11.py

### Activar el entorno YOLO-NAS (`venv_nas`)

# Navega a la raíz del proyecto
cd ruta/del/proyecto

# Activa el entorno virtual (Windows)
.\venv_nas\Scripts\activate

# Ejecuta el script de entrenamiento
python yolo_nas/train_yolo_nas.py

### Activar el entorno EfficientDet (`venv_efficientDet`)

Este entorno está configurado en WSL2 (Windows Subsystem for Linux) para aprovechar la aceleración por GPU y la 
compatibilidad del ecosistema Linux.

# Desde una terminal de WSL2, navega a la raíz del proyecto
cd ruta/del/proyecto

# Activa el entorno virtual
source venv_efficientdet/bin/activate

# Ejecuta el script de entrenamiento o evaluación
> Nota: El modo ('train' o 'eval') se configura dentro del propio script.
python train_main_efficientDet.py

### Activar el entorno RT-DETR (`venv_rtdetr`)

Al igual que EfficientDet, este entorno se configura y ejecuta en WSL2.

# Desde una terminal de WSL2, navega a la raíz del proyecto
cd ruta/del/proyecto

# Activa el entorno virtual
source venv_rtdetr/bin/activate

# Ejecuta el script de entrenamiento
python rtdetr/train_rtdetr.py

## Requisitos y dependencias

Python 3.10+

### Para YOLOv11/RT-DETR: Ultralytics

Ambos modelos utilizan el mismo framework base. La instalación es directa a través de pip.

pip install ultralytics

### Para YOLO-NAS: Super-Gradients

pip install super-gradients==3.7.1

> Nota: Algunas dependencias como onnx y pycocotools pueden requerir que Visual C++ Build Tools esté instalado 
> en el sistema. Se recomienda no utilizar versiones de Python superiores a 3.10, ya que podrían producirse 
> errores de compatibilidad.

Alternativa, se puede usar el archivo de requisitos proporcionado:

pip install -r requirementsYOLO_NAS.txt

### Para Efficient_Det:

# Desde la terminal de WSL2 con el entorno 'venv_efficientdet' activado

pip install -r requirementsEfficientdet.txt

## Estructura del Dataset

Para los modelos basados en Ultralytics (YOLOv11, RT-DETR) y para el modelo YOLO-NAS, se utiliza el formato de 
dataset YOLO, con una estructura de carpetas específica y un archivo data.yaml que describe las rutas y las 
clases. Para EfficientDet, se utiliza el formato TFRecord, generado a partir de las anotaciones en formato COCO 
JSON.

## Resultados y Métricas

Después de completar los entrenamientos:

resultados_entrenamiento.csv → Métricas de los modelos.

El archivo contiene 4 campos: mAP50,mAP75,mAP50-95 y Tiempo de Entrenamiento (s)

## Referencias

- YOLO-NAS: La implementacion de basa en el framework [Super-Gradients](https://zenodo.org/records/7789328) 
desarrollado por Deci AI.

- YOLOV11/RT-DETR: Se hace uso de [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), desarrollado 
por Glenn Jocher y Jing Qiu, bajo licencia AGPL-3.0.

- EfficientDet: Se utiliza la implementación oficial de Google Research 
(https://github.com/google/automl/tree/master/efficientdet) / AutoML bajo licencia Apache 2.0.

- Dataset: El conjunto base utilizado es el conjunto [Constellation 
Dataset](https://universe.roboflow.com/ws-qwbuh/constellation-dsphi) disponible en Roboflow Universe, creado por 
WS (2023).

## Autor
Pablo Antonio Rodriguez Hernandez

Estudiante de Ingeniería Informática.

Trabajo Fin de Grado - 2025.



