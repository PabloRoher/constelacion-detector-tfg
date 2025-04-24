def main():
    from ultralytics import YOLO

    # Cargar un modelo YOLOv11 preentrenado
    model = YOLO("yolo11n.pt")

    # Entrenar el modelo con un dataset personalizado
    train_results = model.train(
        data="C:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.yolov11/data.yaml",  
        epochs=50,
        imgsz=640,
        device="cuda" 
    )

    # Evaluar el modelo en el conjunto de validación
    metrics = model.val()

    # Realizar detección de objetos en una imagen
    #results = model("path/to/image.jpg")
    #results[0].show()  # Mostrar los resultados

    model.export(format="onnx")         # Muy compatible con APIs Python, C++, etc.
    model.export(format="torchscript")  # Rápido y portable en entornos PyTorch


if __name__ == "__main__":
    main()