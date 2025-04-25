def main():
    from ultralytics import YOLO

    # Cargar un modelo YOLOv11 preentrenado
    model = YOLO("yolo11n.pt")

    # Entrenar el modelo con un dataset personalizado
    train_results = model.train(
        data="C:/Users/pablo/Desktop/TFG/Datasets/Constellation.v1i.yolov11/data.yaml",  
        epochs=100,
        imgsz=640,
        device="cuda",
        val=True,           # Asegura que haga validación durante el entrenamiento
        save=True,          # Guarda los mejores modelos
        save_period=10,     # Guarda cada 10 epochs
        patience=20         # Early stopping si no mejora tras 20 epochs
    )

    # Evaluar el modelo en el conjunto de validación
    metrics = model.val()

    # Realizar detección de objetos en una imagen
    #results = model("path/to/image.jpg")
    #results[0].show()  # Mostrar los resultados

    model.export(format="onnx")         # Muy compatible con APIs Python, C++, etc.
    model.export(format="torchscript")  # Rápido y portable en entornos PyTorch
    model.export(format="openvino") # Para Intel o edge devices


if __name__ == "__main__":
    main()