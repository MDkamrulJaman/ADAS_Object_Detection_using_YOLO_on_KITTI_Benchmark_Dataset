from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train/weights/best.onnx")  # load a custom-trained model

# Export the model
model.export(format="onnx")