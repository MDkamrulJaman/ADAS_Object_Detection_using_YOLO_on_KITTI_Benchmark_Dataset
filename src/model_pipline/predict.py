from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.onnx"
SOURCE_IMAGE = PROJECT_ROOT / "app" / "audi-q9.jpeg"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect" / "train"

model = YOLO(str(MODEL_PATH))

model.predict(
    source=str(SOURCE_IMAGE),
    imgsz=640,
    conf=0.01,
    save=True,
    project=str(OUTPUT_DIR),
    name="prediction",
)
