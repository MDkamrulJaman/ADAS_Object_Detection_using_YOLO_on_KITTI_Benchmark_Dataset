from ultralytics import YOLO
 
# SETTINGS 
MODEL = "runs/detect/train/weights/best.onnx"
DATA  = "src/dataset/data.yaml"
IMGSZ = 640

 
# load trained model
model = YOLO(MODEL)
 
# run evaluation on val set
metrics = model.val(
    data  = DATA,
    imgsz = IMGSZ,
)
 
# print results
print("\nEvaluation Results:")
print(f"  mAP50     : {metrics.box.map50:.4f}")
print(f"  mAP50-95  : {metrics.box.map:.4f}")
print(f"  Precision : {metrics.box.mp:.4f}")
print(f"  Recall    : {metrics.box.mr:.4f}")



