# 🚗 ADAS Object Detection using YOLO on KITTI Benchmark Dataset

---

## 📌 Project Overview

This project implements an **Advanced Driver Assistance System (ADAS) perception pipeline** using the **YOLO object detection framework** trained on the **KITTI Vision Benchmark Dataset**.

The system detects important road participants:

- 🚗 Cars
- 🚶 Pedestrians
- 🚲 Cyclists
- 🚚 Vehicles

The objective is to develop a **real-time computer vision perception module** suitable for autonomous driving applications.

The complete pipeline includes:

```
KITTI Dataset
      |
      ↓
Annotation Conversion
(KITTI → YOLO Format)
      |
      ↓
YOLO Model Training
      |
      ↓
Performance Evaluation
      |
      ↓
Real-Time Detection
```

---

## 🎯 Project Objectives

- Build an ADAS object detection system using deep learning
- Convert KITTI annotations into YOLO format
- Train YOLO on autonomous driving images
- Evaluate detection performance using industry-standard metrics
- Analyze model limitations for real-world deployment

---

## 🧠 System Architecture

```
                 KITTI Dataset
                      |
                      ↓
        +----------------------------+
        | Dataset Preprocessing      |
        | KITTI Annotation Parser    |
        +----------------------------+
                      |
                      ↓
        +----------------------------+
        | YOLO Format Conversion     |
        | Bounding Box Normalization |
        +----------------------------+
                      |
                      ↓
        +----------------------------+
        | YOLO Object Detector       |
        | Backbone + Neck + Head     |
        +----------------------------+
                      |
                      ↓
        +----------------------------+
        | Evaluation Pipeline        |
        | mAP Precision Recall       |
        +----------------------------+
                      |
                      ↓
        +----------------------------+
        | ADAS Detection System      |
        | Real-Time Inference        |
        +----------------------------+
```

---

## 🛠️ Tech Stack

**Programming Language**
- Python 3.10

**Deep Learning Framework**
- PyTorch

**Object Detection Framework**
- Ultralytics YOLO

**Computer Vision**
- OpenCV
- NumPy

**Data Processing**
- Pandas
- YAML
- tqdm

**Visualization**
- Matplotlib

**Hardware (Training Environment)**
```
GPU: NVIDIA A100 SXM 80GB
CUDA Enabled Training
```

---

## ⭐ Key Features

### 🚘 Real-Time Object Detection
Detects road objects from camera images using YOLO inference.

### 🔄 KITTI → YOLO Conversion Pipeline
Automatically converts KITTI annotations:

```
KITTI Bounding Box
xmin, ymin, xmax, ymax

↓

YOLO Format
class_id, x_center, y_center, width, height
```

### 📊 Performance Evaluation
Includes:
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- Training loss analysis

### ⚡ Optimized Training Pipeline
Supports:
- GPU acceleration
- Batch training
- Custom YAML configuration
- Model checkpoint saving

---

## 📂 Project Structure

```
ADAS-YOLO-KITTI/
│
├── data/
│   ├── images/
│   │    ├── train/
│   │    └── val/
│   │
│   ├── labels/
│   │    ├── train/
│   │    └── val/
│   │
│   └── kitti_original/
│
├── configs/
│   └── kitti.yaml
│
├── scripts/
│   ├── kitti_to_yolo.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── models/
│   └── best.pt
│
├── results/
│   ├── confusion_matrix.png
│   ├── metrics.csv
│   └── predictions/
│
├── notebooks/
│   └── evaluation.ipynb
│
├── tests/
│   └── test_inference.py
│
├── requirements.txt
│
└── README.md
```

---

## 📦 Dataset

### KITTI Vision Benchmark Dataset

Dataset contains:
- Stereo camera images
- LiDAR data
- Object annotations

Used subset:
```
KITTI Object Detection Dataset
```

Classes:

| Class      | Description    |
|------------|-----------------|
| Car        | Vehicles       |
| Pedestrian | Human objects  |
| Cyclist    | Bicycle riders |

---

## 🔄 KITTI → YOLO Conversion

### Annotation Mapping

| KITTI | YOLO     |
|-------|----------|
| Class | class_id |
| xmin  | x_center |
| ymin  | y_center |
| xmax  | width    |
| ymax  | height   |

### Conversion Formula

```python
x_center = ((xmin + xmax) / 2) / image_width
y_center = ((ymin + ymax) / 2) / image_height
width = (xmax - xmin) / image_width
height = (ymax - ymin) / image_height
```

---

## ⚙️ Installation & Local Setup

### 1. Clone Repository

```bash
git clone https://github.com/MDkamrulJaman/ADAS_Object_Detection_using_YOLO_on_KITTI_Benchmark_Dataset.git
cd ADAS_Object_Detection_using_YOLO_on_KITTI_Benchmark_Dataset
```

### 2. Create Virtual Environment

**Linux / Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📚 Requirements

`requirements.txt`

```
ultralytics
torch
torchvision
opencv-python
numpy
pandas
matplotlib
scikit-learn
pyyaml
tqdm
```

---

## 🏋️ Training

Example YOLO training command:

```bash
yolo detect train \
model=yolov8.pt \
data=configs/kitti.yaml \
epochs=100 \
imgsz=640 \
batch=16
```

---

## 🔍 Inference

Run detection:

```bash
yolo detect predict \
model=models/best.pt \
source=data/images/val \
conf=0.25
```

Output:

```
results/
 └── predictions/
       ├── image1.jpg
       ├── image2.jpg
```

---

## 🧪 Testing

Run automated tests:

```bash
pytest tests/
```

Inference testing:

```bash
python tests/test_inference.py
```

---

## 📈 Model Performance

### Training Configuration

| Parameter     | Value       |
|---------------|-------------|
| Model         | YOLO        |
| Epochs        | 100         |
| Learning Rate | 0.000017    |
| Image Size    | 640         |
| GPU           | NVIDIA A100 |

---

## 📊 Final Results

| Metric       | Score      |
|--------------|------------|
| mAP@0.5      | **0.8667** |
| mAP@0.5:0.95 | **0.6358** |
| Precision    | **0.9111** |
| Recall       | **0.7447** |

---

## 📉 Loss Analysis

| Loss                | Training | Validation |
|----------------------|----------|------------|
| Box Loss            | 0.87872  | 0.90454    |
| Classification Loss | 0.49086  | 0.50919    |
| DFL Loss            | 0.00235  | 0.00482    |

---

## 📊 Performance Analysis

### Strengths

✅ High precision (91%)
- Low false positive detection rate

✅ Strong mAP@0.5
- Reliable object localization

✅ Stable training
- Training and validation losses are close

### Weaknesses

❌ Lower recall compared with precision

1. **Recall (0.74)** indicates missed detections in some cases
2. Difficulty detecting:
   - Small objects (far pedestrians/cyclists)
   - Occluded objects
3. Slight gap in validation loss → minor generalization limitation

---

## ⚠️ Limitations

- KITTI dataset has limited environmental diversity
- No temporal information used
- No object tracking
- Limited night/rain scenarios
- Camera-only perception

---

## 🚀 Future Improvements

### Model Improvements
- YOLOv9 / YOLOv10 comparison
- Transformer-based detectors
- Instance segmentation

### Autonomous Driving Extensions
- LiDAR + Camera fusion
- Sensor calibration
- Object tracking using ByteTrack
- Lane detection integration

### Deployment
- TensorRT optimization
- NVIDIA Jetson deployment
- Real-time vehicle integration

---

## 📚 Citations

### KITTI Dataset

```bibtex
@inproceedings{geiger2012kitti,
title={Are we ready for Autonomous Driving?
The KITTI Vision Benchmark Suite},
author={Geiger, Andreas and Lenz, Philip and Urtasun, Raquel},
booktitle={CVPR},
year={2012}
}
```

### YOLO

```bibtex
@inproceedings{redmon2016yolo,
title={You Only Look Once: Unified, Real-Time Object Detection},
author={Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
booktitle={CVPR},
year={2016}
}
```

---

## 👤 Author

**MD. Kamrul Jaman**
Autonomous Vehicle Engineering | AI Engineer | Computer Vision

Technische Hochschule Ingolstadt

Areas:
- Deep Learning
- Object Detection
- ADAS
- RAG Systems
- AI Engineering

---

## 📜 License

MIT License

---

## ⭐ Acknowledgements

Special thanks to:
- KITTI Vision Benchmark
- Ultralytics YOLO Team
- PyTorch Community
- Open-source Computer Vision Community

---

## 🚗 Project Summary

This repository demonstrates an end-to-end **ADAS perception pipeline** using YOLO and KITTI, covering:

✅ Dataset processing
✅ Annotation conversion
✅ Deep learning training
✅ Evaluation
✅ Real-time object detection

The project represents a practical foundation for autonomous driving perception systems.
