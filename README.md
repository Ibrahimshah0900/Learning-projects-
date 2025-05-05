

## 🧠 Brain Tumor Detection & Segmentation with YOLOv11 + SAM

An AI-powered deep learning pipeline that performs **brain tumor detection** using **YOLOv11** and **precise segmentation** using **Meta’s Segment Anything Model (SAM)**. This project explores how object detection and segmentation can be combined to assist medical professionals in analyzing MRI brain scans faster and more accurately.

---

### 📌 Table of Contents

* [📌 Table of Contents](#-table-of-contents)
* [📸 Demo](#-demo)
* [🚀 Features](#-features)
* [📁 Dataset](#-dataset)
* [🛠️ Technologies Used](#️-technologies-used)
* [⚙️ Setup Instructions](#️-setup-instructions)
* [🔍 Inference Guide](#-inference-guide)
* [📊 Results](#-results)
* [📎 License](#-license)

---

### 📸 Demo

| Detection (YOLOv11)                               | Segmentation (SAM)                                     |
| ------------------------------------------------- | ------------------------------------------------------ |
| ![Detection](./sample_outputs/yolo_detection.png) | ![Segmentation](./sample_outputs/sam_segmentation.png) |

---

### 🚀 Features

✅ Object Detection using YOLOv11
✅ Tumor Segmentation using SAM (Segment Anything Model)
✅ Real-time inference on test MRI images
✅ Combined pipeline for detection + segmentation
✅ Visualized bounding boxes and segmentation masks
✅ GPU-compatible, runs on Google Colab

---

### 📁 Dataset

* **Type**: Brain MRI Scans
* **Classes**: Tumor
* **Annotation Format**: YOLO format (`.txt`)
* **Split**: Train / Valid / Test
* **Source**: \[Your data source, e.g., Kaggle / Roboflow / custom upload]

📄 Dataset config: `data.yaml`

---

### 🛠️ Technologies Used

* Python 3.10+
* [Ultralytics](https://github.com/ultralytics/ultralytics)
* [YOLOv11 (custom-trained)](https://github.com/ultralytics/ultralytics)
* [Meta’s Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
* OpenCV, Matplotlib
* Google Colab / CUDA GPU (for training)

---

### ⚙️ Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/brain-tumor-yolo11-sam.git
cd brain-tumor-yolo11-sam
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
# Or manually install
pip install ultralytics opencv-python matplotlib segment-anything
```

3. **Download or link your dataset**
   Ensure `data.yaml` points to correct `train`, `val`, and `test` paths.

4. **Train the YOLOv11 model**

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # base model
model.train(
    data="path/to/data.yaml",
    epochs=20,
    imgsz=640,
    device=0
)
```

5. **Run inference**

```python
model = YOLO("runs/detect/train/weights/best.pt")
results = model("path/to/test/image.jpg", save=True)
```

6. **Run SAM for segmentation**

```python
# Load pre-trained SAM and run segmentation using detected bounding box from YOLO
# See segment.py for full implementation
```

---

### 🔍 Inference Guide

You can test the model on individual images or in batch mode.

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("test_images/", save=True)

for result in results:
    boxes = result.boxes
    print(boxes)
```

Then pass YOLO-detected regions to SAM for segmentation.

---

### 📊 Results

* **YOLOv11**: Fast and accurate bounding box prediction
* **SAM**: Clean segmentation mask around tumor regions
* **Combined Pipeline**: High-precision tumor localization and outlining

---

### 📎 License

This project is licensed under the [MIT License](LICENSE).

---

