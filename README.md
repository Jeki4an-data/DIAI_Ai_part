# DIAI_Ai_part
# Traffic Sign Detection Training (YOLOv8)

This repository contains the full training workflow for a 3-class traffic sign detection model using the YOLOv8 framework.
The project includes dataset preparation, class filtering, class remapping, YOLO-format conversion, training, validation, and inference.

The trained model detects three specific road signs:

no_stop (original ID: 59 → YOLO class 0)

no_waiting (original ID: 60 → YOLO class 1)

parking_for_disabled (original ID: 157 → YOLO class 2)

📊 Dataset Overview

The dataset is based on a large-scale European traffic sign dataset (originally COCO-annotated), widely used in academic work on traffic-sign recognition and autonomous driving.

Dataset characteristics

Images collected across multiple European municipalities

Captured using vehicle-mounted cameras

Includes urban and rural environments

Contains daytime conditions, shadows, occlusions, angle distortions

Anonymization applied (faces and license plates blurred)

Originally designed for detection + recognition tasks

Annotation details

COCO JSON format with:

Polygon masks for signs larger than 30 px

Bounding boxes for signs between 15–30 px

Difficult samples flagged and ignored during training

~200 different traffic sign categories

More than 13,000 tightly annotated signs (>30 px)

Why filtering was needed

The project required training a lightweight model for only 3 specific traffic signs, so the original dataset was heavily reduced:

Original Dataset	After Filtering
200+ classes	3 classes
~13,000 labeled objects	Only signs with ID 59, 60, 157
Many annotation types	Only YOLO bounding-box labels
All images	Only images containing relevant signs

The result is a smaller, clean, 3-class YOLO dataset optimized for real-time detection.

📂 YOLO Dataset Preparation
The dataset was prepared manually using simple Python code (no external scripts or special tools).
✅ 1. Final Dataset Structure
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/


Each .txt file contains YOLO-formatted labels:

class cx cy w h


(normalized coordinates)

✅ 2. Class Filtering (real code used)

We kept only classes 59, 60, 157.
Labels without these classes were deleted along with their images.

import os
import shutil

SELECTED = [59, 60, 157]

BASE = "/"   # path to dataset root
IMG_TRAIN = f"{BASE}/images/train"
IMG_VAL   = f"{BASE}/images/val"
LBL_TRAIN = f"{BASE}/labels/train"
LBL_VAL   = f"{BASE}/labels/val"

def filter_subset(img_dir, lbl_dir):
    for lbl_file in os.listdir(lbl_dir):
        if not lbl_file.endswith(".txt"):
            continue

        lbl_path = f"{lbl_dir}/{lbl_file}"

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            cls = int(line.split()[0])
            if cls in SELECTED:
                new_lines.append(line)

        # Remove empty-label files + their images
        if len(new_lines) == 0:
            os.remove(lbl_path)

            img_jpg = lbl_file.replace(".txt", ".jpg")
            img_png = lbl_file.replace(".txt", ".png")

            if os.path.exists(f"{img_dir}/{img_jpg}"):
                os.remove(f"{img_dir}/{img_jpg}")

            if os.path.exists(f"{img_dir}/{img_png}"):
                os.remove(f"{img_dir}/{img_png}")

            continue

        with open(lbl_path, "w") as f:
            f.writelines(new_lines)

print("Filtering train...")
filter_subset(IMG_TRAIN, LBL_TRAIN)

print("Filtering val...")
filter_subset(IMG_VAL, LBL_VAL)

print("✔ Filtering completed!")

✅ 3. Remapping Class IDs → YOLO Standard

YOLO requires continuous class indices starting from 0.

So the mapping became:

Original ID	YOLO ID
59	0
60	1
157	2

All label files were updated accordingly.

📄 data.yaml
path: /Users/denisvasilev/Desktop/DFG_yolo

train: images/train
val: images/val

nc: 3

names:
  0: "no_stop"
  1: "no_waiting"
  2: "parking_for_disabled"

🏋️ Model Training

Training was done using real code below — exactly as used in the project:

from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="/",
    epochs=15,
    imgsz=512,
    batch=32,
    device=cpu,
    amp=True
)

Why YOLOv8m?

Good accuracy/speed trade-off

Works well with small datasets

Suitable for CPU/GPU training

Handles small traffic signs better than YOLO-nano models

📊 Model Evaluation

Evaluation was done with:

results = model.val(data="data.yaml")
print(results)


A custom pretty-print formatter was added to show:

Precision

Recall

mAP@50

mAP@50–95

Per-class metrics (precision, recall, AP50)

Example:

==================================================
📊 YOLO Evaluation Results
==================================================

🔹 Overall Metrics:
   - Precision:        0.9906
   - Recall:           0.8157
   - mAP@0.5:          0.9255
   - mAP@0.5:0.95:     0.7629

🔹 Per-Class Metrics:
   Class 0 ('no_stop'):
      Precision: 0.9970
      Recall:    0.9000
      AP@50:     0.8550

   Class 1 ('no_waiting'):
      Precision: 0.9980
      Recall:    0.9000
      AP@50:     0.9159

   Class 2 ('parking_for_disabled'):
      Precision: 0.9990
      Recall:    1.0000
      AP@50:     0.6144
==================================================

🔍 Inference Example
model.predict(
    source="image.jpg",
    save=True,
    conf=0.25
)

✔ Summary

This project documents a full training pipeline for a YOLOv8 detection model:

Included:

Dataset understanding + characteristics

Filtering to 3 target traffic signs

Class remapping (59/60/157 → 0/1/2)

Preparing YOLO dataset structure

Training a YOLOv8m model on 512×512 resolution

Evaluation with readable metric output

Single-image inference
