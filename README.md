## 📊 Datasets Used

### 1️⃣ License Plate Detection Dataset  
The license plate detector (YOLOv8) was trained on a dataset containing:

- Full vehicle images  
- A single bounding box per image labeled **license_plate**  
- Mixed day/night conditions and various camera angles  
- Data converted into YOLO format (`class x_center y_center width height`)

**Goal:** Detect the exact region of the license plate on any vehicle image.

**Dataset composition:**
- 500 images (custom dataset collected for the hackathon)  
- `train / val` split  
- Basic augmentations: flip, blur, brightness and contrast adjustments  
- One class only: `license_plate`

https://nomeroff.net.ua/datasets/ (Public)

---

### 2️⃣ OCR Character Detection Dataset  
The OCR model was trained on a separate, larger dataset that contains **cropped license plates**, each annotated at the **character level**.

**Dataset includes:**
- Cropped plate images  
- One bounding box per character  
- YOLO-format labels  
- 24 character classes:  
  `0,1,2,3,4,5,6,7,8,9,A,B,C,E,H,I,K,M,O,P,T,X,Y,Z`

**OCR model learns to:**
- Detect each character individually  
- Handle one-line and two-line plates  
- Handle square plates and older formats  
- Work in different lighting conditions and perspectives

**Dataset composition:**
- ~2,000–2,500 annotated plate crops  
- Clear `train / val / test` structure  
- Heavy augmentations: rotation, noise, blur, perspective distortion

This dataset structure is similar to character-level annotation used in Nomeroff Net.

---

### 🧠 Why Two Datasets?

| Model | Dataset | Purpose |
|-------|---------|----------|
| License Plate Detector | Vehicle images with plate bounding boxes | Locate the plate on the image |
| OCR Model | Cropped plates with character boxes | Read characters one by one |

<img width="1122" height="279" alt="image" src="https://github.com/user-attachments/assets/cc3f1dd1-f01c-4f9e-8ad5-6eb73d97c150" />


Using two specialized datasets allows the system to be more accurate, stable, and easier to improve.


# Traffic Sign Detection Training (YOLOv8)

This repository documents the complete workflow for training a **3-class traffic sign detection model** using YOLOv8.  
It includes dataset preparation, class filtering, normalization, YOLO-format conversion, model training, evaluation, and inference.

The final model detects three specific traffic signs:

- **no_stop** (original ID 59 → YOLO class 0)  
- **no_waiting** (original ID 60 → YOLO class 1)  
- **parking_for_disabled** (original ID 157 → YOLO class 2)

---

## 📊 Dataset Overview

The dataset originates from a large European traffic-sign detection corpus, annotated for academic research and autonomous driving applications.
https://www.vicos.si/resources/dfg/ (Public)

### Key characteristics
- Images collected across **multiple European municipalities**
- Mixed **urban and rural** environments
- Captured using **vehicle-mounted cameras**
- Includes shadows, occlusions, blurriness, and perspective distortion
- All sensitive data anonymized (faces and plates blurred)
- Originally designed for **detection + classification** research

### Annotation format
- COCO JSON structure  
- Polygons for objects >30px  
- Bounding boxes for objects 15–30px (marked *difficult*)  
- 200+ traffic sign categories  
- ~13,000 tightly annotated signs

### Why filtering was required
The project required a small, fast model to detect only **3 specific traffic sign classes**.

| Original Dataset | After Filtering |
|------------------|----------------|
| 200+ classes | 3 classes |
| ~13,000 labeled objects | Only IDs 59, 60, 157 |
| Many varied signs | Only target categories |
| All images | Only images containing selected classes |

This produced a compact, clean YOLO dataset.

---

## 📂 YOLO Dataset Preparation

Data preparation was done **with plain Python code**, not standalone scripts.

### Final YOLO structure

```dataset/
├── images/
│ ├── train/
│ └── val/
└── labels/
├── train/
└── val/
```

Each `.txt` file uses YOLO format:

```class cx cy w h```


Coordinates are normalized (0–1).

---

## 🧹 1. Class Filtering (Actual Code Used)

The following code keeps **only classes 59, 60, 157**.  
If a `.txt` file contains none of these — it and its corresponding image are deleted.

```
python
import os
import shutil

SELECTED = [59, 60, 157]

BASE = "/"  # dataset path
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


```
🔢 2. Class Remapping
YOLO requires class indices starting at 0.

```
Original ID	YOLO ID
59	0
60	1
157	2
```

Labels were updated accordingly.

🗂️ data.yaml
```
path: /Users/denisvasilev/Desktop/DFG_yolo

train: images/train
val: images/val

nc: 3

names:
  0: "no_stop"
  1: "no_waiting"
  2: "parking_for_disabled"
```
🏋️ Model Training (Actual Code Used)
Training was launched using:
```
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="/",
    epochs=15,
    imgsz=512,
    batch=32,
    device='cpu',
    amp=True
)
```
Why YOLOv8m?
Better recall on small traffic signs

Strong accuracy/speed trade-off

Works well with limited datasets

Faster convergence vs YOLOx/YOLOl models

📊 Model Evaluation
Executed via:
```
results = model.val(data="data.yaml")
print(results)
```
A custom formatted output summarizes the metrics:
```
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
