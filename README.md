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

### Key characteristics
- Images collected across **multiple European municipalities**
- Mixed **urban and rural** environments
- Captured using **vehicle-mounted cameras**
- Includes shadows, occlusions, blurriness, and perspective distortion
- All sensitive data anonymized (faces and plates blurred)
- Initially created for **detection + classification** research

### Annotation format
- COCO JSON structure
- Polygons for objects >30px  
- Bounding boxes for objects 15–30px (marked as *difficult*)
- 200+ traffic sign categories
- ~13,000 tightly annotated signs

### Why filtering was required
The goal of the project was to train a lightweight small-class YOLO model.  
Therefore, only **3 target classes** were extracted:

| Original Dataset | After Filtering |
|------------------|----------------|
| 200+ classes | 3 classes |
| ~13k labeled objects | Only IDs 59, 60, 157 |
| Many annotation types | YOLO bounding boxes only |
| All images | Only those containing relevant signs |

This produced a clean, compact dataset optimized for custom detection tasks.

---

## 📂 YOLO Dataset Preparation

Data preparation was performed with plain Python code (not standalone scripts).

### Final YOLO structure
dataset/
├── images/
│ ├── train/
│ └── val/
└── labels/
├── train/
└── val/


```Each `.txt` file follows the YOLO format:```


All coordinates are normalized to 0–1.

---

## 🧹 1. Class Filtering (real code used)

Only classes **59, 60, 157** were kept.  
If a label file contained none of them, the `.txt` and its corresponding image were deleted.

```python
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

print("✔ Filtering completed!")```



