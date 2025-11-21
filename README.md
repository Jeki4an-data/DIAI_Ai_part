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

