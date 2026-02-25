# Mechatronic-Design

This repo contains code and experiments for classifying bottles vs cans on a conveyor belt using YOLO and OpenCV.

## Structure
- `code/` — scripts (detection + utilities)
- `data/` — datasets (not tracked)
- `labels/` — labeling exports (not tracked)
- `models/` — weights (not tracked)
- `trials/` — training runs (not tracked)
- `results/` — future outputs

## Setup
Activate the Python environment:

```bash
source /Users/leoshaw/Documents/VSCode/.venv314/bin/activate
```

## Run Detection
On Leo's machine:
```bash
python3 "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/MechatronicDesign/Computer Vision/code/belt_objects.py" \
  --video "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/MechatronicDesign/Computer Vision/IMG_1055.mov" \
  --yolo-model "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/MechatronicDesign/Computer Vision/trials/trial5-manual-auto/weights/best.pt" \
  --tracker-type byte \
  --byte-track-config "bytetrack.yaml" \
  --homography-src "0,681;0,0;1079,681;1079,0" \
  --homography-dst "-254,-152.4;-254,152.4;254,-152.4;254,152.4" \
  --homography-units "mm" \
  --yolo-bottle-names "bottle" \
  --yolo-can-names "can" \
  --yolo-conf 0.35 \
  --show
```

Portable command (run from repo root):
```bash
python3 "code/belt_objects.py" \
  --video "IMG_1055.mov" \
  --yolo-model "trials/trial5-manual-auto/weights/best.pt" \
  --tracker-type byte \
  --byte-track-config "bytetrack.yaml" \
  --homography-src "0,681;0,0;1079,681;1079,0" \
  --homography-dst "-254,-152.4;-254,152.4;254,-152.4;254,152.4" \
  --homography-units "mm" \
  --yolo-bottle-names "bottle" \
  --yolo-can-names "can" \
  --yolo-conf 0.35 \
  --show
```

## Extract Frames for Labeling
```bash
python3 "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/MechatronicDesign/Computer Vision/code/extract_frames.py" \
  --video "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/MechatronicDesign/Computer Vision/IMG_0985.mov" \
  --out "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/MechatronicDesign/Computer Vision/labels/labels_src" \
  --count 200 \
  --start 0 \
  --step 5
```

## Train (Detect on PackDet 2-class)
```bash
/Users/leoshaw/Documents/VSCode/.venv314/bin/yolo detect train \
  data="/Users/leoshaw/Documents/VSCode/VS_CMU_S26/MechatronicDesign/Computer Vision/packdet_2class/data.yaml" \
  model="/Users/leoshaw/Documents/VSCode/VS_CMU_S26/MechatronicDesign/Computer Vision/models/yolov8n.pt" \
  imgsz=640 epochs=30 batch=8 \
  project="/Users/leoshaw/Documents/VSCode/VS_CMU_S26/MechatronicDesign/Computer Vision/trials" \
  name=trial4
```

## Notes
- Large datasets, models, and training runs are excluded via `.gitignore`.
- If you want to track models, use Git LFS.
