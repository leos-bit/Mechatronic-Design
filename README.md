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
```bash
python3 "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/Mechatronic Design/code/belt_objects.py" \
  --video "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/Mechatronic Design/IMG_0985.mov" \
  --yolo-model "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/Mechatronic Design/trials/trial3/packdet-2class-detect/weights/best.pt" \
  --yolo-bottle-names "bottle" \
  --yolo-can-names "can" \
  --yolo-conf 0.2 \
  --show
```

## Extract Frames for Labeling
```bash
python3 "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/Mechatronic Design/code/extract_frames.py" \
  --video "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/Mechatronic Design/IMG_0985.mov" \
  --out "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/Mechatronic Design/labels/labels_src" \
  --count 200 \
  --start 0 \
  --step 5
```

## Train (Detect on PackDet 2-class)
```bash
/Users/leoshaw/Documents/VSCode/.venv314/bin/yolo detect train \
  data="/Users/leoshaw/Documents/VSCode/VS_CMU_S26/Mechatronic Design/data/packdet_2class/data.yaml" \
  model="/Users/leoshaw/Documents/VSCode/VS_CMU_S26/Mechatronic Design/models/yolov8n.pt" \
  imgsz=640 epochs=10 batch=8 \
  project="/Users/leoshaw/Documents/VSCode/VS_CMU_S26/Mechatronic Design/trials" \
  name=packdet-2class-detect
```

## Notes
- Large datasets, models, and training runs are excluded via `.gitignore`.
- If you want to track models, use Git LFS.
