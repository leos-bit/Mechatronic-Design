# Mechatronic-Design

This repo contains code and experiments for conveyor-belt object detection using YOLO and OpenCV. The current computer vision pipeline supports bottles, cans, and an optional `six_pack` class.

## Key scripts
- `code/belt_objects.py`: runtime detection, centroid estimation, tracking, and homography projection
- `code/auto_label_video.py`: sample frames from a video and auto-generate YOLO labels
- `code/extract_frames.py`: pull frames for manual labeling

## Detection
Run from `Computer Vision/`:

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
  --yolo-six-pack-names "6-pack,six-pack,six_pack,6pack" \
  --centroid-mode refined \
  --yolo-conf 0.35 \
  --show
```

If you are still using a 2-class model and want to test `six_pack` before retraining, add:

```bash
--enable-six-pack-heuristic --six-pack-min-area-fraction 0.025 --six-pack-min-aspect 1.2
```

## Auto-labeling
Generate a 3-class dataset when your model already has a six-pack class, or when you want to seed labels with the heuristic:

```bash
python3 "code/auto_label_video.py" \
  --video "IMG_1055.mov" \
  --model "trials/trial5-manual-auto/weights/best.pt" \
  --out "manual_3class_auto" \
  --num-frames 250 \
  --step 5 \
  --include-six-pack \
  --enable-six-pack-heuristic
```

## Notes
- `centroid-mode refined` estimates the object center from the silhouette inside each YOLO box instead of using the raw box center.
- Track labels are now stabilized with confidence-weighted votes to reduce frame-to-frame flips.
- Large datasets, models, and training runs are excluded via `.gitignore`.
