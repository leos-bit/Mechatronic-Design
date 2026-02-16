#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_name_list(value):
    if not value:
        return []
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def map_class(raw_label, bottle_names, can_names):
    l = str(raw_label).lower()
    if any(name in l for name in bottle_names):
        return 0
    if any(name in l for name in can_names):
        return 1
    return None


def yolo_txt_line(cls_id, x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    parser = argparse.ArgumentParser(description="Sample frames from video and auto-label with YOLO.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--model", required=True, help="YOLO .pt model path")
    parser.add_argument("--out", required=True, help="Output dataset root")
    parser.add_argument("--num-frames", type=int, default=250, help="Max sampled frames to save")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--step", type=int, default=5, help="Sample every Nth frame")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="YOLO NMS IoU threshold")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--save-empty", action="store_true", help="Save sampled frames with no detections")
    parser.add_argument("--bottle-names", default="bottle", help="Comma-separated bottle class substrings")
    parser.add_argument("--can-names", default="can", help="Comma-separated can class substrings")
    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    out_root = Path(args.out)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    image_train = out_root / "images" / "train"
    image_val = out_root / "images" / "val"
    label_train = out_root / "labels" / "train"
    label_val = out_root / "labels" / "val"
    for p in [image_train, image_val, label_train, label_val]:
        p.mkdir(parents=True, exist_ok=True)

    bottle_names = parse_name_list(args.bottle_names)
    can_names = parse_name_list(args.can_names)

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    kept = 0
    det_frames = 0

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        frame_idx = args.start_frame

    while kept < args.num_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_idx - args.start_frame) % args.step != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        pred = model.predict(frame, verbose=False, imgsz=args.imgsz, conf=args.conf, iou=args.iou)
        boxes = pred[0].boxes if pred else None

        lines = []
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                raw_label = pred[0].names.get(cls_id, str(cls_id))
                mapped = map_class(raw_label, bottle_names, can_names)
                if mapped is None:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                x1 = max(0.0, min(x1, w - 1))
                y1 = max(0.0, min(y1, h - 1))
                x2 = max(0.0, min(x2, w - 1))
                y2 = max(0.0, min(y2, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(yolo_txt_line(mapped, x1, y1, x2, y2, w, h))

        if not lines and not args.save_empty:
            frame_idx += 1
            continue

        split_is_val = ((kept + 1) % max(int(round(1 / max(args.val_ratio, 1e-6))), 2) == 0)
        img_dir = image_val if split_is_val else image_train
        lbl_dir = label_val if split_is_val else label_train

        stem = f"frame_{frame_idx:06d}"
        img_path = img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"

        cv2.imwrite(str(img_path), frame)
        with open(lbl_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        kept += 1
        if lines:
            det_frames += 1
        frame_idx += 1

    cap.release()

    data_yaml = out_root / "data.yaml"
    yaml_text = (
        f"path: {out_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: bottle\n"
        "  1: can\n"
    )
    data_yaml.write_text(yaml_text, encoding="utf-8")

    print(f"Saved frames: {kept}")
    print(f"Frames with detections: {det_frames}")
    print(f"Dataset: {out_root}")
    print(f"YAML: {data_yaml}")


if __name__ == "__main__":
    main()
