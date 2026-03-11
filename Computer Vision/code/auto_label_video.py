#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_name_list(value):
    if not value:
        return []
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def parse_class_aliases(bottle_names, can_names, six_pack_names):
    return {
        "bottle": parse_name_list(bottle_names),
        "can": parse_name_list(can_names),
        "six_pack": parse_name_list(six_pack_names),
    }


def build_class_index(class_aliases, include_six_pack):
    ordered = ["bottle", "can"]
    if include_six_pack:
        ordered.append("six_pack")
    return {name: idx for idx, name in enumerate(ordered)}, ordered


def map_class(raw_label, class_aliases):
    label = str(raw_label).lower()
    for canonical, aliases in class_aliases.items():
        if any(alias in label for alias in aliases):
            return canonical
    return None


def classify_six_pack_from_geometry(label, x1, y1, x2, y2, width, height, min_area_fraction, min_aspect):
    if label != "can":
        return label
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if box_w <= 0 or box_h <= 0:
        return label
    area_fraction = (box_w * box_h) / max(float(width * height), 1.0)
    aspect = box_w / max(box_h, 1.0)
    if area_fraction >= min_area_fraction and aspect >= min_aspect:
        return "six_pack"
    return label


def yolo_txt_line(cls_id, x1, y1, x2, y2, width, height):
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
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
    parser.add_argument("--six-pack-names", default="6-pack,six-pack,six_pack,6pack",
                        help="Comma-separated six-pack class substrings")
    parser.add_argument("--include-six-pack", action="store_true",
                        help="Write a third six_pack class to labels/data.yaml")
    parser.add_argument("--enable-six-pack-heuristic", action="store_true",
                        help="Promote large, wide can detections to six_pack for 2-class models")
    parser.add_argument("--six-pack-min-area-fraction", type=float, default=0.025,
                        help="Minimum bbox area fraction of the frame to promote can -> six_pack")
    parser.add_argument("--six-pack-min-aspect", type=float, default=1.2,
                        help="Minimum width/height ratio to promote can -> six_pack")
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
    for path in [image_train, image_val, label_train, label_val]:
        path.mkdir(parents=True, exist_ok=True)

    class_aliases = parse_class_aliases(args.bottle_names, args.can_names, args.six_pack_names)
    class_to_id, ordered_classes = build_class_index(class_aliases, args.include_six_pack)

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

        height, width = frame.shape[:2]
        pred = model.predict(frame, verbose=False, imgsz=args.imgsz, conf=args.conf, iou=args.iou)
        boxes = pred[0].boxes if pred else None

        lines = []
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                raw_label = pred[0].names.get(cls_id, str(cls_id))
                mapped = map_class(raw_label, class_aliases)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                x1 = max(0.0, min(x1, width - 1))
                y1 = max(0.0, min(y1, height - 1))
                x2 = max(0.0, min(x2, width - 1))
                y2 = max(0.0, min(y2, height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                if mapped is None and args.enable_six_pack_heuristic:
                    mapped = "can"
                if mapped is None:
                    continue
                if args.include_six_pack and args.enable_six_pack_heuristic:
                    mapped = classify_six_pack_from_geometry(
                        mapped,
                        x1,
                        y1,
                        x2,
                        y2,
                        width,
                        height,
                        args.six_pack_min_area_fraction,
                        args.six_pack_min_aspect,
                    )
                if mapped == "six_pack" and not args.include_six_pack:
                    mapped = "can"
                lines.append(yolo_txt_line(class_to_id[mapped], x1, y1, x2, y2, width, height))

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
        with open(lbl_path, "w", encoding="utf-8") as handle:
            if lines:
                handle.write("\n".join(lines) + "\n")

        kept += 1
        if lines:
            det_frames += 1
        frame_idx += 1

    cap.release()

    data_yaml = out_root / "data.yaml"
    yaml_lines = [
        f"path: {out_root}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for class_name in ordered_classes:
        yaml_lines.append(f"  {class_to_id[class_name]}: {class_name}")
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"Saved frames: {kept}")
    print(f"Frames with detections: {det_frames}")
    print(f"Classes: {', '.join(ordered_classes)}")
    print(f"Dataset: {out_root}")
    print(f"YAML: {data_yaml}")


if __name__ == "__main__":
    main()
