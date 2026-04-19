#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
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


def yolo_txt_line(cls_id, x1, y1, x2, y2, width, height):
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(a_area + b_area - inter, 1e-6)
    return inter / union


def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y2 = max(0, min(int(round(y2)), height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def write_data_yaml(out_root, ordered_classes):
    class_to_id = {name: idx for idx, name in enumerate(ordered_classes)}
    yaml_lines = [
        f"path: {out_root}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for class_name in ordered_classes:
        yaml_lines.append(f"  {class_to_id[class_name]}: {class_name}")
    (out_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def ensure_dirs(root):
    dirs = {
        "images_train": root / "next_batch" / "images" / "train",
        "images_val": root / "next_batch" / "images" / "val",
        "labels_train": root / "next_batch" / "labels" / "train",
        "labels_val": root / "next_batch" / "labels" / "val",
        "fp_crops": root / "review" / "fp_crops",
        "fn_crops": root / "review" / "fn_crops",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Error-analysis loop: mine FP/FN candidates and build next training batch."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--model", required=True, help="YOLO .pt model path")
    parser.add_argument("--out", required=True, help="Output root folder")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--step", type=int, default=4, help="Sample every Nth frame")
    parser.add_argument("--max-frames", type=int, default=400, help="Max sampled frames")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf-low", type=float, default=0.15, help="Low confidence threshold")
    parser.add_argument("--conf-high", type=float, default=0.50, help="Trusted confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="YOLO NMS IoU threshold")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--motion-min-area-fraction", type=float, default=0.0015,
                        help="Min moving contour area fraction to consider as missed object")
    parser.add_argument("--unmatched-iou-max", type=float, default=0.2,
                        help="Max IoU with detections to count moving blob as FN candidate")
    parser.add_argument("--max-fp-crops", type=int, default=500, help="Cap FP candidate crops")
    parser.add_argument("--max-fn-crops", type=int, default=500, help="Cap FN candidate crops")
    parser.add_argument("--bottle-names", default="bottle", help="Comma-separated bottle aliases")
    parser.add_argument("--can-names", default="can", help="Comma-separated can aliases")
    parser.add_argument("--six-pack-names", default="6-pack,six-pack,six_pack,6pack",
                        help="Comma-separated six-pack aliases")
    parser.add_argument("--include-six-pack", action="store_true",
                        help="Write third six_pack class in next_batch labels")
    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    out_root = Path(args.out)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    class_aliases = parse_class_aliases(args.bottle_names, args.can_names, args.six_pack_names)
    class_to_id, ordered_classes = build_class_index(class_aliases, args.include_six_pack)
    dirs = ensure_dirs(out_root)
    write_data_yaml(out_root / "next_batch", ordered_classes)

    metadata_path = out_root / "review" / "candidates.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["type", "frame", "crop_path", "class", "conf", "x1", "y1", "x2", "y2"])

        model = YOLO(str(model_path))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)

        frame_idx = 0
        sampled = 0
        saved_frames = 0
        saved_labels = 0
        fp_saved = 0
        fn_saved = 0

        if args.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
            frame_idx = args.start_frame

        while sampled < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if (frame_idx - args.start_frame) % args.step != 0:
                frame_idx += 1
                continue

            sampled += 1
            height, width = frame.shape[:2]

            pred = model.predict(frame, verbose=False, imgsz=args.imgsz, conf=args.conf_low, iou=args.iou)
            boxes = pred[0].boxes if pred else None

            all_dets = []
            trusted_dets = []
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
                    raw_label = pred[0].names.get(cls_id, str(cls_id))
                    mapped = map_class(raw_label, class_aliases)
                    if mapped is None:
                        continue
                    if mapped == "six_pack" and not args.include_six_pack:
                        mapped = "can"
                    conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, width, height)
                    det = {
                        "class": mapped,
                        "conf": conf,
                        "xyxy": (x1, y1, x2, y2),
                    }
                    all_dets.append(det)
                    if conf >= args.conf_high:
                        trusted_dets.append(det)

                    if conf < args.conf_high and fp_saved < args.max_fp_crops:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            crop_name = f"fp_frame{frame_idx:06d}_{fp_saved:04d}.jpg"
                            crop_path = dirs["fp_crops"] / crop_name
                            cv2.imwrite(str(crop_path), crop)
                            writer.writerow(
                                ["fp", frame_idx, str(crop_path), mapped, f"{conf:.4f}", x1, y1, x2, y2]
                            )
                            fp_saved += 1

            # Motion-based FN candidates: moving blobs not explained by any detection.
            fg = bg_sub.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_motion_area = float(width * height) * args.motion_min_area_fraction
            fn_in_frame = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_motion_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                blob_xyxy = (x, y, x + w, y + h)
                best_iou = 0.0
                for det in all_dets:
                    best_iou = max(best_iou, iou_xyxy(blob_xyxy, det["xyxy"]))
                if best_iou <= args.unmatched_iou_max and fn_saved < args.max_fn_crops:
                    x1, y1, x2, y2 = clamp_box(x, y, x + w, y + h, width, height)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_name = f"fn_frame{frame_idx:06d}_{fn_saved:04d}.jpg"
                        crop_path = dirs["fn_crops"] / crop_name
                        cv2.imwrite(str(crop_path), crop)
                        writer.writerow(
                            ["fn", frame_idx, str(crop_path), "unknown", "", x1, y1, x2, y2]
                        )
                        fn_saved += 1
                        fn_in_frame += 1

            # Build next_batch frame list from trusted detections + hard frames.
            hard_frame = (len(all_dets) > len(trusted_dets)) or (fn_in_frame > 0)
            keep_frame = bool(trusted_dets) or hard_frame
            if keep_frame:
                split_is_val = ((saved_frames + 1) % max(int(round(1 / max(args.val_ratio, 1e-6))), 2) == 0)
                img_dir = dirs["images_val"] if split_is_val else dirs["images_train"]
                lbl_dir = dirs["labels_val"] if split_is_val else dirs["labels_train"]
                stem = f"frame_{frame_idx:06d}"
                img_path = img_dir / f"{stem}.jpg"
                lbl_path = lbl_dir / f"{stem}.txt"
                cv2.imwrite(str(img_path), frame)

                lines = []
                for det in trusted_dets:
                    cls_id = class_to_id[det["class"]]
                    x1, y1, x2, y2 = det["xyxy"]
                    lines.append(yolo_txt_line(cls_id, x1, y1, x2, y2, width, height))
                with open(lbl_path, "w", encoding="utf-8") as label_file:
                    if lines:
                        label_file.write("\n".join(lines) + "\n")
                saved_frames += 1
                if lines:
                    saved_labels += 1

            frame_idx += 1

        cap.release()

    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Sampled frames: {sampled}")
    print(f"Next-batch frames: {saved_frames}")
    print(f"Next-batch labeled frames: {saved_labels}")
    print(f"FP candidate crops: {fp_saved}")
    print(f"FN candidate crops: {fn_saved}")
    print(f"Review metadata: {metadata_path}")
    print(f"Dataset YAML: {out_root / 'next_batch' / 'data.yaml'}")


if __name__ == "__main__":
    main()
