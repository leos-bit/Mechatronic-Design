#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from pathlib import Path

CLASSES = ["bottle", "can"]

# Heuristic thresholds (tune these for your setup).
MIN_AREA = 500
SIX_PACK_AREA = 2000
SIX_PACK_ASPECT_MIN = 1.2
SIX_PACK_EXTENT_MIN = 0.5
SIX_PACK_SOLIDITY_MIN = 0.6

BOTTLE_ELONGATION_MIN = 2.0
BOTTLE_ASPECT_MAX = 0.6

CAN_CIRCULARITY_MIN = 0.75 
CAN_ELONGATION_MAX = 1.2

TRACK_MAX_DISTANCE = 50
TRACK_MAX_AGE = 12

# Sanity filters to avoid classifying the whole belt.
MAX_AREA_FRACTION = 0.4
MAX_DIM_FRACTION = 0.8

# Lock label per object once it is confidently classified.
LOCK_CONF_THRESHOLD = 0.35


def load_classifier(model_path):
    if model_path is None:
        return None
    net = cv2.dnn.readNetFromONNX(str(model_path))
    return net


def classify_with_model(net, roi_bgr):
    # Assumes a standard image classifier outputting logits over CLASSES.
    blob = cv2.dnn.blobFromImage(roi_bgr, scalefactor=1 / 255.0, size=(224, 224), swapRB=True)
    net.setInput(blob)
    out = net.forward()
    idx = int(np.argmax(out))
    return CLASSES[idx]


def compute_shape_features(contour):
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    circularity = 0.0
    if perim > 0:
        circularity = 4 * np.pi * area / (perim * perim)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1)

    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / max(h, 1)
    rect_area = w * h
    extent = area / max(rect_area, 1)

    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    long_side = max(rw, rh)
    short_side = max(min(rw, rh), 1)
    elongation = long_side / short_side

    return {
        "area": area,
        "perim": perim,
        "circularity": circularity,
        "solidity": solidity,
        "aspect": aspect,
        "extent": extent,
        "elongation": elongation,
    }


def classify_heuristic(contour):
    # Shape-only heuristic (tune thresholds per video).
    features = compute_shape_features(contour)
    area = features["area"]
    aspect = features["aspect"]
    extent = features["extent"]
    solidity = features["solidity"]
    circularity = features["circularity"]
    elongation = features["elongation"]

    # Bottle-like shape: tall/elongated.
    bottle_like = elongation > BOTTLE_ELONGATION_MIN and aspect < BOTTLE_ASPECT_MAX

    if bottle_like:
        return "bottle"

    # Can: compact and round.
    if circularity > CAN_CIRCULARITY_MIN and elongation < CAN_ELONGATION_MAX:
        return "can"

    return "can"


class SimpleTracker:
    def __init__(self, max_distance=TRACK_MAX_DISTANCE, max_age=TRACK_MAX_AGE):
        self.max_distance = max_distance
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, frame_idx):
        # detections: list of (cx, cy, label, conf)
        results = []
        used_tracks = set()

        for cx, cy, label, conf in detections:
            best_id = None
            best_dist = self.max_distance + 1

            for tid, tr in self.tracks.items():
                if tid in used_tracks:
                    continue
                dist = float(np.hypot(cx - tr["cx"], cy - tr["cy"]))
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None and best_dist <= self.max_distance:
                tr = self.tracks[best_id]
                tr["cx"], tr["cy"] = cx, cy
                tr["last_seen"] = frame_idx
                if tr.get("locked_label") is None:
                    tr["labels"][label] = tr["labels"].get(label, 0) + 1
                    tr["last_label"] = label
                    if conf >= LOCK_CONF_THRESHOLD:
                        tr["locked_label"] = label
                used_tracks.add(best_id)
                stable = tr.get("locked_label") or max(tr["labels"].items(), key=lambda kv: kv[1])[0]
                results.append((cx, cy, stable, conf, best_id))
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "cx": cx,
                    "cy": cy,
                    "last_seen": frame_idx,
                    "labels": {label: 1},
                    "last_label": label,
                    "locked_label": label if conf >= LOCK_CONF_THRESHOLD else None,
                }
                used_tracks.add(tid)
                results.append((cx, cy, label, conf, tid))

        stale = [tid for tid, tr in self.tracks.items() if frame_idx - tr["last_seen"] > self.max_age]
        for tid in stale:
            del self.tracks[tid]

        return results


def is_reasonable_bbox(w, h, frame_shape):
    fh, fw = frame_shape[:2]
    if fw <= 0 or fh <= 0:
        return False
    if w <= 0 or h <= 0:
        return False
    if w > fw * MAX_DIM_FRACTION or h > fh * MAX_DIM_FRACTION:
        return False
    if (w * h) > (fw * fh * MAX_AREA_FRACTION):
        return False
    return True


def load_yolo(model_path):
    if model_path is None:
        return None
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics not installed. Install with: pip install ultralytics"
        ) from exc
    return YOLO(str(model_path))


def run_yolo(model, frame_bgr, imgsz=640, conf=0.25, iou=0.7):
    # Returns list of (x, y, w, h, label, conf)
    results = model.predict(frame_bgr, verbose=False, imgsz=imgsz, conf=conf, iou=iou)
    if not results:
        return []
    det = results[0]
    boxes = det.boxes
    if boxes is None or len(boxes) == 0:
        return []

    out = []
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy.tolist()
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
        conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0

        if cls_id >= 0 and cls_id < len(det.names):
            label = det.names[cls_id]
        else:
            label = "unknown"

        out.append((x1, y1, w, h, label, conf))
    return out


def parse_name_list(value):
    if not value:
        return []
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def map_yolo_label(raw_label, bottle_names, can_names):
    l = raw_label.lower()
    if any(name in l for name in bottle_names):
        return "bottle"
    if any(name in l for name in can_names):
        return "can"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Absolute path to video file")
    parser.add_argument("--show", action="store_true", help="Show debug window")
    parser.add_argument("--model", help="Optional ONNX classifier path")
    parser.add_argument("--yolo-model", help="Optional YOLOv8 model path (.pt)")
    parser.add_argument("--no-track", action="store_true", help="Disable simple tracking")
    parser.add_argument("--yolo-bottle-names", default="bottle",
                        help="Comma-separated substrings for YOLO bottle class names")
    parser.add_argument("--yolo-can-names", default="can",
                        help="Comma-separated substrings for YOLO can class names")
    parser.add_argument("--yolo-imgsz", type=int, default=640,
                        help="YOLO inference image size")
    parser.add_argument("--yolo-conf", type=float, default=0.2,
                        help="YOLO confidence threshold")
    parser.add_argument("--yolo-iou", type=float, default=0.6,
                        help="YOLO NMS IoU threshold")
    parser.add_argument("--debug-features", action="store_true",
                        help="Overlay shape features on detections")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    net = load_classifier(Path(args.model)) if args.model else None
    yolo = load_yolo(Path(args.yolo_model)) if args.yolo_model else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video.")

    if yolo is None and net is None:
        raise RuntimeError("No model provided. Use --yolo-model or --model.")

    backsub = None
    if net is not None:
        backsub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=False)
    tracker = SimpleTracker() if not args.no_track else None
    bottle_names = parse_name_list(args.yolo_bottle_names)
    can_names = parse_name_list(args.yolo_can_names)

    frame_idx = 0
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            results = []

            if yolo is not None:
                detections = run_yolo(
                    yolo,
                    frame,
                    imgsz=args.yolo_imgsz,
                    conf=args.yolo_conf,
                    iou=args.yolo_iou,
                )
                mapped = []
                for x, y, w, h, raw_label, conf in detections:
                    if w == 0 or h == 0:
                        continue
                    if not is_reasonable_bbox(w, h, frame.shape):
                        continue
                    label = map_yolo_label(raw_label, bottle_names, can_names)
                    if label == "unknown":
                        continue
                    # Compute approximate features from bbox for debug only.
                    if args.debug_features and args.show:
                        aspect = w / max(h, 1)
                        # Approximate circularity using bbox as a proxy (not contour-accurate).
                        perim = 2 * (w + h)
                        area = w * h
                        circularity = 0.0 if perim == 0 else (4 * np.pi * area) / (perim * perim)
                    cx = x + w // 2
                    cy = y + h // 2
                    mapped.append((cx, cy, label, conf))

                    if args.show:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        if args.debug_features:
                            text = f"asp={aspect:.2f} circ~={circularity:.2f}"
                            cv2.putText(frame, text, (x, y + h + 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                results.extend(mapped)
            elif net is not None:
                fg = backsub.apply(frame)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
                fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel, iterations=1)

                contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < MIN_AREA:
                        continue

                    x, y, w, h = cv2.boundingRect(cnt)
                    if not is_reasonable_bbox(w, h, frame.shape):
                        continue
                    cx = x + w // 2
                    cy = y + h // 2

                    roi = frame[y:y + h, x:x + w]

                    label = classify_with_model(net, roi)

                    results.append((cx, cy, label, 0.0))

                    if args.show:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, label, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        if args.debug_features:
                            feats = compute_shape_features(cnt)
                            text = f"asp={feats['aspect']:.2f} circ={feats['circularity']:.2f}"
                            cv2.putText(frame, text, (x, y + h + 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            if results and tracker is not None:
                results = tracker.update(results, frame_idx)

            if results:
                for item in results:
                    if tracker is not None:
                        cx, cy, label, conf, tid = item
                    else:
                        cx, cy, label, conf = item
                        tid = None
                    if conf > 0:
                        track_msg = f" track={tid}" if tid is not None else ""
                        print(
                            f"frame={frame_idx} centroid=({cx},{cy}) class={label} conf={conf:.2f}{track_msg}"
                        )
                    else:
                        track_msg = f" track={tid}" if tid is not None else ""
                        print(f"frame={frame_idx} centroid=({cx},{cy}) class={label}{track_msg}")

        if args.show:
            cv2.imshow("belt", frame)
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key == 27:
                break
            if key == ord('p'):
                paused = not paused

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
