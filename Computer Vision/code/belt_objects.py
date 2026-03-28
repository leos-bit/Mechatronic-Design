#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["bottle", "can", "six_pack"]

MIN_AREA = 500
TRACK_MAX_DISTANCE = 50
TRACK_MAX_AGE = 12
MAX_AREA_FRACTION = 0.4
MAX_DIM_FRACTION = 0.8
LOCK_CONF_THRESHOLD = 0.6
LOCK_MIN_HITS = 3
CENTROID_MASK_MIN_AREA = 0.08
CENTROID_BORDER_FRACTION = 0.12
CENTROID_THRESHOLD_BIAS = 10
SIX_PACK_MIN_AREA_FRACTION = 0.025
SIX_PACK_ASPECT_MIN = 1.2


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


def load_classifier(model_path):
    if model_path is None:
        return None
    net = cv2.dnn.readNetFromONNX(str(model_path))
    return net


def classify_with_model(net, roi_bgr):
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


class SimpleTracker:
    def __init__(
        self,
        max_distance=TRACK_MAX_DISTANCE,
        max_age=TRACK_MAX_AGE,
        lock_conf_threshold=LOCK_CONF_THRESHOLD,
        lock_min_hits=LOCK_MIN_HITS,
    ):
        self.max_distance = max_distance
        self.max_age = max_age
        self.lock_conf_threshold = lock_conf_threshold
        self.lock_min_hits = lock_min_hits
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, frame_idx):
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
                    tr["hits"] += 1
                    tr["labels"][label] = tr["labels"].get(label, 0) + 1
                    tr["label_scores"][label] = tr["label_scores"].get(label, 0.0) + conf
                    tr["label_max_conf"][label] = max(tr["label_max_conf"].get(label, 0.0), conf)
                    tr["last_label"] = label
                    stable, _, stable_ratio = choose_stable_label(tr)
                    stable_conf = tr["label_max_conf"].get(stable, 0.0)
                    if (
                        tr["hits"] >= self.lock_min_hits
                        and stable_ratio >= 0.65
                        and stable_conf >= self.lock_conf_threshold
                    ):
                        tr["locked_label"] = stable
                used_tracks.add(best_id)
                stable = tr.get("locked_label") or choose_stable_label(tr)[0]
                results.append((cx, cy, stable, conf, best_id))
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "cx": cx,
                    "cy": cy,
                    "last_seen": frame_idx,
                    "hits": 1,
                    "labels": {label: 1},
                    "label_scores": {label: conf},
                    "label_max_conf": {label: conf},
                    "last_label": label,
                    "locked_label": None,
                }
                used_tracks.add(tid)
                results.append((cx, cy, label, conf, tid))

        stale = [tid for tid, tr in self.tracks.items() if frame_idx - tr["last_seen"] > self.max_age]
        for tid in stale:
            del self.tracks[tid]

        return results


def choose_stable_label(track):
    score_items = track.get("label_scores", {})
    if not score_items:
        stable, stable_count = max(track["labels"].items(), key=lambda kv: kv[1])
        stable_ratio = stable_count / max(track.get("hits", 1), 1)
        return stable, float(stable_count), stable_ratio
    stable, stable_score = max(score_items.items(), key=lambda kv: (kv[1], track["labels"].get(kv[0], 0)))
    total_score = max(sum(score_items.values()), 1e-6)
    stable_ratio = stable_score / total_score
    return stable, stable_score, stable_ratio


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


def refine_centroid(frame_bgr, x, y, w, h, threshold_bias=CENTROID_THRESHOLD_BIAS):
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(frame_bgr.shape[1], int(x + w))
    y2 = min(frame_bgr.shape[0], int(y + h))
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return x + w // 2, y + h // 2, None

    rh, rw = roi.shape[:2]
    border = max(1, int(round(min(rh, rw) * CENTROID_BORDER_FRACTION)))
    border = min(border, max(1, min(rh, rw) // 3))

    border_mask = np.zeros((rh, rw), dtype=np.uint8)
    border_mask[:border, :] = 255
    border_mask[-border:, :] = 255
    border_mask[:, :border] = 255
    border_mask[:, -border:] = 255

    border_pixels = roi[border_mask == 255]
    if border_pixels.size == 0:
        return x + w // 2, y + h // 2, None

    bg_color = np.median(border_pixels.reshape(-1, 3), axis=0).astype(np.float32)
    diff = np.linalg.norm(roi.astype(np.float32) - bg_color, axis=2)
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    otsu_threshold, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    applied_threshold = int(np.clip(otsu_threshold + threshold_bias, 0, 255))
    _, mask = cv2.threshold(diff, applied_threshold, 255, cv2.THRESH_BINARY)

    kernel_size = max(3, int(round(min(rh, rw) * 0.08)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return x + w // 2, y + h // 2, None

    box_area = float(max(rw * rh, 1))
    roi_center = np.array([rw / 2.0, rh / 2.0], dtype=np.float32)
    best_contour = None
    best_score = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < box_area * CENTROID_MASK_MIN_AREA:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        dist = float(np.hypot(cx - roi_center[0], cy - roi_center[1]))
        score = (dist, -area)
        if best_score is None or score < best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return x + w // 2, y + h // 2, None

    moments = cv2.moments(best_contour)
    if moments["m00"] <= 0:
        return x + w // 2, y + h // 2, None

    cx = int(round(x1 + (moments["m10"] / moments["m00"])))
    cy = int(round(y1 + (moments["m01"] / moments["m00"])))
    contour_global = best_contour + np.array([[[x1, y1]]], dtype=np.int32)
    return cx, cy, contour_global


def classify_six_pack_from_geometry(label, w, h, frame_shape, min_area_fraction, min_aspect):
    if label != "can":
        return label
    fh, fw = frame_shape[:2]
    if fh <= 0 or fw <= 0:
        return label
    area_fraction = (w * h) / float(fw * fh)
    aspect = w / max(h, 1)
    if area_fraction >= min_area_fraction and aspect >= min_aspect:
        return "six_pack"
    return label


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


def run_yolo_bytetrack(model, frame_bgr, imgsz=640, conf=0.25, iou=0.7, tracker_cfg="bytetrack.yaml"):
    results = model.track(
        frame_bgr,
        persist=True,
        verbose=False,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        tracker=tracker_cfg,
    )
    if not results:
        return []
    det = results[0]
    boxes = det.boxes
    if boxes is None or len(boxes) == 0:
        return []

    out = []
    ids = boxes.id if hasattr(boxes, "id") else None
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy.tolist()
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
        conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
        track_id = int(ids[i].item()) if ids is not None else None

        if cls_id >= 0 and cls_id < len(det.names):
            label = det.names[cls_id]
        else:
            label = "unknown"

        out.append((x1, y1, w, h, label, conf, track_id))
    return out


def parse_points(value):
    if not value:
        return np.zeros((0, 2), dtype=np.float32)
    pairs = [p.strip() for p in value.split(";") if p.strip()]
    pts = []
    for pair in pairs:
        xy = [v.strip() for v in pair.split(",")]
        if len(xy) != 2:
            raise ValueError(f"Invalid point '{pair}'. Use x,y pairs separated by ';'.")
        pts.append((float(xy[0]), float(xy[1])))
    return np.array(pts, dtype=np.float32)


def load_homography_matrix(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Homography file not found: {p}")
    if p.suffix.lower() == ".npy":
        h = np.load(str(p))
    else:
        h = np.loadtxt(str(p), dtype=np.float64)
    h = np.array(h, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got shape {h.shape}")
    return h


def build_homography(args):
    if args.homography_matrix:
        return load_homography_matrix(args.homography_matrix)

    if args.homography_src or args.homography_dst:
        if not args.homography_src or not args.homography_dst:
            raise ValueError("Provide both --homography-src and --homography-dst.")
        src = parse_points(args.homography_src)
        dst = parse_points(args.homography_dst)
        if len(src) != len(dst):
            raise ValueError("Homography src/dst point counts must match.")
        if len(src) < 4:
            raise ValueError("Homography requires at least 4 point pairs.")
        h, _ = cv2.findHomography(src, dst, method=0)
        if h is None:
            raise RuntimeError("Failed to compute homography from provided points.")
        return h

    return None


def project_point(h, x, y):
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, h)
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def map_yolo_label(raw_label, class_aliases):
    label = str(raw_label).lower()
    for canonical, aliases in class_aliases.items():
        if any(alias in label for alias in aliases):
            return canonical
    return "unknown"


def detect_objects_in_frame(
    frame_bgr,
    yolo_model,
    class_aliases,
    imgsz=640,
<<<<<<< HEAD
    conf=0.20,
=======
    conf=0.35,
>>>>>>> 862fc2ea743f06ffae296121bb474039693f9401
    iou=0.6,
    tracker_type="simple",
    byte_track_config="bytetrack.yaml",
    centroid_mode="refined",
    centroid_threshold_bias=CENTROID_THRESHOLD_BIAS,
    enable_six_pack_heuristic=False,
    six_pack_min_area_fraction=SIX_PACK_MIN_AREA_FRACTION,
    six_pack_min_aspect=SIX_PACK_ASPECT_MIN,
    homography=None,
    homography_units="world",
):
    """
    Detect and classify objects in a single frame.

    Returns a list of dicts:
      {
        "centroid": (cx, cy),
        "class": "bottle|can|six_pack",
        "confidence": conf,
        "bbox_xyxy": (x1, y1, x2, y2),
        "track_id": id or None,
        "world": (X, Y) or None,
        "world_units": homography_units or None,
      }
    """
    if tracker_type == "byte":
        raw = run_yolo_bytetrack(
            yolo_model,
            frame_bgr,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            tracker_cfg=byte_track_config,
        )
        detections = [(x, y, w, h, raw_label, score, track_id) for x, y, w, h, raw_label, score, track_id in raw]
    else:
        raw = run_yolo(yolo_model, frame_bgr, imgsz=imgsz, conf=conf, iou=iou)
        detections = [(x, y, w, h, raw_label, score, None) for x, y, w, h, raw_label, score in raw]

    results = []
    for x, y, w, h, raw_label, score, track_id in detections:
        if w == 0 or h == 0:
            continue
        if not is_reasonable_bbox(w, h, frame_bgr.shape):
            continue

        label = map_yolo_label(raw_label, class_aliases)
        if label == "unknown" and enable_six_pack_heuristic:
            label = "can"
        if label == "unknown":
            continue
        if enable_six_pack_heuristic:
            label = classify_six_pack_from_geometry(
                label,
                w,
                h,
                frame_bgr.shape,
                min_area_fraction=six_pack_min_area_fraction,
                min_aspect=six_pack_min_aspect,
            )

        if centroid_mode == "refined":
            cx, cy, _ = refine_centroid(
                frame_bgr,
                x,
                y,
                w,
                h,
                threshold_bias=centroid_threshold_bias,
            )
        else:
            cx, cy = x + w // 2, y + h // 2

        x1, y1, x2, y2 = x, y, x + w, y + h
        world_xy = project_point(homography, cx, cy) if homography is not None else None
        results.append(
            {
                "centroid": (int(cx), int(cy)),
                "class": label,
                "confidence": float(score),
                "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
                "track_id": None if track_id is None else int(track_id),
                "world": world_xy,
                "world_units": homography_units if world_xy is not None else None,
            }
        )
    return results


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
    parser.add_argument("--yolo-six-pack-names", default="6-pack,six-pack,six_pack,6pack",
                        help="Comma-separated substrings for YOLO six-pack class names")
    parser.add_argument("--yolo-imgsz", type=int, default=640,
                        help="YOLO inference image size")
    parser.add_argument("--yolo-conf", type=float, default=0.35,
                        help="YOLO confidence threshold")
    parser.add_argument("--yolo-iou", type=float, default=0.6,
                        help="YOLO NMS IoU threshold")
    parser.add_argument("--tracker-type", choices=["simple", "byte"], default="simple",
                        help="Tracking backend for YOLO detections")
    parser.add_argument("--byte-track-config", default="bytetrack.yaml",
                        help="Ultralytics tracker config path/name for ByteTrack")
    parser.add_argument("--homography-matrix",
                        help="Path to 3x3 homography matrix (.npy or text)")
    parser.add_argument("--homography-src",
                        help="Source pixel points as 'x1,y1;x2,y2;...'")
    parser.add_argument("--homography-dst",
                        help="Destination world points as 'X1,Y1;X2,Y2;...'")
    parser.add_argument("--homography-units", default="world",
                        help="Label for mapped coordinates (e.g., cm, mm)")
    parser.add_argument("--track-lock-conf", type=float, default=LOCK_CONF_THRESHOLD,
                        help="Track label lock confidence threshold")
    parser.add_argument("--track-lock-min-hits", type=int, default=LOCK_MIN_HITS,
                        help="Minimum matched detections before locking a track label")
    parser.add_argument("--centroid-mode", choices=["bbox", "refined"], default="refined",
                        help="Use bbox center or a refined silhouette centroid inside each detection")
    parser.add_argument("--centroid-threshold-bias", type=int, default=CENTROID_THRESHOLD_BIAS,
                        help="Threshold bias for centroid refinement mask extraction")
    parser.add_argument("--enable-six-pack-heuristic", action="store_true",
                        help="Promote large, wide can detections to six_pack when using a 2-class model")
    parser.add_argument("--six-pack-min-area-fraction", type=float, default=SIX_PACK_MIN_AREA_FRACTION,
                        help="Minimum bbox area fraction of the frame to promote can -> six_pack")
    parser.add_argument("--six-pack-min-aspect", type=float, default=SIX_PACK_ASPECT_MIN,
                        help="Minimum width/height ratio to promote can -> six_pack")
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
    use_byte_track = (yolo is not None) and (args.tracker_type == "byte") and (not args.no_track)
    homography = build_homography(args)
    tracker = (
        SimpleTracker(
            lock_conf_threshold=args.track_lock_conf,
            lock_min_hits=args.track_lock_min_hits,
        )
        if (not args.no_track and not use_byte_track)
        else None
    )
    class_aliases = parse_class_aliases(
        args.yolo_bottle_names,
        args.yolo_can_names,
        args.yolo_six_pack_names,
    )

    frame_idx = 0
    paused = False
    frame = None
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        elif frame is None:
            continue

        if not paused:
            frame_idx += 1
            results = []

            if yolo is not None:
                if use_byte_track:
                    detections = run_yolo_bytetrack(
                        yolo,
                        frame,
                        imgsz=args.yolo_imgsz,
                        conf=args.yolo_conf,
                        iou=args.yolo_iou,
                        tracker_cfg=args.byte_track_config,
                    )
                else:
                    detections = run_yolo(
                        yolo,
                        frame,
                        imgsz=args.yolo_imgsz,
                        conf=args.yolo_conf,
                        iou=args.yolo_iou,
                    )
                mapped = []
                for det_item in detections:
                    if use_byte_track:
                        x, y, w, h, raw_label, conf, tid = det_item
                    else:
                        x, y, w, h, raw_label, conf = det_item
                        tid = None
                    if w == 0 or h == 0:
                        continue
                    if not is_reasonable_bbox(w, h, frame.shape):
                        continue
                    label = map_yolo_label(raw_label, class_aliases)
                    if label == "unknown" and args.enable_six_pack_heuristic:
                        label = "can"
                    if label == "unknown":
                        continue
                    if args.enable_six_pack_heuristic:
                        label = classify_six_pack_from_geometry(
                            label,
                            w,
                            h,
                            frame.shape,
                            min_area_fraction=args.six_pack_min_area_fraction,
                            min_aspect=args.six_pack_min_aspect,
                        )

                    if args.debug_features and args.show:
                        aspect = w / max(h, 1)
                        perim = 2 * (w + h)
                        area = w * h
                        circularity = 0.0 if perim == 0 else (4 * np.pi * area) / (perim * perim)

                    if args.centroid_mode == "refined":
                        cx, cy, centroid_contour = refine_centroid(
                            frame,
                            x,
                            y,
                            w,
                            h,
                            threshold_bias=args.centroid_threshold_bias,
                        )
                    else:
                        cx, cy, centroid_contour = x + w // 2, y + h // 2, None

                    if use_byte_track:
                        mapped.append((cx, cy, label, conf, tid))
                    else:
                        mapped.append((cx, cy, label, conf))

                    if args.show:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if centroid_contour is not None:
                            cv2.drawContours(frame, [centroid_contour], -1, (0, 200, 255), 1)
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        track_tag = f" id={tid}" if tid is not None else ""
                        cv2.putText(frame, f"{label} {conf:.2f}{track_tag}", (x, y - 5),
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
                    if tracker is not None or use_byte_track:
                        cx, cy, label, conf, tid = item
                    else:
                        cx, cy, label, conf = item
                        tid = None
                    world_xy = project_point(homography, cx, cy) if homography is not None else None
                    track_msg = f" track={tid}" if tid is not None else ""
                    world_msg = (
                        f" {args.homography_units}=({world_xy[0]:.2f},{world_xy[1]:.2f})"
                        if world_xy is not None
                        else ""
                    )
                    if conf > 0:
                        print(f"frame={frame_idx} centroid=({cx},{cy}) class={label} conf={conf:.2f}{track_msg}{world_msg}")
                    else:
                        print(f"frame={frame_idx} centroid=({cx},{cy}) class={label}{track_msg}{world_msg}")
                    if args.show and world_xy is not None:
                        cv2.putText(
                            frame,
                            f"{args.homography_units}=({world_xy[0]:.1f},{world_xy[1]:.1f})",
                            (cx + 6, cy + 14),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (0, 255, 255),
                            1,
                        )

        if args.show:
            cv2.imshow("belt", frame)
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key == 27:
                break
            if key == ord("p"):
                paused = not paused

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
