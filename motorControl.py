import signal
import time
import sys
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DEMO_DIR = CURRENT_DIR / "Motor Control" / "board_demo"
# CAMERA_DIR = CURRENT_DIR / "cameraCode"
CV_CODE_DIR = CURRENT_DIR / "Computer Vision" / "code"
IK_DIR = CURRENT_DIR / "Inverse Kinematics"

# Add path to access board SDK, camera module, and CV helpers
sys.path.insert(0, str(MOTOR_DEMO_DIR))
# sys.path.insert(0, str(CAMERA_DIR))
sys.path.insert(0, str(CV_CODE_DIR))
sys.path.insert(0, str(IK_DIR))

import ros_robot_controller_sdk as rrc
import takePhoto
import inverseKinematics
from belt_objects import load_yolo, parse_class_aliases, detect_objects_in_frame, map_yolo_label

# Global control variables
running = True
servo_ids = [3, 5, 7]  # IDs of the three bus servos in IK order
MOVE_DURATION_S = 0.3
SERVO_ZERO_OFFSETS_DEG = {3: 87.0, 5: 90.0, 7: 90.0}
SERVO_DIRECTIONS = {3: -1.0, 5: -1.0, 7: -1.0}
SERVO_ANGLE_SCALES = {3: 1.0, 5: 1.0, 7: 1.0}
SERVO_STARTUP_ENABLE_DELAY_S = 0.35

# Board initialization
board = None
try:
    print("Initializing board...")
    board = rrc.Board()
    board.enable_reception()
    print("Board initialized successfully")
except Exception as e:
    print(f"WARNING: Could not initialize board: {e}")
    print("Continuing in camera-only mode (servos disabled)")

# Camera initialization
camera = None
target_format = None
cv_model = None
cv_class_aliases = None
cv_homography = None
cv_ready = False
rf_client = None

# CV inference config
CV_BACKEND = "roboflow_workflow"  # roboflow_workflow | yolo_local
CV_MODEL_PATH = CURRENT_DIR / "Computer Vision" / "trials" / "trial5-manual-auto" / "weights" / "best.pt"
CV_CONF = 0.35
CV_IOU = 0.6
CV_IMGSZ = 640
CV_TRACKER_TYPE = "byte"
CV_BYTE_TRACK_CONFIG = "bytetrack.yaml"
CV_USE_HOMOGRAPHY = True
CV_HOMOGRAPHY_UNITS = "mm"
CV_HOMOGRAPHY_SRC = "0,681;0,0;1079,681;1079,0"
CV_HOMOGRAPHY_DST = "-254,-152.4;-254,152.4;254,-152.4;254,152.4"
CV_CENTROID_MODE = "refined"
CV_ENABLE_SIX_PACK_HEURISTIC = False
CV_TARGET_MODE = "belt_order"  # belt_order | nearest_center | highest_confidence
CV_CONTROL_MODE = "world_ik"  # world_ik | centroid
CV_TARGET_Z_MM = -660.0
CV_WORLD_X_BIAS_MM = 0.0
CV_WORLD_Y_BIAS_MM = 0.0
CV_FALLBACK_TO_CENTROID = False
CV_Y_ONLY_MODE = True
CV_ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
CV_ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
CV_ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "leos-workspace-qswhy")
CV_ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "yolov11")
CV_ROBOFLOW_INPUT_NAME = os.getenv("ROBOFLOW_INPUT_NAME", "image")
CV_ROBOFLOW_USE_CACHE = os.getenv("ROBOFLOW_USE_CACHE", "true").lower() in ("1", "true", "yes")
CV_WORKFLOW_IMAGE_PATH = CURRENT_DIR / "cameraCode" / "photos" / "default.jpg"

LOOP_DELAY_S = 0.03

# Servo mapping config
SERVO_NEUTRAL = 120.0
SERVO_DELTA_MAX = 60.0
TARGET_CLASS_PRIORITY = {"bottle": 0, "can": 1, "six_pack": 2}


def stop_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\nStopping...")
    running = False


signal.signal(signal.SIGINT, stop_handler)


def initialize_camera():
    try:
        print("Initializing camera...")
        camera, target_format = takePhoto.initialzeCamera()
        print("Camera initialized successfully")
        return camera, target_format
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return None, None


def capture_photo(camera, target_format):
    try:
        if camera is None or target_format is None:
            return None
        
        bgr = takePhoto.takePhoto(camera, target_format, save_photo=True)
        
        if bgr is None:
            print(f"Photo not taken")
            return 
        else:
            return bgr
        
        
    except Exception as e:
        print(f"Error capturing photo: {e}")
        return None


def get_next_frame():
    return capture_photo(camera, target_format)


def _parse_points(value):
    if not value:
        return np.zeros((0, 2), dtype=np.float32)
    pairs = [p.strip() for p in value.split(";") if p.strip()]
    pts = []
    for pair in pairs:
        x_str, y_str = [v.strip() for v in pair.split(",")]
        pts.append((float(x_str), float(y_str)))
    return np.array(pts, dtype=np.float32)


def _parse_xy_pair(value):
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected pair 'x,y', got: {value}")
    return (float(parts[0]), float(parts[1]))


def initialize_cv():
    global cv_model, cv_class_aliases, cv_homography, cv_ready, rf_client
    try:
        cv_class_aliases = parse_class_aliases(
            "bottle",
            "can",
            "6-pack,six-pack,six_pack,6pack",
        )
        if CV_BACKEND == "yolo_local":
            if not CV_MODEL_PATH.exists():
                print(f"[CV] Model not found: {CV_MODEL_PATH}")
                cv_ready = False
                return
            cv_model = load_yolo(CV_MODEL_PATH)
        elif CV_BACKEND == "roboflow_workflow":
            if InferenceHTTPClient is None:
                print("[CV] inference_sdk is not installed")
                cv_ready = False
                return
            if not CV_ROBOFLOW_API_KEY:
                print("[CV] ROBOFLOW_API_KEY is not set")
                cv_ready = False
                return
            rf_client = InferenceHTTPClient(
                api_url=CV_ROBOFLOW_API_URL,
                api_key=CV_ROBOFLOW_API_KEY,
            )
        else:
            raise ValueError(f"Unsupported CV_BACKEND: {CV_BACKEND}")
        if CV_USE_HOMOGRAPHY:
            src = _parse_points(CV_HOMOGRAPHY_SRC)
            dst = _parse_points(CV_HOMOGRAPHY_DST)
            if len(src) >= 4 and len(src) == len(dst):
                h, _ = cv2.findHomography(src, dst, method=0)
                cv_homography = h
        cv_ready = True
        if CV_BACKEND == "yolo_local":
            print(f"[CV] Initialized YOLO model: {CV_MODEL_PATH}")
        else:
            print(
                f"[CV] Initialized Roboflow workflow backend: "
                f"{CV_ROBOFLOW_WORKSPACE}/{CV_ROBOFLOW_WORKFLOW_ID}"
            )
    except Exception as e:
        print(f"[CV] Initialization failed: {e}")
        cv_ready = False


def _project_world_from_centroid(cx, cy):
    if cv_homography is None:
        return None
    vec = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(vec, cv_homography)
    return (float(mapped[0, 0, 0]), float(mapped[0, 0, 1]))


def _normalize_label(label):
    return map_yolo_label(label, cv_class_aliases) if cv_class_aliases is not None else str(label)


def _coerce_points_list(node):
    if not isinstance(node, list):
        return None
    points = []
    for item in node:
        if isinstance(item, dict):
            x = item.get("x")
            y = item.get("y")
            if x is None or y is None:
                return None
            points.append((float(x), float(y)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((float(item[0]), float(item[1])))
        else:
            return None
    return points if len(points) >= 3 else None


def _extract_polygon_contour(prediction):
    candidates = [
        prediction.get("points"),
        prediction.get("polygon"),
        prediction.get("vertices"),
    ]
    for candidate in candidates:
        points = _coerce_points_list(candidate)
        if points is not None:
            contour = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
            return contour.astype(np.int32)

    mask = prediction.get("mask")
    if isinstance(mask, dict):
        for key in ("points", "polygon", "vertices"):
            points = _coerce_points_list(mask.get(key))
            if points is not None:
                contour = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
                return contour.astype(np.int32)
    return None


def _contour_to_bbox_centroid(contour):
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    cx = int(round(moments["m10"] / moments["m00"]))
    cy = int(round(moments["m01"] / moments["m00"]))
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, x + w, y + h), (cx, cy)


def _workflow_prediction_to_detection(prediction):
    label = _normalize_label(prediction.get("class") or prediction.get("label") or "unknown")
    confidence = float(prediction.get("confidence", prediction.get("score", 0.0)))
    contour = _extract_polygon_contour(prediction)
    if contour is not None:
        parsed = _contour_to_bbox_centroid(contour)
        if parsed is None:
            return None
        (x1, y1, x2, y2), (cx, cy) = parsed
        mask_polygon = contour
    else:
        x = prediction.get("x")
        y = prediction.get("y")
        w = prediction.get("width", prediction.get("w"))
        h = prediction.get("height", prediction.get("h"))

        if all(v is not None for v in (x, y, w, h)):
            x1 = int(round(float(x) - (float(w) / 2.0)))
            y1 = int(round(float(y) - (float(h) / 2.0)))
            x2 = int(round(float(x) + (float(w) / 2.0)))
            y2 = int(round(float(y) + (float(h) / 2.0)))
            cx = int(round(float(x)))
            cy = int(round(float(y)))
        else:
            x1 = prediction.get("x1", prediction.get("left"))
            y1 = prediction.get("y1", prediction.get("top"))
            x2 = prediction.get("x2", prediction.get("right"))
            y2 = prediction.get("y2", prediction.get("bottom"))
            if any(v is None for v in (x1, y1, x2, y2)):
                return None
            x1 = int(round(float(x1)))
            y1 = int(round(float(y1)))
            x2 = int(round(float(x2)))
            y2 = int(round(float(y2)))
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
        mask_polygon = None

    world_xy = _project_world_from_centroid(cx, cy)
    return {
        "bbox_xyxy": (x1, y1, x2, y2),
        "centroid": (cx, cy),
        "class": label,
        "confidence": confidence,
        "track_id": None,
        "world": world_xy,
        "world_units": CV_HOMOGRAPHY_UNITS if world_xy is not None else None,
        "mask_polygon": mask_polygon,
    }


def _collect_predictions(node, out):
    if isinstance(node, dict):
        preds = node.get("predictions")
        if isinstance(preds, list):
            for pred in preds:
                if isinstance(pred, dict):
                    out.append(pred)
        for value in node.values():
            _collect_predictions(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_predictions(item, out)


def _detect_with_roboflow_workflow(frame_bgr):
    CV_WORKFLOW_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(CV_WORKFLOW_IMAGE_PATH), frame_bgr):
        raise RuntimeError(f"Failed to write workflow input image: {CV_WORKFLOW_IMAGE_PATH}")
    data = rf_client.run_workflow(
        workspace_name=CV_ROBOFLOW_WORKSPACE,
        workflow_id=CV_ROBOFLOW_WORKFLOW_ID,
        images={CV_ROBOFLOW_INPUT_NAME: str(CV_WORKFLOW_IMAGE_PATH)},
        use_cache=CV_ROBOFLOW_USE_CACHE,
    )
    predictions = []
    _collect_predictions(data, predictions)
    detections = []
    for prediction in predictions:
        det = _workflow_prediction_to_detection(prediction)
        if det is not None and det["class"] != "unknown":
            detections.append(det)
    detections.sort(key=lambda det: det.get("confidence", 0.0), reverse=True)
    return detections


def _detect_with_yolo(frame_bgr):
    return detect_objects_in_frame(
        frame_bgr=frame_bgr,
        yolo_model=cv_model,
        class_aliases=cv_class_aliases,
        imgsz=CV_IMGSZ,
        conf=CV_CONF,
        iou=CV_IOU,
        tracker_type=CV_TRACKER_TYPE,
        byte_track_config=CV_BYTE_TRACK_CONFIG,
        centroid_mode=CV_CENTROID_MODE,
        enable_six_pack_heuristic=CV_ENABLE_SIX_PACK_HEURISTIC,
        homography=cv_homography,
        homography_units=CV_HOMOGRAPHY_UNITS,
    )


def _detect_with_cv(frame_bgr):
    if CV_BACKEND == "roboflow_workflow":
        return _detect_with_roboflow_workflow(frame_bgr)
    return _detect_with_yolo(frame_bgr)
def _choose_by_pixel_centroid(detections, centroid_xy):
    if not detections:
        return None
    cx0, cy0 = centroid_xy
    return sorted(
        detections,
        key=lambda d: (float(np.hypot(d["centroid"][0] - cx0, d["centroid"][1] - cy0)), -d.get("confidence", 0.0)),
    )[0]


def _annotate_target(frame_bgr, detections, target, requested_centroid=None):
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    frame_center = (int(w / 2), int(h / 2))
    cv2.circle(out, frame_center, 6, (255, 255, 0), 2)

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cx, cy = det["centroid"]
        label = det["class"]
        score = det["confidence"]
        color = (0, 200, 255) if det is target else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if det.get("mask_polygon") is not None:
            cv2.polylines(out, [det["mask_polygon"]], True, color, 1)
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            out,
            f"{label} {score:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
        )
    if target is not None:
        tcx, tcy = target["centroid"]
        cv2.circle(out, (tcx, tcy), 9, (0, 255, 255), 2)
        cv2.line(out, frame_center, (tcx, tcy), (0, 255, 255), 1)
    if requested_centroid is not None:
        rcx, rcy = int(requested_centroid[0]), int(requested_centroid[1])
        cv2.circle(out, (rcx, rcy), 6, (255, 0, 255), 2)
    return out


def _choose_target(detections, frame_center):
    if not detections:
        return None
    cx0, cy0 = frame_center

    if CV_TARGET_MODE == "belt_order":
        return detections[0]
    if CV_TARGET_MODE == "highest_confidence":
        return sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)[0]

    def key_fn(det):
        cls = det.get("class", "can")
        pr = TARGET_CLASS_PRIORITY.get(cls, 99)
        cx, cy = det["centroid"]
        dist = float(np.hypot(cx - cx0, cy - cy0))
        return (pr, dist, -det.get("confidence", 0.0))

    return sorted(detections, key=key_fn)[0]


def _centroid_to_servo_angles(cx, cy, width, height):
    if width <= 0 or height <= 0:
        return {3: SERVO_NEUTRAL, 5: SERVO_NEUTRAL, 7: SERVO_NEUTRAL}
    dx_norm = (cx - (width / 2.0)) / max(width / 2.0, 1.0)
    dy_norm = (cy - (height / 2.0)) / max(height / 2.0, 1.0)
    if CV_Y_ONLY_MODE:
        dx_norm = 0.0

    angle_3 = SERVO_NEUTRAL + (dx_norm * SERVO_DELTA_MAX)
    angle_4 = SERVO_NEUTRAL - (dy_norm * SERVO_DELTA_MAX)
    angle_5 = SERVO_NEUTRAL

    return {
        3: float(np.clip(angle_3, 0, 240)),
        5: float(np.clip(angle_4, 0, 240)),
        7: float(np.clip(angle_5, 0, 240)),
    }


def _world_to_servo_angles(world_xy):
    if world_xy is None:
        return None
    cam_x, cam_y = world_xy
    # Camera frame is mounted with +x up the square and +y to the right,
    # while robot +x is right and robot +y is up. That means swap axes.
    x = float(cam_y) + CV_WORLD_X_BIAS_MM
    y = float(cam_x) + CV_WORLD_Y_BIAS_MM
    if CV_Y_ONLY_MODE:
        x = 0.0
    z = float(CV_TARGET_Z_MM)
    print(f"getting angles for (x:{x},y:{y},z:{z})")
    angles = inverseKinematics.getAngles(x, y, z)
    if angles is None:
        return None
    return {7: float(angles[0]), 3: float(angles[1]), 5: float(angles[2])}


def analyze_photo(image):
    try:
        if image is None:
            print("[ANALYSIS] No image to analyze")
            return None, {3: 120, 5: 120, 7: 120}

        image_with_centroid = image.copy()
        neutral = {3: SERVO_NEUTRAL, 5: SERVO_NEUTRAL, 7: SERVO_NEUTRAL}

        if not cv_ready:
            print("[ANALYSIS] CV not initialized, using neutral servo positions")
            return image_with_centroid, neutral

        detections = _detect_with_cv(image)

        h, w = image.shape[:2]
        frame_center = (w / 2.0, h / 2.0)

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            cx, cy = det["centroid"]
            label = det["class"]
            score = det["confidence"]
            cv2.rectangle(image_with_centroid, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if det.get("mask_polygon") is not None:
                cv2.polylines(image_with_centroid, [det["mask_polygon"]], True, (0, 200, 255), 1)
            cv2.circle(image_with_centroid, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                image_with_centroid,
                f"{label} {score:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1,
            )

        target = _choose_target(detections, frame_center)
        if target is None:
            print("[ANALYSIS] No CV detections, using neutral servo positions")
            return image_with_centroid, neutral

        cx, cy = target["centroid"]
        world = target.get("world")

        servo_positions = None
        if CV_CONTROL_MODE == "world_ik":
            if world is not None:
                servo_positions = _world_to_servo_angles(world)
                if servo_positions is None:
                    print(
                        f"[ANALYSIS] world IK invalid at world=({world[0]:.1f},{world[1]:.1f}) "
                        f"z={CV_TARGET_Z_MM:.1f} mm"
                    )
            if servo_positions is None and not CV_FALLBACK_TO_CENTROID:
                print("[ANALYSIS] No valid world x,y target; skipping movement")
                return image_with_centroid, neutral
        if servo_positions is None:
            servo_positions = _centroid_to_servo_angles(cx, cy, w, h)

        cv2.circle(image_with_centroid, (int(frame_center[0]), int(frame_center[1])), 6, (255, 255, 0), 2)
        cv2.circle(image_with_centroid, (cx, cy), 8, (0, 255, 255), 2)
        cv2.line(
            image_with_centroid,
            (int(frame_center[0]), int(frame_center[1])),
            (cx, cy),
            (0, 255, 255),
            1,
        )

        world_msg = f", world=({world[0]:.1f},{world[1]:.1f}) {target.get('world_units')}" if world else ""
        print(
            f"[ANALYSIS] target={target['class']} centroid=({cx},{cy}) conf={target['confidence']:.2f}"
            f"{world_msg} mode={CV_CONTROL_MODE} -> angles: {servo_positions}"
        )
        return image_with_centroid, servo_positions

    except Exception as e:
        print(f"Error analyzing photo: {e}")
        return image, {3: 120, 5: 120, 7: 120}


def move_servos(servo_positions):
    if board is None:
        print("Board not initialized, skipping servo movement")
        return
    
    try:
        # Convert angles to servo positions and move servos
        servo_commands = []
        for servo_id in servo_ids:
            if servo_id in servo_positions:
                logical_angle = float(servo_positions[servo_id])
                direction = SERVO_DIRECTIONS.get(servo_id, 1.0)
                scale = SERVO_ANGLE_SCALES.get(servo_id, 1.0)
                physical_angle = SERVO_ZERO_OFFSETS_DEG.get(servo_id, 0.0) + (direction * scale * logical_angle)
                physical_angle = max(0.0, min(240.0, physical_angle))
                raw_pos = int((physical_angle / 240.0) * 1000)
                servo_commands.append([servo_id, raw_pos])
        
        if servo_commands:
            print(f"Moving servos: {servo_commands}")
            board.bus_servo_set_position(MOVE_DURATION_S * 3, servo_commands)
            time.sleep(MOVE_DURATION_S * 3 + 0.05)
            
            # Read back positions for confirmation
            for servo_id in servo_ids:
                try:
                    pos = board.bus_servo_read_position(servo_id)
                    if pos:
                        print(f"Servo {servo_id} current position: {pos[0]}")
                except Exception as e:
                    print(f"Could not read position for servo {servo_id}: {e}")
    
    except Exception as e:
        print(f"Error moving servos: {e}")


if __name__ == '__main__':
    print(f"Servo IDs: {servo_ids}")
    parser = argparse.ArgumentParser(description="CV-driven arm control")
    parser.add_argument("--loop", action="store_true", help="Run continuous camera loop mode")
    parser.add_argument("--test-centroid", type=str, help="Select detection nearest to pixel centroid x,y")
    parser.add_argument("--photo-path", type=str, default=str(CURRENT_DIR / "cameraCode" / "photos" / "default.jpg"),
                        help="Output photo path for captured/annotated frame")
    parser.add_argument("--isaac", action="store_true", help="isaac method testing")
    args = parser.parse_args()
    
    class_aliases = parse_class_aliases(
        "bottle",
        "can",
        "6-pack,six-pack,six_pack,6pack",
    )

    try:
        initialize_cv()
        camera, target_format = initialize_camera()
        if camera is None or target_format is None:
            raise RuntimeError("Camera initialization failed")
        
        # Initialize servos - set to torque ON (0)
        if board is not None:
            print(f"Initializing servos sequentially (delay={SERVO_STARTUP_ENABLE_DELAY_S:.2f}s)...")
            for servo_id in servo_ids:
                try:
                    board.bus_servo_enable_torque(servo_id, 0)
                    print(f"Servo {servo_id} torque ON")
                    time.sleep(SERVO_STARTUP_ENABLE_DELAY_S)
                except Exception as e:
                    print(f"Warning: Could not enable servo {servo_id}: {e}")
        else:
            print("Board not initialized, skipping servo initialization")

        #  Isaac testing section DO NOT MODIFY -------------------------------------------------------
        TOTAL_PIXEL_WIDTH = 1920 # correct this
        TOTAL_PIXEL_HEIGHT = 1080 # correct this
        angle_offset = 45
        import math
        time.sleep(0.5)
        if args.isaac:
            while True:
                input("type enter to do test (set the object(s) wherever you want them)")

                photo_path = Path(args.photo_path)
                photo_path.parent.mkdir(parents=True, exist_ok=True)

                image = capture_photo(camera, target_format)
                if image is None:
                    raise RuntimeError("No image captured from camera")
                cv2.imwrite(str(photo_path), image)
                time.sleep(0.5)
                print(f"Captured and saved photo to {photo_path}")

                detections = _detect_with_cv(image)
                if not detections:
                    print("[PHOTO] No detections found in photo")
                    continue
                else:
                    print("[PHOTO] Detected objects, just testing first")
                    det = detections[0]
                    cx, cy = det["centroid"]
                    print(f"centroid coordinate: ({cx}, {cy})")
                    # This part recenters around middle being zero and then multiplies
                    # by a conversion to mm based on a rough estimation
                    res_width = 1100 # estimation that can be improved (in mm)
                    res_height = res_width * TOTAL_PIXEL_HEIGHT / TOTAL_PIXEL_WIDTH
                    cx = (cx - TOTAL_PIXEL_WIDTH / 2) * (res_width / TOTAL_PIXEL_WIDTH)
                    cy = (cy - TOTAL_PIXEL_HEIGHT / 2) * (res_height / TOTAL_PIXEL_HEIGHT)
                    z = -500-70 # about the height of the arm
                    print(f"PRE-ROTATION coordinates (x, y, z)=({cx},{cy},{z})")

                    # cx2 = cx * math.cos(math.radians(angle_offset)) - cy * math.sin(math.radians(angle_offset))
                    # cy2 = cy * math.cos(math.radians(angle_offset)) + cx * math.sin(math.radians(angle_offset))
                    # print(f"calculated coordinates (x, y, z)=({cx2},{cy2},{z})")
                    angles = inverseKinematics.getAngles(cx, cy, z)
                    print(f"calculated angles: {angles}")
                    # 4->3, 5->4, 3->5
                    servo_commands = {5: angles[0], 3: angles[1], 4: angles[2]}
                    move_servos(servo_commands)


        #  end Isaac testing section -------------------------------------------------------

        if not args.loop:
            while running:
                print("Photo-once mode: takePhoto -> detect centroid/world -> confirm move")
                photo_path = Path(args.photo_path)
                photo_path.parent.mkdir(parents=True, exist_ok=True)

                image = capture_photo(camera, target_format)
                if image is None:
                    raise RuntimeError("No image captured from camera")
                cv2.imwrite(str(photo_path), image)
                time.sleep(0.5)
                print(f"Captured and saved photo to {photo_path}")

                if not cv_ready:
                    raise RuntimeError("CV not initialized; cannot detect centroids")

                detections = _detect_with_cv(image)
                if not detections:
                    print("[PHOTO] No detections found in photo")
                    continue
                else:
                    print("[PHOTO] Detected objects:")
                    for i, det in enumerate(detections, start=1):
                        cx, cy = det["centroid"]
                        world = det.get("world")
                        world_msg = f" world=({world[0]:.2f},{world[1]:.2f}) {det.get('world_units')}" if world else ""
                        print(
                            f"  #{i}: class={det['class']} conf={det['confidence']:.2f} "
                            f"centroid=({cx},{cy}){world_msg}"
                        )

                    h, w = image.shape[:2]
                    centroid_xy = _parse_xy_pair(args.test_centroid) if args.test_centroid else None
                    if centroid_xy is not None:
                        target = _choose_by_pixel_centroid(detections, centroid_xy)
                    else:
                        target = _choose_target(detections, (w / 2.0, h / 2.0))
                    if target is None:
                        print("[PHOTO] No target selected")
                        continue
                    else:
                        cx, cy = target["centroid"]
                        world = target.get("world")
                        if CV_CONTROL_MODE == "world_ik":
                            servo_positions = _world_to_servo_angles(world) if world is not None else None
                            if servo_positions is None and not CV_FALLBACK_TO_CENTROID:
                                print("[PHOTO] Selected target has no valid world x,y -> skipping movement")
                                servo_positions = None
                            elif servo_positions is None:
                                servo_positions = _centroid_to_servo_angles(cx, cy, w, h)
                        else:
                            servo_positions = _centroid_to_servo_angles(cx, cy, w, h)

                        overlay = _annotate_target(image, detections, target)
                        cv2.imwrite(str(photo_path), overlay)
                        print(f"[PHOTO] Saved annotated photo to {photo_path}")

                        world_msg = f" world=({world[0]:.2f},{world[1]:.2f}) {target.get('world_units')}" if world else ""
                        print(
                            f"[PHOTO] Selected target class={target['class']} conf={target['confidence']:.2f} "
                            f"centroid=({cx},{cy}){world_msg} -> angles={servo_positions}"
                        )

                        if servo_positions is not None:
                            choice = input("Move arm to selected target? [y/N]: ").strip().lower()
                            if choice in ("y", "yes"):
                                move_servos(servo_positions)
                            else:
                                print("[PHOTO] Move canceled by user.")
                                choice = input("Do you want to stop the system [y/N]: ").strip().lower()
                                if choice in ("y", "yes"):
                                    running = False
                                else:
                                    continue
        else:
            print("Starting main loop (camera source). Press Ctrl+C to stop.")
            
            # Main loop - sequential frame processing for stable tracking
            frame_count = 0
            while running:
                try:
                    frame_count += 1
                    print(f"\n=== Frame {frame_count} ===")
                    image = get_next_frame()
                    
                    if image is not None:
                        # Analyze frame (returns image with overlay and servo positions)
                        image_with_centroid, servo_positions = analyze_photo(image)
                        
                        # Save image with centroid to photos directory
                        try:
                            photo_path = CURRENT_DIR / "cameraCode" / "photos" / "default.jpg"
                            photo_path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(photo_path), image_with_centroid)
                            print(f"Saved photo to {photo_path}")
                        except Exception as e:
                            print(f"Error saving photo: {e}")
                        
                        # Move servos from sequential CV outputs
                        move_servos(servo_positions)
                        time.sleep(LOOP_DELAY_S)
                    else:
                        print("No frame captured (camera issue)")
                        time.sleep(0.05)
                
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    time.sleep(0.5)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        running = False
    
    finally:
        print("\nShutting down...")
        
        # Stop all servos
        if board is not None:
            try:
                for servo_id in servo_ids:
                    board.bus_servo_enable_torque(servo_id, 1)  # Torque OFF
                    print(f"Servo {servo_id} torque OFF")
            except Exception as e:
                print(f"Error disabling servos: {e}")
        
        # Close camera
        if camera is not None:
            try:
                takePhoto.closeCamera(camera)
                print("Camera closed")
            except Exception as e:
                print(f"Error closing camera: {e}")
        
        print("Shutdown complete")
