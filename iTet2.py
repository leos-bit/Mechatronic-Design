import signal
import time
import sys
import os
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DEMO_DIR = CURRENT_DIR / "Motor Control" / "board_demo"
CAMERA_DIR = CURRENT_DIR / "cameraCode"
CV_CODE_DIR = CURRENT_DIR / "Computer Vision" / "code"
IK_DIR = CURRENT_DIR / "Inverse Kinematics"

# Add path to access board SDK, camera module, and CV helpers
sys.path.insert(0, str(MOTOR_DEMO_DIR))
sys.path.insert(0, str(CAMERA_DIR))
sys.path.insert(0, str(CV_CODE_DIR))
sys.path.insert(0, str(IK_DIR))
import suctionControl
import ros_robot_controller_sdk as rrc
import takePhoto
import inverseKinematics
from belt_objects import load_yolo, parse_class_aliases, detect_objects_in_frame, map_yolo_label

# Global control variables
running = True
servo_ids = [1, 2, 3]  # IDs of the three bus servos in IK order
MOVE_DURATION_S = 0.3
SERVO_ZERO_OFFSETS_DEG = {1: 90.0, 2: 90.0, 3: 90.0}
SERVO_DIRECTIONS = {1: -1.0, 2: -1.0, 3: -1.0}
SERVO_ANGLE_SCALES = {1: 1.0, 2: 1.0, 3: 1.0}
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
CV_HOMOGRAPHY_SRC = "493.0,998.0;588.0,91.0;1500.0,112.0;1573.0,918.0;1006.0,461.0"
CV_HOMOGRAPHY_DST = "305.0,-305.0;-290.0,-305.0;-290.0,305.0;305.0,305.0;0.0,0.0"
CV_CENTROID_MODE = "refined"
CV_ENABLE_SIX_PACK_HEURISTIC = False
CV_TARGET_MODE = "belt_order"  # belt_order | nearest_center | highest_confidence
CV_CONTROL_MODE = "world_ik"  # world_ik | centroid
CV_TARGET_Z_MM = -660.0
CV_WORLD_SCALE_X = 1.0
CV_WORLD_SCALE_Y = 1.0
CV_WORLD_X_BIAS_MM = 0.0
CV_WORLD_Y_BIAS_MM = 0.0
CV_FALLBACK_TO_CENTROID = False
CV_Y_ONLY_MODE = False
CV_ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
CV_ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "oPusoqJbAhSfo6zbicdc")
CV_ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "leos-workspace-qswhy")
CV_ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "yolov11")
CV_ROBOFLOW_INPUT_NAME = os.getenv("ROBOFLOW_INPUT_NAME", "image")
CV_ROBOFLOW_USE_CACHE = os.getenv("ROBOFLOW_USE_CACHE", "false").lower() in ("1", "true", "yes")
CV_WORKFLOW_IMAGE_PATH = CURRENT_DIR / "cameraCode" / "photos" / "default.jpg"

LOOP_DELAY_S = 0.03
SUCTION_Y_OFFSET_MM = 0.0
CONVEYOR_Y_OFFSET_MM = 200.0
DETECTION_MAX_FRAMES = 10
DETECTION_FRAME_DELAY_S = 0.05
PICK_Z_MM = -710.0
PICK_Z_SEARCH_STEP_MM = 5.0
STARTUP_HOME_X_MM = 0.0
STARTUP_HOME_Y_MM = 0.0
STARTUP_HOME_Z_MM = -550.0
Z_XY_CORRECTION_FACTOR = 0.18

# Servo mapping config
SERVO_NEUTRAL = 120.0
SERVO_DELTA_MAX = 60.0
TARGET_CLASS_PRIORITY = {"bottle": 0, "can": 1, "six_pack": 2}


def pixel_centroid_from_bbox(x1, y1, x2, y2):
    return (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))


def center_relative_centroid(pixel_centroid, frame_shape):
    height, width = frame_shape[:2]
    cx, cy = pixel_centroid
    image_x = cx - (width / 2.0)
    image_y = cy - (height / 2.0)
    return (
        int(round(image_y)),
        int(round(-image_x)),
    )


def logical_angles_to_servo_positions(angles):
    return {
        servo_id: float(angle)
        for servo_id, angle in zip(servo_ids, angles)
    }


def ik_angles_to_raw_servo_commands(angles_deg):
    angle_factor = 1000.0 / 240.0
    return [
        [servo_id, int(angle_deg * angle_factor)]
        for servo_id, angle_deg in zip(servo_ids, angles_deg)
    ]


def compute_servo_angles_for_xyz(x, y, z, label):
    angles = inverseKinematics.getAngles(x, y, z)
    if angles is None:
        print(f"No IK solution for {label} at x={x:.1f}, y={y:.1f}, z={z:.1f}")
        return None
    return [90 - v for v in angles]


def find_deepest_feasible_z(x, y, preferred_z, safe_z, step_mm=PICK_Z_SEARCH_STEP_MM):
    trial_z = float(preferred_z)
    safe_z = float(safe_z)
    step_mm = abs(float(step_mm))
    if step_mm <= 0:
        step_mm = 5.0

    while trial_z <= safe_z:
        if inverseKinematics.getAngles(x, y, trial_z) is not None:
            return trial_z
        trial_z += step_mm
    return None


def apply_xy_z_adjustment(x, y, z):
    eX = Z_XY_CORRECTION_FACTOR * abs(x)
    eY = Z_XY_CORRECTION_FACTOR * abs(y)
    eXY = float(np.hypot(eX, eY))
    adjusted_z = float(z) + eXY
    print(
        f"z xy adjustment: eX={eX:.1f}, eY={eY:.1f}, eXY={eXY:.1f}, "
        f"z {float(z):.1f}->{adjusted_z:.1f}"
    )
    return adjusted_z


def apply_suction_offset(x, y):
    return (x, y)


def apply_conveyor_offset(world_xy):
    if world_xy is None:
        return None
    wx, wy = world_xy
    return (float(wx), float(wy) + CONVEYOR_Y_OFFSET_MM)


def stop_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\nStopping...")
    running = False
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, stop_handler)


def should_stop():
    return not running


def interruptible_sleep(seconds, poll_interval=0.05):
    deadline = time.time() + max(0.0, float(seconds))
    while time.time() < deadline:
        if should_stop():
            return False
        time.sleep(min(poll_interval, deadline - time.time()))
    return True


def run_servo_move(servo_commands, settle_s):
    if should_stop():
        return False
    print(f"Moving servos: {servo_commands}")
    board.bus_servo_set_position(MOVE_DURATION_S * 3, servo_commands)
    return interruptible_sleep(settle_s)


def initialize_camera():
    global camera
    try:
        print("Initializing camera...")
        camera = takePhoto.initialzeCamera()
        print("Camera initialized successfully")
        return camera
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return None


def capture_photo(camera):
    try:
        if camera is None:
            return None
        
        bgr = takePhoto.takePhoto(camera, save_photo=False)
        
        if bgr is None:
            print(f"Photo not taken")
            return 
        else:
            return bgr
        
        
    except Exception as e:
        print(f"Error capturing photo: {e}")
        return None


def get_next_frame():
    return capture_photo(camera)


def acquire_detected_frame(camera, max_frames=DETECTION_MAX_FRAMES):
    last_image = None
    for frame_idx in range(1, max_frames + 1):
        if should_stop():
            break
        image = capture_photo(camera)
        if image is None:
            continue
        last_image = image
        detections = _detect_with_cv(image)
        if detections:
            return image, detections, frame_idx
        if not interruptible_sleep(DETECTION_FRAME_DELAY_S):
            break
    return last_image, [], max_frames


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


def _workflow_prediction_to_detection(prediction, frame_shape):
    label = _normalize_label(prediction.get("class") or prediction.get("label") or "unknown")
    confidence = float(prediction.get("confidence", prediction.get("score", 0.0)))
    contour = _extract_polygon_contour(prediction)
    if contour is not None:
        parsed = _contour_to_bbox_centroid(contour)
        if parsed is None:
            return None
        (x1, y1, x2, y2), pixel_centroid = parsed
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
            pixel_centroid = (int(round(float(x))), int(round(float(y))))
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
            pixel_centroid = pixel_centroid_from_bbox(x1, y1, x2, y2)
        mask_polygon = None

    cx, cy = pixel_centroid
    world_xy = _project_world_from_centroid(cx, cy)
    return {
        "bbox_xyxy": (x1, y1, x2, y2),
        "pixel_centroid": pixel_centroid,
        "centroid": center_relative_centroid(pixel_centroid, frame_shape),
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
    with tempfile.NamedTemporaryFile(
        prefix="itet_workflow_",
        suffix=".jpg",
        dir=str(CV_WORKFLOW_IMAGE_PATH.parent),
        delete=False,
    ) as temp_file:
        workflow_image_path = Path(temp_file.name)
    try:
        if not cv2.imwrite(str(workflow_image_path), frame_bgr):
            raise RuntimeError(f"Failed to write workflow input image: {workflow_image_path}")
        data = rf_client.run_workflow(
            workspace_name=CV_ROBOFLOW_WORKSPACE,
            workflow_id=CV_ROBOFLOW_WORKFLOW_ID,
            images={CV_ROBOFLOW_INPUT_NAME: str(workflow_image_path)},
            use_cache=CV_ROBOFLOW_USE_CACHE,
        )
    finally:
        try:
            workflow_image_path.unlink(missing_ok=True)
        except Exception:
            pass
    predictions = []
    _collect_predictions(data, predictions)
    detections = []
    for prediction in predictions:
        det = _workflow_prediction_to_detection(prediction, frame_bgr.shape)
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
        key=lambda d: (
            float(np.hypot(d["pixel_centroid"][0] - cx0, d["pixel_centroid"][1] - cy0)),
            -d.get("confidence", 0.0),
        ),
    )[0]


def _annotate_target(frame_bgr, detections, target, requested_centroid=None):
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    frame_center = (int(w / 2), int(h / 2))
    cv2.circle(out, frame_center, 6, (255, 255, 0), 2)
    cv2.putText(
        out,
        "image (0,0)",
        (frame_center[0] + 10, max(20, frame_center[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cx, cy = det["pixel_centroid"]
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
        tcx, tcy = target["pixel_centroid"]
        cv2.circle(out, (tcx, tcy), 9, (0, 255, 255), 2)
        cv2.line(out, frame_center, (tcx, tcy), (0, 255, 255), 1)
        centroid_x, centroid_y = target["centroid"]
        centroid_text = f"centroid=({centroid_x},{centroid_y})"
        world_xy = target.get("world")
        if world_xy is not None:
            centroid_text += f" world=({world_xy[0]:.1f},{world_xy[1]:.1f})"
        cv2.putText(
            out,
            centroid_text,
            (max(10, tcx + 12), max(30, tcy - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
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
        cx, cy = det["pixel_centroid"]
        dist = float(np.hypot(cx - cx0, cy - cy0))
        return (pr, dist, -det.get("confidence", 0.0))

    return sorted(detections, key=key_fn)[0]


def _centroid_to_servo_angles(cx, cy, width, height):
    if width <= 0 or height <= 0:
        return {servo_id: SERVO_NEUTRAL for servo_id in servo_ids}
    dx_norm = (cx - (width / 2.0)) / max(width / 2.0, 1.0)
    dy_norm = (cy - (height / 2.0)) / max(height / 2.0, 1.0)
    if CV_Y_ONLY_MODE:
        dx_norm = 0.0

    angle_3 = SERVO_NEUTRAL + (dx_norm * SERVO_DELTA_MAX)
    angle_4 = SERVO_NEUTRAL - (dy_norm * SERVO_DELTA_MAX)
    angle_5 = SERVO_NEUTRAL

    return {
        1: float(np.clip(angle_3, 0, 240)),
        2: float(np.clip(angle_4, 0, 240)),
        3: float(np.clip(angle_5, 0, 240)),
    }


def _world_to_robot_xy(world_xy):
    if world_xy is None:
        return None
    wx, wy = world_xy
    x = (float(wx) * CV_WORLD_SCALE_X) + CV_WORLD_X_BIAS_MM
    y = (float(wy) * CV_WORLD_SCALE_Y) + CV_WORLD_Y_BIAS_MM
    if CV_Y_ONLY_MODE:
        x = 0.0
    return (x, y)


def _world_to_servo_angles(world_xy):
    robot_xy = _world_to_robot_xy(world_xy)
    if robot_xy is None:
        return None
    x, y = robot_xy
    z = float(CV_TARGET_Z_MM)
    print(f"getting angles for (x:{x},y:{y},z:{z})")
    angles = inverseKinematics.getAngles(x, y, z)
    if angles is None:
        return None
    return logical_angles_to_servo_positions(angles)


def analyze_photo(image):
    try:
        if image is None:
            print("[ANALYSIS] No image to analyze")
            return None, {servo_id: 120 for servo_id in servo_ids}

        image_with_centroid = image.copy()
        neutral = {servo_id: SERVO_NEUTRAL for servo_id in servo_ids}

        if not cv_ready:
            print("[ANALYSIS] CV not initialized, using neutral servo positions")
            return image_with_centroid, neutral

        detections = _detect_with_cv(image)

        h, w = image.shape[:2]
        frame_center = (w / 2.0, h / 2.0)

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            cx, cy = det["pixel_centroid"]
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
        pixel_cx, pixel_cy = target["pixel_centroid"]
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
            servo_positions = _centroid_to_servo_angles(pixel_cx, pixel_cy, w, h)

        cv2.circle(image_with_centroid, (int(frame_center[0]), int(frame_center[1])), 6, (255, 255, 0), 2)
        cv2.circle(image_with_centroid, (pixel_cx, pixel_cy), 8, (0, 255, 255), 2)
        cv2.line(
            image_with_centroid,
            (int(frame_center[0]), int(frame_center[1])),
            (pixel_cx, pixel_cy),
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
        return image, {servo_id: 120 for servo_id in servo_ids}


if __name__ == '__main__':
    print(f"Servo IDs: {servo_ids}")
    parser = argparse.ArgumentParser(description="CV-driven arm control")
    parser.add_argument("--test-centroid", type=str, help="Select detection nearest to pixel centroid x,y")
    parser.add_argument("--photo-path", type=str, default=str(CURRENT_DIR / "cameraCode" / "photos" / "default.jpg"),
                        help="Output photo path for captured/annotated frame")
    parser.add_argument("--no_cam", action="store_true")
    parser.add_argument("--vacuum", dest="use_vacuum", action="store_true", help="Enable vacuum system")
    parser.add_argument("--no-vacuum", dest="use_vacuum", action="store_false", help="Disable vacuum system")
    parser.set_defaults(use_vacuum=None)
    args = parser.parse_args()
    
    class_aliases = parse_class_aliases(
        "bottle",
        "can",
        "6-pack,six-pack,six_pack,6pack",
    )

    if args.no_cam:
        while running:
            try:
                x, y, z = input("enter coordinates: [x],[y],[z]: ").split(",")
            except KeyboardInterrupt:
                stop_handler(None, None)
                break
            x, y, z = int(x), int(y), int(z)
            x += 1/3 * x
            y += 1/3 * y
            z = apply_xy_z_adjustment(x, y, z)
            angles = compute_servo_angles_for_xyz(x, y, z, "manual target")
            if angles is None:
                continue
            print(f"calculated angles: {angles}")
            servo_commands = ik_angles_to_raw_servo_commands(angles)
            if not run_servo_move(servo_commands, MOVE_DURATION_S * 3):
                break


    try:
        ser = None
        use_vacuum = args.use_vacuum
        if use_vacuum is None:
            try:
                vacuum_choice = input("Use vacuum? [y/N]: ").strip().lower()
            except KeyboardInterrupt:
                stop_handler(None, None)
                vacuum_choice = ""
            use_vacuum = vacuum_choice in ("y", "yes")

        no_suction = not use_vacuum
        if use_vacuum:
            try:
                ser = suctionControl.initializeSerial()
                print("Vacuum control enabled")
            except Exception as e:
                no_suction = True
                print(f"Vacuum unavailable: {e}")
        else:
            print("Vacuum control disabled")
        initialize_cv()
        camera = initialize_camera()
        if camera is None:
            raise RuntimeError("Camera initialization failed")
        
        # Initialize servos - set to torque ON (0)
        if board is not None:
            print(f"Initializing servos sequentially (delay={SERVO_STARTUP_ENABLE_DELAY_S:.2f}s)...")
            for servo_id in servo_ids:
                try:
                    board.bus_servo_enable_torque(servo_id, 0)
                    print(f"Servo {servo_id} torque ON")
                    if not interruptible_sleep(SERVO_STARTUP_ENABLE_DELAY_S):
                        break
                except Exception as e:
                    print(f"Warning: Could not enable servo {servo_id}: {e}")
        else:
            print("Board not initialized, skipping servo initialization")
        if board is not None and not should_stop():
            startup_angles = compute_servo_angles_for_xyz(
                STARTUP_HOME_X_MM,
                STARTUP_HOME_Y_MM,
                apply_xy_z_adjustment(STARTUP_HOME_X_MM, STARTUP_HOME_Y_MM, STARTUP_HOME_Z_MM),
                "startup home position",
            )
            if startup_angles is not None:
                print(
                    f"Moving to startup home at "
                    f"x={STARTUP_HOME_X_MM:.1f}, y={STARTUP_HOME_Y_MM:.1f}, z={STARTUP_HOME_Z_MM:.1f}"
                )
                startup_commands = ik_angles_to_raw_servo_commands(startup_angles)
                if not run_servo_move(startup_commands, 2):
                    raise KeyboardInterrupt
        TOTAL_PIXEL_WIDTH = 1920 # correct this
        TOTAL_PIXEL_HEIGHT = 1080 # correct this
        angle_offset = 45
        import math
        if not interruptible_sleep(0.5):
            raise KeyboardInterrupt
        while running:
            try:
                input("press enter to search up to 10 frames for a classified object")
            except KeyboardInterrupt:
                stop_handler(None, None)
                break
            if should_stop():
                break

            photo_path = Path(args.photo_path)
            photo_path.parent.mkdir(parents=True, exist_ok=True)

            image, detections, frames_used = acquire_detected_frame(camera)
            if image is None:
                raise RuntimeError("No image captured from camera")
            if not interruptible_sleep(0.5):
                break

            if not detections:
                annotated_image = _annotate_target(image, [], None)
                cv2.imwrite(str(photo_path), annotated_image)
                print(f"Captured and saved photo to {photo_path}")
                print(f"[PHOTO] No detections found in {frames_used} frame(s)")
                continue
            else:
                print(f"[PHOTO] Detected objects after {frames_used} frame(s)")
                frame_center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
                det = _choose_target(detections, frame_center)
                annotated_image = _annotate_target(image, detections, det)
                cv2.imwrite(str(photo_path), annotated_image)
                print(f"Captured and saved photo to {photo_path}")
                centroid_x, centroid_y = det["centroid"]
                print(f"centroid coordinate: ({centroid_x}, {centroid_y})")
                world_xy = det.get("world")
                if world_xy is None:
                    print("Selected detection has no world coordinate; skipping cycle")
                    continue
                conveyor_world_xy = apply_conveyor_offset(world_xy)
                robot_xy = _world_to_robot_xy(conveyor_world_xy)
                if robot_xy is None:
                    print("Could not convert world coordinate into robot coordinate; skipping cycle")
                    continue
                target_x, target_y = robot_xy
                arm_x, arm_y = apply_suction_offset(target_x, target_y)
                print(
                    f"arm target: "
                    f"({arm_x:.1f}, {arm_y:.1f}) mm "
                    f"from conveyor-world=({conveyor_world_xy[0]:.1f}, {conveyor_world_xy[1]:.1f}) {det.get('world_units')} "
                    f"(base world=({world_xy[0]:.1f}, {world_xy[1]:.1f}), conveyor +y={CONVEYOR_Y_OFFSET_MM:.1f} mm, "
                    f"suction offset disabled)"
                )
                # go to three desired positions (above -> pick up -> drop off)
                suction_on = False
                try:
                    # above location first:
                    above_z = -550.0
                    above_z = apply_xy_z_adjustment(arm_x, arm_y, above_z)
                    angles = compute_servo_angles_for_xyz(arm_x, arm_y, above_z, "above position")
                    if angles is None:
                        continue
                    print(f"calculated angles for above position: {angles}")
                    servo_commands = ik_angles_to_raw_servo_commands(angles)
                    if not run_servo_move(servo_commands, 2):
                        break

                    # activate the suction
                    if not no_suction and not should_stop():
                        suctionControl.turnOnSuction(ser)
                        suction_on = True

                    # pick up location next:
                    preferred_pick_z = apply_xy_z_adjustment(arm_x, arm_y, PICK_Z_MM)
                    pick_z = find_deepest_feasible_z(arm_x, arm_y, preferred_pick_z, above_z)
                    if pick_z is None:
                        print(
                            f"No feasible pick depth found between z={preferred_pick_z:.1f} and z={above_z:.1f} "
                            f"at x={arm_x:.1f}, y={arm_y:.1f}"
                        )
                        continue
                    if abs(pick_z - preferred_pick_z) > 1e-6:
                        print(
                            f"Requested pick depth z={preferred_pick_z:.1f} mm was unreachable; "
                            f"using deepest feasible z={pick_z:.1f} mm"
                        )
                    angles = compute_servo_angles_for_xyz(arm_x, arm_y, pick_z, "pick position")
                    if angles is None:
                        continue
                    print(f"calculated angles for above position: {angles}")
                    servo_commands = ik_angles_to_raw_servo_commands(angles)
                    if not run_servo_move(servo_commands, 2):
                        break

                    # drop off location
                    z = -400
                    dropoff_x = 0.0
                    dropoff_y = 200.0
                    arm_dropoff_x, arm_dropoff_y = apply_suction_offset(dropoff_x, dropoff_y)
                    z = apply_xy_z_adjustment(arm_dropoff_x, arm_dropoff_y, z)
                    angles = compute_servo_angles_for_xyz(arm_dropoff_x, arm_dropoff_y, z, "drop-off position")
                    if angles is None:
                        continue
                    print(f"calculated angles for above position: {angles}")
                    servo_commands = ik_angles_to_raw_servo_commands(angles)
                    if not run_servo_move(servo_commands, 2):
                        break
                finally:
                    if suction_on:
                        try:
                            suctionControl.turnOffSuction(ser)
                        except Exception as e:
                            print(f"Warning: failed to turn off suction: {e}")


       
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
