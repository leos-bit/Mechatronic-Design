import signal
import time
import sys
import os
from pathlib import Path

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DEMO_DIR = CURRENT_DIR / "Motor Control" / "board_demo"
CAMERA_DIR = CURRENT_DIR / "cameraCode"
CV_CODE_DIR = CURRENT_DIR / "Computer Vision" / "code"

# Add path to access board SDK, camera module, and CV helpers
sys.path.insert(0, str(MOTOR_DEMO_DIR))
sys.path.insert(0, str(CAMERA_DIR))
sys.path.insert(0, str(CV_CODE_DIR))

import ros_robot_controller_sdk as rrc
import takePhoto
from belt_objects import load_yolo, parse_class_aliases, detect_objects_in_frame

# Global control variables
running = True
servo_ids = [3, 4, 5]  # IDs of the three bus servos
MOVE_DURATION_S = 0.8

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
video_cap = None
cv_model = None
cv_class_aliases = None
cv_homography = None
cv_ready = False

# CV inference config
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

# Input source config
INPUT_SOURCE_MODE = "recorded"  # live | recorded
RECORDED_VIDEO_PATH = CURRENT_DIR / "Computer Vision" / "IMG_1055.mov"
RECORDED_VIDEO_START_FRAME = 0
RECORDED_VIDEO_FRAME_STEP = 1
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
        
        bgr = takePhoto.takePhoto(camera, target_format)
        
        if bgr is None:
            print(f"Photo not taken")
            return 
        else:
            return bgr
        
        
    except Exception as e:
        print(f"Error capturing photo: {e}")
        return None


def initialize_recorded_video(video_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[VIDEO] Failed to open recorded video: {video_path}")
            return None
        if RECORDED_VIDEO_START_FRAME > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, RECORDED_VIDEO_START_FRAME)
        print(f"[VIDEO] Using recorded source: {video_path}")
        return cap
    except Exception as e:
        print(f"[VIDEO] Initialization error: {e}")
        return None


def get_next_frame():
    global video_cap
    if INPUT_SOURCE_MODE == "recorded":
        if video_cap is None:
            return None
        ok, frame = video_cap.read()
        if not ok:
            return None
        if RECORDED_VIDEO_FRAME_STEP > 1:
            for _ in range(RECORDED_VIDEO_FRAME_STEP - 1):
                if not video_cap.grab():
                    break
        return frame
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


def initialize_cv():
    global cv_model, cv_class_aliases, cv_homography, cv_ready
    try:
        if not CV_MODEL_PATH.exists():
            print(f"[CV] Model not found: {CV_MODEL_PATH}")
            cv_ready = False
            return
        cv_model = load_yolo(CV_MODEL_PATH)
        cv_class_aliases = parse_class_aliases(
            "bottle",
            "can",
            "6-pack,six-pack,six_pack,6pack",
        )
        if CV_USE_HOMOGRAPHY:
            src = _parse_points(CV_HOMOGRAPHY_SRC)
            dst = _parse_points(CV_HOMOGRAPHY_DST)
            if len(src) >= 4 and len(src) == len(dst):
                h, _ = cv2.findHomography(src, dst, method=0)
                cv_homography = h
        cv_ready = True
        print(f"[CV] Initialized with model: {CV_MODEL_PATH}")
    except Exception as e:
        print(f"[CV] Initialization failed: {e}")
        cv_ready = False


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
        return {3: SERVO_NEUTRAL, 4: SERVO_NEUTRAL, 5: SERVO_NEUTRAL}
    dx_norm = (cx - (width / 2.0)) / max(width / 2.0, 1.0)
    dy_norm = (cy - (height / 2.0)) / max(height / 2.0, 1.0)

    angle_3 = SERVO_NEUTRAL + (dx_norm * SERVO_DELTA_MAX)
    angle_4 = SERVO_NEUTRAL - (dy_norm * SERVO_DELTA_MAX)
    angle_5 = SERVO_NEUTRAL

    return {
        3: float(np.clip(angle_3, 0, 240)),
        4: float(np.clip(angle_4, 0, 240)),
        5: float(np.clip(angle_5, 0, 240)),
    }


def analyze_photo(image):
    try:
        if image is None:
            print("[ANALYSIS] No image to analyze")
            return None, {3: 120, 4: 120, 5: 120}

        image_with_centroid = image.copy()
        neutral = {3: SERVO_NEUTRAL, 4: SERVO_NEUTRAL, 5: SERVO_NEUTRAL}

        if not cv_ready:
            print("[ANALYSIS] CV not initialized, using neutral servo positions")
            return image_with_centroid, neutral

        detections = detect_objects_in_frame(
            frame_bgr=image,
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

        h, w = image.shape[:2]
        frame_center = (w / 2.0, h / 2.0)

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            cx, cy = det["centroid"]
            label = det["class"]
            score = det["confidence"]
            cv2.rectangle(image_with_centroid, (x1, y1), (x2, y2), (0, 255, 0), 2)
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

        world = target.get("world")
        world_msg = f", world=({world[0]:.1f},{world[1]:.1f}) {target.get('world_units')}" if world else ""
        print(
            f"[ANALYSIS] target={target['class']} centroid=({cx},{cy}) conf={target['confidence']:.2f}"
            f"{world_msg} -> angles: {servo_positions}"
        )
        return image_with_centroid, servo_positions

    except Exception as e:
        print(f"Error analyzing photo: {e}")
        return image, {3: 120, 4: 120, 5: 120}


def move_servos(servo_positions):
    if board is None:
        print("Board not initialized, skipping servo movement")
        return
    
    try:
        # Convert angles to servo positions and move servos
        servo_commands = []
        for servo_id in servo_ids:
            if servo_id in servo_positions:
                angle = servo_positions[servo_id]
                # Clamp angle to valid range (0-240)
                angle = max(0, min(240, angle))
                # Convert angle to position (0-1000)
                raw_pos = int((angle / 240.0) * 1000)
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
    
    try:
        if INPUT_SOURCE_MODE == "recorded":
            video_cap = initialize_recorded_video(RECORDED_VIDEO_PATH)
            if video_cap is None:
                raise RuntimeError("Recorded video source initialization failed")
        else:
            camera, target_format = initialize_camera()
        initialize_cv()
        
        # Initialize servos - set to torque ON (0)
        if board is not None:
            print("Initializing servos...")
            for servo_id in servo_ids:
                try:
                    board.bus_servo_enable_torque(servo_id, 0)
                    print(f"Servo {servo_id} torque ON")
                except Exception as e:
                    print(f"Warning: Could not enable servo {servo_id}: {e}")
        else:
            print("Board not initialized, skipping servo initialization")
        
        time.sleep(0.5)
        
        print(f"Starting main loop ({INPUT_SOURCE_MODE} source). Press Ctrl+C to stop.")
        
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
                    if INPUT_SOURCE_MODE != "recorded":
                        time.sleep(LOOP_DELAY_S)
                else:
                    print("No frame captured (end of stream or camera issue)")
                    if INPUT_SOURCE_MODE == "recorded":
                        break
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
        if video_cap is not None:
            try:
                video_cap.release()
                print("Recorded video source closed")
            except Exception as e:
                print(f"Error closing recorded video source: {e}")

        if camera is not None:
            try:
                takePhoto.closeCamera(camera)
                print("Camera closed")
            except Exception as e:
                print(f"Error closing camera: {e}")
        
        print("Shutdown complete")
