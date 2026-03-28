import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DEMO_DIR = CURRENT_DIR / "Motor Control" / "board_demo"
CAMERA_DIR = CURRENT_DIR / "cameraCode"
IK_DIR = CURRENT_DIR / "Inverse Kinematics"

sys.path.insert(0, str(MOTOR_DEMO_DIR))
sys.path.insert(0, str(CAMERA_DIR))
sys.path.insert(0, str(IK_DIR))

import ros_robot_controller_sdk as rrc
import takePhoto
import inverseKinematics

running = True
servo_ids = [3, 5, 7]
MOVE_DURATION_S = 0.35
SETTLE_S = 0.6
SERVO_ZERO_OFFSETS_DEG = {3: 90.0, 5: 90.0, 7: 90.0}
SERVO_DIRECTIONS = {3: -1.0, 5: -1.0, 7: -1.0}
SERVO_ANGLE_SCALES = {3: 1.0, 5: 1.0, 7: 1.0}
CALIBRATION_Z_MM = -500.0
ARUCO_DICT_NAME = "DICT_ARUCO_ORIGINAL"
ARUCO_TAG_ID = 0
OUTPUT_PATH = CURRENT_DIR / "cameraCode" / "aruco_end_effector_calibration.json"
PHOTO_PATH = CURRENT_DIR / "cameraCode" / "photos" / "aruco_calibration_latest.jpg"
CALIBRATION_POINTS_MM = [
    (-100.0, -100.0),
    (0.0, -100.0),
    (100.0, -100.0),
    (-100.0, 0.0),
    (0.0, 0.0),
    (100.0, 0.0),
    (-100.0, 100.0),
    (0.0, 100.0),
    (100.0, 100.0),
]


def stop_handler(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, stop_handler)


def logical_angle_to_raw(servo_id, logical_angle):
    direction = SERVO_DIRECTIONS.get(servo_id, 1.0)
    scale = SERVO_ANGLE_SCALES.get(servo_id, 1.0)
    physical_angle = SERVO_ZERO_OFFSETS_DEG.get(servo_id, 0.0) + (direction * scale * float(logical_angle))
    physical_angle = max(0.0, min(240.0, physical_angle))
    return int((physical_angle / 240.0) * 1000)


def move_robot(board, xyz):
    angles = inverseKinematics.getAngles(*xyz)
    if angles is None:
        raise RuntimeError(f"No IK solution for point {xyz}")
    commands = [
        [servo_id, logical_angle_to_raw(servo_id, angle)]
        for servo_id, angle in zip(servo_ids, angles)
    ]
    board.bus_servo_set_position(MOVE_DURATION_S * 3, commands)
    time.sleep(MOVE_DURATION_S * 3 + SETTLE_S)


def get_aruco_dictionary():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is not available")
    dict_id = getattr(cv2.aruco, ARUCO_DICT_NAME)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "Dictionary_get"):
        return cv2.aruco.Dictionary_get(dict_id)
    raise RuntimeError("No supported ArUco dictionary API found")


def create_aruco_detector(dictionary):
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    elif hasattr(cv2.aruco, "DetectorParameters_create"):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = None

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters), parameters
    return None, parameters


def detect_tag_center(frame_bgr, tag_id, dictionary, detector=None, parameters=None):
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(frame_bgr)
    elif hasattr(cv2.aruco, "detectMarkers"):
        corners, ids, _ = cv2.aruco.detectMarkers(frame_bgr, dictionary, parameters=parameters)
    else:
        raise RuntimeError("No supported ArUco detectMarkers API found in this OpenCV build")
    if ids is None:
        return None, frame_bgr
    annotated = frame_bgr.copy()
    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    ids_flat = ids.flatten().tolist()
    if tag_id not in ids_flat:
        return None, annotated
    idx = ids_flat.index(tag_id)
    pts = corners[idx][0]
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    cv2.circle(annotated, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
    cv2.putText(
        annotated,
        f"id={tag_id} center=({cx:.1f},{cy:.1f})",
        (int(round(cx)) + 8, int(round(cy)) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
    )
    return (cx, cy), annotated


def fit_affine(pixel_points, robot_points):
    pixel_arr = np.asarray(pixel_points, dtype=np.float32)
    robot_arr = np.asarray(robot_points, dtype=np.float32)
    if len(pixel_arr) < 3:
        raise RuntimeError("Need at least 3 samples to fit affine transform")
    matrix, inliers = cv2.estimateAffine2D(pixel_arr, robot_arr)
    if matrix is None:
        raise RuntimeError("Failed to fit affine transform")
    return matrix, inliers


def apply_affine(matrix, xy):
    x, y = xy
    vec = np.array([x, y, 1.0], dtype=np.float32)
    result = matrix @ vec
    return float(result[0]), float(result[1])


def capture_tag_sample(camera, target_format, dictionary, detector, parameters):
    frame = takePhoto.takePhoto(
        camera,
        target_format,
        save_photo=True,
        destination=str(PHOTO_PATH.parent) + "/",
        name=PHOTO_PATH.name,
    )
    if frame is None:
        return None, None
    center_xy, annotated = detect_tag_center(frame, ARUCO_TAG_ID, dictionary, detector=detector, parameters=parameters)
    cv2.imwrite(str(PHOTO_PATH), annotated)
    return center_xy, annotated


def save_calibration(zeroed_pixel_to_robot_matrix, zero_pixel_xy, samples):
    payload = {
        "aruco_dict": ARUCO_DICT_NAME,
        "aruco_tag_id": ARUCO_TAG_ID,
        "calibration_z_mm": CALIBRATION_Z_MM,
        "zero_pixel_xy": [zero_pixel_xy[0], zero_pixel_xy[1]],
        "affine_zeroed_pixel_to_robot_command_xy": zeroed_pixel_to_robot_matrix.tolist(),
        "samples": samples,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved calibration to {OUTPUT_PATH}")


def main():
    dictionary = get_aruco_dictionary()
    detector, parameters = create_aruco_detector(dictionary)

    print("Initializing board...")
    board = rrc.Board()
    board.enable_reception()
    for servo_id in servo_ids:
        board.bus_servo_enable_torque(servo_id, 0)
        time.sleep(0.2)

    print("Initializing camera...")
    camera, target_format = takePhoto.initialzeCamera()

    zeroed_pixel_points = []
    command_points = []
    samples = []

    try:
        print("\nMove the robot to commanded zero (0, 0, z) so the ArUco tag defines world zero.")
        move_robot(board, (0.0, 0.0, CALIBRATION_Z_MM))
        input("Press Enter when the robot is at zero and the ArUco tag is clearly visible...")

        zero_pixel_xy, _ = capture_tag_sample(camera, target_format, dictionary, detector, parameters)
        if zero_pixel_xy is None:
            raise RuntimeError(f"Could not detect ArUco tag {ARUCO_TAG_ID} at robot zero")
        print(f"Zero tag pixel = ({zero_pixel_xy[0]:.1f}, {zero_pixel_xy[1]:.1f})")

        for idx, (x_mm, y_mm) in enumerate(CALIBRATION_POINTS_MM, start=1):
            if not running:
                break
            xyz = (x_mm, y_mm, CALIBRATION_Z_MM)
            print(f"\nPoint {idx}/{len(CALIBRATION_POINTS_MM)} -> robot ({x_mm:.1f}, {y_mm:.1f}, {CALIBRATION_Z_MM:.1f})")
            move_robot(board, xyz)
            input("Press Enter when the robot is settled and the ArUco tag is visible...")

            center_xy, _ = capture_tag_sample(camera, target_format, dictionary, detector, parameters)
            if center_xy is None:
                print(f"Tag {ARUCO_TAG_ID} not found. Latest image: {PHOTO_PATH}")
                retry = input("Retry this point? [Y/n]: ").strip().lower()
                if retry in ("", "y", "yes"):
                    continue
                print("Skipping sample")
                continue

            zeroed_pixel_x = center_xy[0] - zero_pixel_xy[0]
            zeroed_pixel_y = center_xy[1] - zero_pixel_xy[1]

            zeroed_pixel_points.append((zeroed_pixel_x, zeroed_pixel_y))
            command_points.append((x_mm, y_mm))
            samples.append(
                {
                    "command_xy_mm": [x_mm, y_mm],
                    "pixel_xy": [center_xy[0], center_xy[1]],
                    "zeroed_pixel_xy": [zeroed_pixel_x, zeroed_pixel_y],
                }
            )
            print(
                f"Captured tag center pixel=({center_xy[0]:.1f}, {center_xy[1]:.1f}) "
                f"zeroed_pixel=({zeroed_pixel_x:.1f}, {zeroed_pixel_y:.1f})"
            )

        if len(zeroed_pixel_points) < 3:
            raise RuntimeError("Not enough valid samples collected")

        zeroed_pixel_to_robot_matrix, inliers = fit_affine(zeroed_pixel_points, command_points)
        save_calibration(zeroed_pixel_to_robot_matrix, zero_pixel_xy, samples)
        print("Affine zeroed-pixel -> robot-command transform:")
        print(zeroed_pixel_to_robot_matrix)
        if inliers is not None:
            print(f"Inliers: {int(np.sum(inliers))}/{len(inliers)}")

    finally:
        try:
            takePhoto.closeCamera(camera)
        except Exception:
            pass
        try:
            for servo_id in servo_ids:
                board.bus_servo_enable_torque(servo_id, 1)
        except Exception:
            pass


if __name__ == "__main__":
    main()
