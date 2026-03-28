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
ARUCO_DICT_NAME = "DICT_ARUCO_ORIGINAL"
ARUCO_TAG_ID = 0
ARUCO_TAG_SIZE_MM = 20.0
INTRINSICS_PATH = CURRENT_DIR / "cameraCode" / "camera_intrinsics.json"
OUTPUT_PATH = CURRENT_DIR / "cameraCode" / "aruco_end_effector_calibration_3d.json"
PHOTO_PATH = CURRENT_DIR / "cameraCode" / "photos" / "aruco_3d_calibration_latest.jpg"
CALIBRATION_POINTS_MM = [
    (-100.0, -100.0, -550.0),
    (0.0, -100.0, -550.0),
    (100.0, -100.0, -550.0),
    (-100.0, 0.0, -550.0),
    (0.0, 0.0, -550.0),
    (100.0, 0.0, -550.0),
    (-100.0, 100.0, -550.0),
    (0.0, 100.0, -550.0),
    (100.0, 100.0, -550.0),
    (-75.0, -75.0, -450.0),
    (75.0, -75.0, -450.0),
    (0.0, 75.0, -450.0),
    (-75.0, -75.0, -650.0),
    (75.0, -75.0, -650.0),
    (0.0, 75.0, -650.0),
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
    commands = [[servo_id, logical_angle_to_raw(servo_id, angle)] for servo_id, angle in zip(servo_ids, angles)]
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


def load_intrinsics():
    if not INTRINSICS_PATH.exists():
        raise FileNotFoundError(f"Camera intrinsics not found: {INTRINSICS_PATH}")
    data = json.loads(INTRINSICS_PATH.read_text())
    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float32)
    return camera_matrix, dist_coeffs


def estimate_tag_pose(frame_bgr, dictionary, detector, parameters, camera_matrix, dist_coeffs):
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(frame_bgr)
    elif hasattr(cv2.aruco, "detectMarkers"):
        corners, ids, _ = cv2.aruco.detectMarkers(frame_bgr, dictionary, parameters=parameters)
    else:
        raise RuntimeError("No supported ArUco detectMarkers API found in this OpenCV build")

    annotated = frame_bgr.copy()
    if ids is None:
        return None, annotated

    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    ids_flat = ids.flatten().tolist()
    if ARUCO_TAG_ID not in ids_flat:
        return None, annotated

    idx = ids_flat.index(ARUCO_TAG_ID)
    marker_corners = corners[idx][0].astype(np.float32)
    half = ARUCO_TAG_SIZE_MM / 2.0
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        marker_corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None, annotated

    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, rvec, tvec, ARUCO_TAG_SIZE_MM * 0.5)
    center = np.mean(marker_corners, axis=0)
    cv2.circle(annotated, (int(round(center[0])), int(round(center[1]))), 5, (0, 0, 255), -1)
    return {
        "rvec": rvec.reshape(3),
        "tvec": tvec.reshape(3),
        "pixel_center": (float(center[0]), float(center[1])),
    }, annotated


def fit_rigid_transform(camera_points, robot_points):
    src = np.asarray(camera_points, dtype=np.float64)
    dst = np.asarray(robot_points, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise RuntimeError("Need matching 3D point sets with at least 3 samples")

    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    h = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = dst_centroid - (r @ src_centroid)
    return r, t


def save_calibration(rotation_matrix, translation_vector, samples):
    payload = {
        "aruco_dict": ARUCO_DICT_NAME,
        "aruco_tag_id": ARUCO_TAG_ID,
        "aruco_tag_size_mm": ARUCO_TAG_SIZE_MM,
        "camera_to_robot_rotation": rotation_matrix.tolist(),
        "camera_to_robot_translation_mm": translation_vector.tolist(),
        "samples": samples,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved 3D calibration to {OUTPUT_PATH}")


def main():
    dictionary = get_aruco_dictionary()
    detector, parameters = create_aruco_detector(dictionary)
    camera_matrix, dist_coeffs = load_intrinsics()

    print("Initializing board...")
    board = rrc.Board()
    board.enable_reception()
    for servo_id in servo_ids:
        board.bus_servo_enable_torque(servo_id, 0)
        time.sleep(0.2)

    print("Initializing camera...")
    camera, target_format = takePhoto.initialzeCamera()

    camera_points = []
    robot_points = []
    samples = []

    try:
        for idx, xyz in enumerate(CALIBRATION_POINTS_MM, start=1):
            if not running:
                break
            print(f"\nPoint {idx}/{len(CALIBRATION_POINTS_MM)} -> robot {xyz}")
            move_robot(board, xyz)
            input("Press Enter when the robot is settled and the ArUco tag is visible...")

            frame = takePhoto.takePhoto(
                camera,
                target_format,
                save_photo=True,
                destination=str(PHOTO_PATH.parent) + "/",
                name=PHOTO_PATH.name,
            )
            if frame is None:
                print("No frame captured; skipping sample")
                continue

            pose, annotated = estimate_tag_pose(frame, dictionary, detector, parameters, camera_matrix, dist_coeffs)
            cv2.imwrite(str(PHOTO_PATH), annotated)
            if pose is None:
                print(f"Tag {ARUCO_TAG_ID} not detected/posed. Latest image: {PHOTO_PATH}")
                retry = input("Retry this point? [Y/n]: ").strip().lower()
                if retry in ("", "y", "yes"):
                    continue
                print("Skipping sample")
                continue

            camera_point = pose["tvec"]
            camera_points.append(camera_point)
            robot_points.append(np.asarray(xyz, dtype=np.float64))
            samples.append(
                {
                    "robot_xyz_mm": list(map(float, xyz)),
                    "camera_tvec_mm": list(map(float, camera_point)),
                    "pixel_center": [pose["pixel_center"][0], pose["pixel_center"][1]],
                }
            )
            print(
                f"Captured tag pose camera_t=({camera_point[0]:.1f}, {camera_point[1]:.1f}, {camera_point[2]:.1f}) mm "
                f"pixel=({pose['pixel_center'][0]:.1f}, {pose['pixel_center'][1]:.1f})"
            )

        if len(camera_points) < 4:
            raise RuntimeError("Not enough valid 3D samples collected")

        rotation_matrix, translation_vector = fit_rigid_transform(camera_points, robot_points)
        save_calibration(rotation_matrix, translation_vector, samples)
        print("Camera -> robot rotation:")
        print(rotation_matrix)
        print("Camera -> robot translation (mm):")
        print(translation_vector)
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
