import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
CAMERA_DIR = CURRENT_DIR / "cameraCode"
sys.path.insert(0, str(CAMERA_DIR))

import takePhoto

INTRINSICS_PATH = CURRENT_DIR / "cameraCode" / "camera_intrinsics.json"
CALIBRATION_PATH = CURRENT_DIR / "cameraCode" / "aruco_end_effector_calibration_3d.json"
PHOTO_PATH = CURRENT_DIR / "cameraCode" / "photos" / "aruco_live_xyz_latest.jpg"
SHOW_GUI = os.environ.get("ARUCO_SHOW_GUI", "").lower() in ("1", "true", "yes")
HEADLESS = not SHOW_GUI


def load_intrinsics():
    if not INTRINSICS_PATH.exists():
        raise FileNotFoundError(f"Camera intrinsics not found: {INTRINSICS_PATH}")
    data = json.loads(INTRINSICS_PATH.read_text())
    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float32)
    return camera_matrix, dist_coeffs


def load_calibration():
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"3D calibration file not found: {CALIBRATION_PATH}")
    data = json.loads(CALIBRATION_PATH.read_text())
    rotation = np.asarray(data["camera_to_robot_rotation"], dtype=np.float32)
    translation = np.asarray(data["camera_to_robot_translation_mm"], dtype=np.float32).reshape(3)
    tag_id = int(data.get("aruco_tag_id", 0))
    tag_size_mm = float(data.get("aruco_tag_size_mm", 20.0))
    dict_name = data.get("aruco_dict", "DICT_ARUCO_ORIGINAL")
    return rotation, translation, tag_id, tag_size_mm, dict_name


def get_aruco_dictionary(dict_name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is not available")
    dict_id = getattr(cv2.aruco, dict_name)
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


def estimate_tag_pose(frame_bgr, dictionary, detector, parameters, camera_matrix, dist_coeffs, tag_id, tag_size_mm):
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
    if tag_id not in ids_flat:
        return None, annotated

    idx = ids_flat.index(tag_id)
    marker_corners = corners[idx][0].astype(np.float32)
    half = tag_size_mm / 2.0
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
        cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, rvec, tvec, tag_size_mm * 0.5)

    center = np.mean(marker_corners, axis=0)
    cv2.circle(annotated, (int(round(center[0])), int(round(center[1]))), 5, (0, 0, 255), -1)
    return {
        "rvec": rvec.reshape(3),
        "tvec": tvec.reshape(3),
        "pixel_center": (float(center[0]), float(center[1])),
    }, annotated


def camera_to_robot_xyz(rotation, translation, camera_tvec):
    camera_point = np.asarray(camera_tvec, dtype=np.float32).reshape(3)
    robot_point = rotation @ camera_point + translation
    return float(robot_point[0]), float(robot_point[1]), float(robot_point[2])


def main():
    camera_matrix, dist_coeffs = load_intrinsics()
    rotation, translation, tag_id, tag_size_mm, dict_name = load_calibration()
    dictionary = get_aruco_dictionary(dict_name)
    detector, parameters = create_aruco_detector(dictionary)

    camera = None
    try:
        camera, target_format = takePhoto.initialzeCamera()
        print(f"Live ArUco robot XYZ verification started. tag_id={tag_id}")
        if HEADLESS:
            print(f"Headless mode: annotated frames will be saved to {PHOTO_PATH}")
            print("Press Ctrl+C to quit.")
        else:
            print("Press 'q' or ESC to quit.")

        while True:
            frame = takePhoto.takePhoto(
                camera,
                target_format,
                save_photo=True,
                destination=str(PHOTO_PATH.parent) + "/",
                name=PHOTO_PATH.name,
            )
            if frame is None:
                continue

            pose, annotated = estimate_tag_pose(
                frame,
                dictionary,
                detector,
                parameters,
                camera_matrix,
                dist_coeffs,
                tag_id,
                tag_size_mm,
            )

            if pose is not None:
                robot_x, robot_y, robot_z = camera_to_robot_xyz(rotation, translation, pose["tvec"])
                text = (
                    f"pixel=({pose['pixel_center'][0]:.1f},{pose['pixel_center'][1]:.1f}) "
                    f"camera_t=({pose['tvec'][0]:.1f},{pose['tvec'][1]:.1f},{pose['tvec'][2]:.1f}) "
                    f"robot=({robot_x:.1f},{robot_y:.1f},{robot_z:.1f}) mm"
                )
                cv2.putText(
                    annotated,
                    text,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
                print(text)
            else:
                cv2.putText(
                    annotated,
                    f"tag {tag_id} not detected",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            cv2.imwrite(str(PHOTO_PATH), annotated)
            if not HEADLESS:
                cv2.imshow("Aruco Robot XYZ", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        if camera is not None:
            takePhoto.closeCamera(camera)
        if not HEADLESS:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
