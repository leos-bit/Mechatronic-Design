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

CALIBRATION_PATH = CURRENT_DIR / "cameraCode" / "aruco_end_effector_calibration.json"
PHOTO_PATH = CURRENT_DIR / "cameraCode" / "photos" / "aruco_live_latest.jpg"
HEADLESS = not bool(os.environ.get("DISPLAY"))


def load_calibration():
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"Calibration file not found: {CALIBRATION_PATH}")
    data = json.loads(CALIBRATION_PATH.read_text())
    matrix = np.asarray(data["affine_pixel_to_world_xy"], dtype=np.float32)
    tag_id = int(data.get("aruco_tag_id", 0))
    dict_name = data.get("aruco_dict", "DICT_ARUCO_ORIGINAL")
    zero_pixel_xy = tuple(data.get("zero_pixel_xy", [0.0, 0.0]))
    return matrix, tag_id, dict_name, zero_pixel_xy


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
    return (cx, cy), annotated


def pixel_to_robot_xy(matrix, pixel_xy):
    u, v = pixel_xy
    vec = np.array([u, v, 1.0], dtype=np.float32)
    xy = matrix @ vec
    return float(xy[0]), float(xy[1])


def main():
    matrix, tag_id, dict_name, zero_pixel_xy = load_calibration()
    dictionary = get_aruco_dictionary(dict_name)
    detector, parameters = create_aruco_detector(dictionary)

    camera = None
    try:
        camera, target_format = takePhoto.initialzeCamera()
        print(f"Live ArUco robot XY view started. tag_id={tag_id}")
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

            center_xy, annotated = detect_tag_center(
                frame,
                tag_id,
                dictionary,
                detector=detector,
                parameters=parameters,
            )

            if center_xy is not None:
                world_x, world_y = pixel_to_robot_xy(matrix, center_xy)
                text = (
                    f"pixel=({center_xy[0]:.1f},{center_xy[1]:.1f}) "
                    f"zero_pixel=({zero_pixel_xy[0]:.1f},{zero_pixel_xy[1]:.1f}) "
                    f"world=({world_x:.1f},{world_y:.1f}) mm"
                )
                cv2.putText(
                    annotated,
                    text,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
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
                cv2.imshow("Aruco Robot XY", annotated)
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
