import json
import sys
from pathlib import Path

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
CAMERA_DIR = CURRENT_DIR / "cameraCode"
sys.path.insert(0, str(CAMERA_DIR))

import takePhoto

OUTPUT_PATH = CURRENT_DIR / "cameraCode" / "camera_intrinsics.json"
PHOTO_PATH = CURRENT_DIR / "cameraCode" / "photos" / "intrinsics_latest.jpg"
PATTERN_SIZE = (7, 9)
SQUARE_SIZE_MM = 20.0
MIN_SAMPLES = 12


def capture_checkerboard(camera, target_format, objp):
    frame = takePhoto.takePhoto(
        camera,
        target_format,
        save_photo=True,
        destination=str(PHOTO_PATH.parent) + "/",
        name=PHOTO_PATH.name,
    )
    if frame is None:
        return None, None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
    annotated = frame.copy()
    if not found:
        cv2.imwrite(str(PHOTO_PATH), annotated)
        return None, None, annotated

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    cv2.drawChessboardCorners(annotated, PATTERN_SIZE, refined, found)
    cv2.imwrite(str(PHOTO_PATH), annotated)
    return objp.copy(), refined, annotated


def main():
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    object_points = []
    image_points = []
    image_size = None

    camera = None
    try:
        camera, target_format = takePhoto.initialzeCamera()
        print(f"Collect at least {MIN_SAMPLES} checkerboard views. Press Enter to capture, q to finish.")
        while True:
            cmd = input("capture> ").strip().lower()
            if cmd == "q":
                break
            obj_pts, img_pts, annotated = capture_checkerboard(camera, target_format, objp)
            if annotated is None:
                print("No frame captured")
                continue
            if obj_pts is None or img_pts is None:
                print(f"Checkerboard not found. Latest image: {PHOTO_PATH}")
                continue

            h, w = annotated.shape[:2]
            image_size = (w, h)
            object_points.append(obj_pts)
            image_points.append(img_pts)
            print(f"Captured checkerboard sample {len(object_points)}")

        if len(object_points) < MIN_SAMPLES:
            raise RuntimeError(f"Need at least {MIN_SAMPLES} valid samples, got {len(object_points)}")

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
        payload = {
            "pattern_size": list(PATTERN_SIZE),
            "square_size_mm": SQUARE_SIZE_MM,
            "image_size": list(image_size),
            "reprojection_error": float(ret),
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
        }
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
        print(f"Saved camera intrinsics to {OUTPUT_PATH}")
        print(f"Reprojection error: {ret:.4f}")
    finally:
        if camera is not None:
            takePhoto.closeCamera(camera)


if __name__ == "__main__":
    main()
