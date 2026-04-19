import argparse
from pathlib import Path

import cv2
import numpy as np

import iTet


WINDOW_NAME = "Homography Calibration"
CLICK_COLOR = (0, 255, 255)
TEXT_COLOR = (255, 255, 255)
HELP_COLOR = (0, 255, 0)
OUTPUT_IMAGE_PATH = iTet.CURRENT_DIR / "cameraCode" / "photos" / "homography_calibration_points.jpg"


def format_points(points):
    return ";".join(f"{x:.1f},{y:.1f}" for x, y in points)


def load_frame(args):
    if args.capture:
        camera = iTet.initialize_camera()
        if camera is None:
            raise RuntimeError("Could not initialize camera")
        try:
            frame = iTet.capture_photo(camera)
        finally:
            try:
                iTet.takePhoto.closeCamera(camera)
            except Exception:
                pass
        if frame is None:
            raise RuntimeError("Could not capture frame from camera")
        image_path = Path(args.output_image)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), frame)
        print(f"Captured calibration image to {image_path}")
        return frame

    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return frame


def draw_overlay(base_image, clicked_points):
    image = base_image.copy()
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    cv2.drawMarker(image, center, (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    cv2.putText(
        image,
        "image center",
        (center[0] + 10, max(25, center[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    for idx, (x, y) in enumerate(clicked_points, start=1):
        pt = (int(round(x)), int(round(y)))
        cv2.circle(image, pt, 7, CLICK_COLOR, -1)
        cv2.putText(
            image,
            str(idx),
            (pt[0] + 10, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2,
        )

    help_lines = [
        "Click calibration points in order.",
        "Keys: u=undo, r=reset, c=compute, q=quit",
    ]
    for i, line in enumerate(help_lines):
        cv2.putText(
            image,
            line,
            (20, 35 + (i * 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            HELP_COLOR,
            2,
        )
    return image


def collect_clicked_points(base_image):
    clicked_points = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((float(x), float(y)))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        preview = draw_overlay(base_image, clicked_points)
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            clicked_points = []
            break
        if key == ord("u") and clicked_points:
            clicked_points.pop()
        elif key == ord("r"):
            clicked_points.clear()
        elif key == ord("c"):
            if len(clicked_points) >= 4:
                break
            print("Need at least 4 points before computing homography.")

    cv2.destroyWindow(WINDOW_NAME)
    return clicked_points


def collect_world_points(src_points, args):
    if args.dst:
        dst_points = iTet._parse_points(args.dst)
        if len(dst_points) != len(src_points):
            raise ValueError(
                f"--dst provided {len(dst_points)} points, but {len(src_points)} image points were clicked"
            )
        return [tuple(map(float, pt)) for pt in dst_points]

    print("")
    print("Enter robot/world XY coordinates in millimeters for each clicked point.")
    print("Use the same order you clicked in the image.")
    dst_points = []
    for idx, (px, py) in enumerate(src_points, start=1):
        while True:
            raw = input(f"Point {idx} at pixel ({px:.1f}, {py:.1f}) -> world x,y mm: ").strip()
            try:
                x_str, y_str = [part.strip() for part in raw.split(",")]
                dst_points.append((float(x_str), float(y_str)))
                break
            except Exception:
                print("Expected input like: 120.5,-45.0")
    return dst_points


def main():
    parser = argparse.ArgumentParser(description="Click image points and compute a new homography")
    parser.add_argument(
        "--image",
        default=str(iTet.CURRENT_DIR / "cameraCode" / "photos" / "default.jpg"),
        help="Existing image to calibrate against",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Capture a fresh frame using cameraCode/takePhoto.py before calibration",
    )
    parser.add_argument(
        "--output-image",
        default=str(OUTPUT_IMAGE_PATH),
        help="Where to save a captured or annotated calibration image",
    )
    parser.add_argument(
        "--dst",
        help="Optional semicolon-separated world points like 'x1,y1;x2,y2;...'",
    )
    args = parser.parse_args()

    frame = load_frame(args)
    src_points = collect_clicked_points(frame)
    if len(src_points) < 4:
        print("Calibration cancelled.")
        return

    dst_points = collect_world_points(src_points, args)
    src_np = np.array(src_points, dtype=np.float32)
    dst_np = np.array(dst_points, dtype=np.float32)
    homography, _ = cv2.findHomography(src_np, dst_np, method=0)
    if homography is None:
        raise RuntimeError("Failed to compute homography from the provided points")

    annotated = draw_overlay(frame, src_points)
    output_image_path = Path(args.output_image)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image_path), annotated)

    print("")
    print(f"Saved annotated calibration image to {output_image_path}")
    print("Update iTet.py with:")
    print(f'CV_HOMOGRAPHY_SRC = "{format_points(src_points)}"')
    print(f'CV_HOMOGRAPHY_DST = "{format_points(dst_points)}"')
    print("")
    print("Homography matrix:")
    print(homography)


if __name__ == "__main__":
    main()
