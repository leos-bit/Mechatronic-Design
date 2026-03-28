import os
import sys
import time
from pathlib import Path

import cv2

try:
    from inference_sdk import InferenceHTTPClient
except ImportError as exc:
    raise ImportError(
        "inference_sdk is required. Activate the venv where you installed inference-sdk."
    ) from exc

CURRENT_DIR = Path(__file__).resolve().parent
CAMERA_DIR = CURRENT_DIR / "cameraCode"
sys.path.insert(0, str(CAMERA_DIR))

import takePhoto

ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "leos-workspace-qswhy")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "yolov11")
ROBOFLOW_INPUT_NAME = os.getenv("ROBOFLOW_INPUT_NAME", "image")
ROBOFLOW_USE_CACHE = os.getenv("ROBOFLOW_USE_CACHE", "false").lower() in ("1", "true", "yes")
PHOTO_PATH = CURRENT_DIR / "cameraCode" / "photos" / "default.jpg"
FRAME_DELAY_S = float(os.getenv("ROBOFLOW_FRAME_DELAY_S", "0.15"))


def normalize_label(label):
    value = str(label).lower()
    if "bottle" in value:
        return "bottle"
    if "six" in value or "6-pack" in value or "6pack" in value:
        return "six_pack"
    if "can" in value:
        return "can"
    return str(label)


def collect_predictions(node, out):
    if isinstance(node, dict):
        preds = node.get("predictions")
        if isinstance(preds, list):
            for pred in preds:
                if isinstance(pred, dict):
                    out.append(pred)
        for value in node.values():
            collect_predictions(value, out)
    elif isinstance(node, list):
        for item in node:
            collect_predictions(item, out)


def prediction_to_overlay(prediction):
    label = normalize_label(prediction.get("class") or prediction.get("label") or "unknown")
    conf = float(prediction.get("confidence", prediction.get("score", 0.0)))
    x = prediction.get("x")
    y = prediction.get("y")
    w = prediction.get("width", prediction.get("w"))
    h = prediction.get("height", prediction.get("h"))

    if all(v is not None for v in (x, y, w, h)):
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        x1 = int(round(float(x) - (float(w) / 2.0)))
        y1 = int(round(float(y) - (float(h) / 2.0)))
        x2 = int(round(float(x) + (float(w) / 2.0)))
        y2 = int(round(float(y) + (float(h) / 2.0)))
        return {"label": label, "confidence": conf, "bbox": (x1, y1, x2, y2), "centroid": (cx, cy)}

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
    return {
        "label": label,
        "confidence": conf,
        "bbox": (x1, y1, x2, y2),
        "centroid": (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))),
    }


def run_workflow(client, image_path):
    result = client.run_workflow(
        workspace_name=ROBOFLOW_WORKSPACE,
        workflow_id=ROBOFLOW_WORKFLOW_ID,
        images={ROBOFLOW_INPUT_NAME: str(image_path)},
        use_cache=ROBOFLOW_USE_CACHE,
    )
    predictions = []
    collect_predictions(result, predictions)
    overlays = []
    for prediction in predictions:
        parsed = prediction_to_overlay(prediction)
        if parsed is not None:
            overlays.append(parsed)
    overlays.sort(key=lambda item: item["confidence"], reverse=True)
    return overlays


def draw_overlays(frame, overlays):
    annotated = frame.copy()
    for item in overlays:
        x1, y1, x2, y2 = item["bbox"]
        cx, cy = item["centroid"]
        label = item["label"]
        conf = item["confidence"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"{label} {conf:.2f} c=({cx},{cy})",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
    return annotated


def main():
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY is not set")

    client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=ROBOFLOW_API_KEY,
    )

    camera = None
    try:
        camera, target_format = takePhoto.initialzeCamera()
        print(f"Live workflow view started: {ROBOFLOW_WORKSPACE}/{ROBOFLOW_WORKFLOW_ID}")
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
                print("No frame captured")
                time.sleep(FRAME_DELAY_S)
                continue

            overlays = run_workflow(client, PHOTO_PATH)
            annotated = draw_overlays(frame, overlays)
            cv2.imshow("Roboflow Live View", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            time.sleep(FRAME_DELAY_S)
    finally:
        if camera is not None:
            takePhoto.closeCamera(camera)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
