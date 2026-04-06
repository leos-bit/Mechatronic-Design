import argparse
import os
import tempfile
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "opencv-python is required. Install it in the Python environment you will use."
    ) from exc

try:
    from inference_sdk import InferenceHTTPClient
except ImportError as exc:
    raise ImportError(
        "inference_sdk is required. Install inference-sdk in the Python environment you will use."
    ) from exc


DEFAULT_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
DEFAULT_API_KEY = os.getenv("ROBOFLOW_API_KEY", "oPusoqJbAhSfo6zbicdc")
DEFAULT_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "leos-workspace-qswhy")
DEFAULT_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "yolov11")
DEFAULT_INPUT_NAME = os.getenv("ROBOFLOW_INPUT_NAME", "image")
DEFAULT_USE_CACHE = os.getenv("ROBOFLOW_USE_CACHE", "true").lower() in ("1", "true", "yes")
DEFAULT_VIDEO_PATH = Path(
    "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/Computer Vision Testing/IMG_1178.MOV"
)
DEFAULT_OUTPUT_PATH = DEFAULT_VIDEO_PATH.with_name(f"{DEFAULT_VIDEO_PATH.stem}_annotated.mp4")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a Roboflow workflow on frames from a prerecorded video."
    )
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO_PATH),
        help="Single prerecorded input video path.",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use a live webcam feed instead of a prerecorded video file.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for live webcam mode.",
    )
    parser.add_argument(
        "--video-dir",
        default="Videos",
        help="Directory of prerecorded videos to process when --video is empty.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Annotated output video path for --video.",
    )
    parser.add_argument(
        "--output-dir",
        default="Videos/outputs",
        help="Directory for annotated output videos when processing a folder.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Roboflow inference API URL.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="Roboflow API key. Prefer ROBOFLOW_API_KEY env var.",
    )
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help="Roboflow workspace name.",
    )
    parser.add_argument(
        "--workflow-id",
        default=DEFAULT_WORKFLOW_ID,
        help="Roboflow workflow ID.",
    )
    parser.add_argument(
        "--input-name",
        default=DEFAULT_INPUT_NAME,
        help="Workflow image input name.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Run Roboflow inference on every Nth frame. OpenCV tracking fills the frames in between.",
    )
    parser.add_argument(
        "--tracker",
        default="csrt",
        choices=("csrt", "kcf", "mil"),
        help="OpenCV tracker to use between Roboflow inference calls.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on processed frames. 0 means all frames.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated video while processing.",
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_CACHE,
        help="Enable or disable Roboflow cache usage.",
    )
    return parser


def discover_videos(video_dir):
    supported_suffixes = {".mp4", ".mov", ".avi", ".mkv"}
    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )


def build_output_path(video_path, output_dir):
    return output_dir / f"{video_path.stem}_roboflow.mp4"


def clamp_bbox(bbox, frame_shape):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return (x1, y1, x2, y2)


def create_tracker(tracker_name):
    tracker_name = tracker_name.lower()
    constructor_names = {
        "csrt": "TrackerCSRT_create",
        "kcf": "TrackerKCF_create",
        "mil": "TrackerMIL_create",
    }
    constructor_name = constructor_names[tracker_name]

    constructor = getattr(cv2, constructor_name, None)
    if constructor is not None:
        return constructor()

    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        constructor = getattr(legacy, constructor_name, None)
        if constructor is not None:
            return constructor()

    raise RuntimeError(
        f"OpenCV tracker '{tracker_name}' is not available in this build. "
        "Try --tracker mil or install opencv-contrib-python."
    )


def overlays_to_trackers(frame, overlays, tracker_name):
    tracked_items = []
    for overlay in overlays:
        bbox = clamp_bbox(overlay["bbox"], frame.shape)
        x1, y1, x2, y2 = bbox
        tracker = create_tracker(tracker_name)
        ok = tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        if ok is False:
            continue
        tracked_items.append(
            {
                "tracker": tracker,
                "label": overlay["label"],
                "confidence": overlay["confidence"],
            }
        )
    return tracked_items


def trackers_to_overlays(frame, tracked_items):
    overlays = []
    active_trackers = []
    for item in tracked_items:
        ok, bbox = item["tracker"].update(frame)
        if not ok:
            continue

        x, y, w, h = bbox
        x1, y1, x2, y2 = clamp_bbox((x, y, x + w, y + h), frame.shape)
        overlays.append(
            {
                "label": item["label"],
                "confidence": item["confidence"],
                "bbox": (x1, y1, x2, y2),
                "centroid": (
                    int(round((x1 + x2) / 2.0)),
                    int(round((y1 + y2) / 2.0)),
                ),
            }
        )
        active_trackers.append(item)
    return overlays, active_trackers


def collect_predictions(node, out):
    if isinstance(node, dict):
        predictions = node.get("predictions")
        if isinstance(predictions, list):
            for prediction in predictions:
                if isinstance(prediction, dict):
                    out.append(prediction)
        for value in node.values():
            collect_predictions(value, out)
    elif isinstance(node, list):
        for item in node:
            collect_predictions(item, out)


def collect_classifications(node, out):
    if isinstance(node, dict):
        if isinstance(node.get("top"), str):
            out.append(
                {
                    "label": node["top"],
                    "confidence": float(node.get("confidence", node.get("score", 0.0))),
                }
            )
        classifications = node.get("predictions")
        if isinstance(classifications, dict):
            for label, score in classifications.items():
                try:
                    out.append({"label": str(label), "confidence": float(score)})
                except (TypeError, ValueError):
                    continue
        for value in node.values():
            collect_classifications(value, out)
    elif isinstance(node, list):
        for item in node:
            collect_classifications(item, out)


def prediction_to_overlay(prediction):
    label = str(prediction.get("class") or prediction.get("label") or "unknown")
    confidence = float(prediction.get("confidence", prediction.get("score", 0.0)))
    x = prediction.get("x")
    y = prediction.get("y")
    width = prediction.get("width", prediction.get("w"))
    height = prediction.get("height", prediction.get("h"))

    if all(value is not None for value in (x, y, width, height)):
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        x1 = int(round(float(x) - (float(width) / 2.0)))
        y1 = int(round(float(y) - (float(height) / 2.0)))
        x2 = int(round(float(x) + (float(width) / 2.0)))
        y2 = int(round(float(y) + (float(height) / 2.0)))
        return {
            "label": label,
            "confidence": confidence,
            "bbox": (x1, y1, x2, y2),
            "centroid": (cx, cy),
        }

    x1 = prediction.get("x1", prediction.get("left"))
    y1 = prediction.get("y1", prediction.get("top"))
    x2 = prediction.get("x2", prediction.get("right"))
    y2 = prediction.get("y2", prediction.get("bottom"))
    if any(value is None for value in (x1, y1, x2, y2)):
        return None

    x1 = int(round(float(x1)))
    y1 = int(round(float(y1)))
    x2 = int(round(float(x2)))
    y2 = int(round(float(y2)))
    return {
        "label": label,
        "confidence": confidence,
        "bbox": (x1, y1, x2, y2),
        "centroid": (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))),
    }


def run_workflow(client, image_path, workspace, workflow_id, input_name, use_cache):
    result = client.run_workflow(
        workspace_name=workspace,
        workflow_id=workflow_id,
        images={input_name: str(image_path)},
        use_cache=use_cache,
    )

    raw_predictions = []
    collect_predictions(result, raw_predictions)
    overlays = []
    for prediction in raw_predictions:
        parsed = prediction_to_overlay(prediction)
        if parsed is not None:
            overlays.append(parsed)
    overlays.sort(key=lambda item: item["confidence"], reverse=True)

    classifications = []
    collect_classifications(result, classifications)
    classifications.sort(key=lambda item: item["confidence"], reverse=True)

    return overlays, classifications[:3]


def draw_annotations(frame, overlays, classifications, frame_index):
    annotated = frame.copy()

    for item in overlays:
        x1, y1, x2, y2 = item["bbox"]
        cx, cy = item["centroid"]
        label = item["label"]
        confidence = item["confidence"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"{label} {confidence:.2f} c=({cx},{cy})",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
        )

    cv2.putText(
        annotated,
        f"frame={frame_index}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    for index, item in enumerate(classifications[:3], start=1):
        cv2.putText(
            annotated,
            f"class {index}: {item['label']} {item['confidence']:.2f}",
            (16, 28 + (index * 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    return annotated


def process_capture(client, capture, writer, args, source_label, frame_count=0, fps=30.0):
    inference_calls = 0
    processed_frames = 0
    last_overlays = []
    last_classifications = []
    tracked_items = []

    print(f"Input source: {source_label}")
    if writer is not None:
        print(f"Output video: {args.output}")
    print(
        f"Workflow: {args.workspace}/{args.workflow_id} | frames={frame_count} | "
        f"fps={fps:.2f} | frame_step={args.frame_step} | tracker={args.tracker}"
    )

    with tempfile.TemporaryDirectory(prefix="roboflow_video_") as temp_dir:
        temp_frame_path = Path(temp_dir) / "frame.jpg"

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            processed_frames += 1
            frame_index = processed_frames - 1

            if args.max_frames and processed_frames > args.max_frames:
                break

            if frame_index % args.frame_step == 0:
                if not cv2.imwrite(str(temp_frame_path), frame):
                    raise RuntimeError(f"Failed to write temp frame: {temp_frame_path}")
                last_overlays, last_classifications = run_workflow(
                    client=client,
                    image_path=temp_frame_path,
                    workspace=args.workspace,
                    workflow_id=args.workflow_id,
                    input_name=args.input_name,
                    use_cache=args.use_cache,
                )
                inference_calls += 1
                tracked_items = overlays_to_trackers(
                    frame=frame,
                    overlays=last_overlays,
                    tracker_name=args.tracker,
                )
                print(
                    f"[frame {frame_index}] detections={len(last_overlays)} "
                    f"classifications={len(last_classifications)}"
                )
            elif tracked_items:
                last_overlays, tracked_items = trackers_to_overlays(
                    frame=frame,
                    tracked_items=tracked_items,
                )

            annotated = draw_annotations(
                frame,
                last_overlays,
                last_classifications,
                frame_index=frame_index,
            )
            if writer is not None:
                writer.write(annotated)

            if args.show or args.webcam:
                cv2.imshow("Roboflow Video Workflow", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    capture.release()
    if writer is not None:
        writer.release()
    if args.show or args.webcam:
        cv2.destroyAllWindows()

    print(
        f"Completed: wrote {processed_frames if not args.max_frames else min(processed_frames, args.max_frames)} "
        f"frames, made {inference_calls} workflow calls."
    )

    return {
        "video_path": source_label,
        "output_path": args.output if writer is not None else "",
        "frames_written": processed_frames if not args.max_frames else min(processed_frames, args.max_frames),
        "inference_calls": inference_calls,
    }


def process_video(client, video_path, output_path, args):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video for writing: {output_path}")

    args.output = str(output_path)
    return process_capture(
        client=client,
        capture=capture,
        writer=writer,
        args=args,
        source_label=str(video_path),
        frame_count=frame_count,
        fps=fps,
    )


def process_webcam(client, args):
    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera_index}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open output video for writing: {output_path}")
        args.output = str(output_path)
    else:
        writer = None

    return process_capture(
        client=client,
        capture=capture,
        writer=writer,
        args=args,
        source_label=f"webcam:{args.camera_index}",
        frame_count=0,
        fps=fps,
    )


def main():
    args = build_parser().parse_args()
    if not args.api_key:
        raise RuntimeError("Roboflow API key missing. Set ROBOFLOW_API_KEY or pass --api-key.")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1")

    client = InferenceHTTPClient(api_url=args.api_url, api_key=args.api_key)

    if args.webcam:
        process_webcam(client, args)
        return

    if args.video:
        video_jobs = [
            (
                Path(args.video).expanduser().resolve(),
                Path(args.output).expanduser().resolve()
                if args.output
                else build_output_path(
                    Path(args.video).expanduser().resolve(),
                    Path(args.video).expanduser().resolve().parent / "outputs",
                ),
            )
        ]
    else:
        video_dir = Path(args.video_dir).expanduser().resolve()
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        if not video_dir.is_dir():
            raise NotADirectoryError(f"Video directory is not a folder: {video_dir}")
        output_dir = Path(args.output_dir).expanduser().resolve()
        videos = discover_videos(video_dir)
        if not videos:
            raise RuntimeError(f"No supported video files found in: {video_dir}")
        video_jobs = [(video_path, build_output_path(video_path, output_dir)) for video_path in videos]

    summaries = []
    for index, (video_path, output_path) in enumerate(video_jobs, start=1):
        print(f"\n=== Video {index}/{len(video_jobs)} ===")
        summaries.append(process_video(client, video_path, output_path, args))

    print("\nBatch complete:")
    for summary in summaries:
        print(
            f"{summary['video_path'].name} -> {summary['output_path']} | "
            f"frames={summary['frames_written']} | calls={summary['inference_calls']}"
        )


if __name__ == "__main__":
    main()
