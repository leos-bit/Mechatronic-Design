#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--out", required=True, help="Output folder for frames")
    parser.add_argument("--count", type=int, default=200, help="Number of frames to extract")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--step", type=int, default=5, help="Frame step")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    saved = 0

    # Skip to start
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
        frame_idx = args.start

    while saved < args.count:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - args.start) % args.step == 0:
            out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
