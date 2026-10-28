"""Extract sharp, sufficiently different color frames and grayscale YOLO copies."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "dataset" / "extracted")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between candidates")
    parser.add_argument("--blur-threshold", type=float, default=80.0)
    parser.add_argument("--duplicate-threshold", type=float, default=4.0,
                        help="Minimum mean absolute difference from last accepted frame")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Video does not report a valid frame rate")
    step = max(1, round(fps * args.interval))
    color_dir = args.output / "color" / "images" / "unlabeled"
    gray_dir = args.output / "yolo_shape" / "images" / "unlabeled"
    color_dir.mkdir(parents=True, exist_ok=True)
    gray_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    previous_small = None
    frame_index = accepted = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step:
            frame_index += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        difference = 255.0 if previous_small is None else float(cv2.absdiff(small, previous_small).mean())
        reason = ""
        if blur < args.blur_threshold:
            reason = "blurred"
        elif difference < args.duplicate_threshold:
            reason = "near_duplicate"
        if not reason:
            name = f"{args.video.stem}_{frame_index:08d}.jpg"
            cv2.imwrite(str(color_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(gray_dir / name), gray_bgr, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            previous_small = small
            accepted += 1
        rows.append({"frame": frame_index, "time_s": f"{frame_index / fps:.3f}",
                     "blur_score": f"{blur:.2f}", "difference": f"{difference:.2f}",
                     "accepted": not reason, "reason": reason})
        frame_index += 1
    capture.release()
    report = args.output / "extraction_report.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else
                                ["frame", "time_s", "blur_score", "difference", "accepted", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"fps={fps:.2f} candidates={len(rows)} accepted={accepted}")
    print(f"annotate={color_dir}")
    print(f"gray_training_copies={gray_dir}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
