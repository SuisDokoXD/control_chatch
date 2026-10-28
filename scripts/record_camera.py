"""Preview and record a UVC camera for dataset collection."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    roles = json.loads((PROJECT_DIR / '配置' / 'camera_roles.json').read_text(encoding='utf-8'))
    settings = json.loads((PROJECT_DIR / '配置' / 'camera_settings.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description="Preview and record the LRCP USB3.0 camera")
    parser.add_argument("--camera", type=int, default=roles['vertical']['index'])
    parser.add_argument("--width", type=int, default=settings['width'])
    parser.add_argument("--height", type=int, default=settings['height'])
    parser.add_argument("--fps", type=float, default=settings['fps'])
    parser.add_argument("--codec", default="MJPG", choices=["MJPG", "YUY2"])
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "dataset" / "videos")
    parser.add_argument("--duration", type=float, default=0, help="Auto-stop seconds; 0 means manual")
    parser.add_argument("--auto-start", action="store_true", help="Start recording immediately")
    parser.add_argument("--headless", action="store_true", help="Record without opening a preview window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.codec))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("Camera opened but returned no frame")

    actual_width = int(frame.shape[1])
    actual_height = int(frame.shape[0])
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    writer_fps = actual_fps if actual_fps > 1 else args.fps
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"capture_{datetime.now():%Y%m%d_%H%M%S}.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), writer_fps,
        (actual_width, actual_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create video: {path}")

    recording = args.auto_start
    written = 0
    started = time.perf_counter()
    print(f"camera={args.camera} actual={actual_width}x{actual_height} requested_fps={args.fps}")
    print("SPACE=start/pause recording, S=save still, ESC=finish")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Lost camera stream")
            if recording:
                writer.write(frame)
                written += 1
            view = frame.copy()
            status = "REC" if recording else "READY"
            color = (0, 0, 255) if recording else (0, 220, 0)
            cv2.putText(view, f"{status}  {actual_width}x{actual_height}  frames={written}",
                        (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            key = 255
            if not args.headless:
                cv2.imshow("LRCP USB3.0 recorder", view)
                key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32:
                recording = not recording
                if recording:
                    started = time.perf_counter()
            if key in (ord("s"), ord("S")):
                still = args.output / f"still_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                cv2.imwrite(str(still), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"still={still}")
            if recording and args.duration > 0 and time.perf_counter() - started >= args.duration:
                break
    finally:
        writer.release()
        capture.release()
        cv2.destroyAllWindows()
    if written == 0:
        path.unlink(missing_ok=True)
        print("No video recorded")
    else:
        print(f"video={path} frames={written}")


if __name__ == "__main__":
    main()
