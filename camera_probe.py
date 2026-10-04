"""List usable OpenCV camera indexes on Windows."""

from __future__ import annotations

import argparse
import sys
import time

try:
    import cv2
except ModuleNotFoundError as exc:
    print("缺少 OpenCV，请先执行: python -m pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(2) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="探测可用摄像头编号")
    parser.add_argument("--max-index", type=int, default=9)
    parser.add_argument("--show", action="store_true", help="显示每个可用设备的短暂预览")
    args = parser.parse_args()
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    found = 0
    for index in range(args.max_index + 1):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            found += 1
            print(f"设备编号 {index}: {frame.shape[1]}x{frame.shape[0]}, 实际FPS约 {cap.get(cv2.CAP_PROP_FPS):.1f}")
            if args.show:
                window = f"camera {index}"
                cv2.imshow(window, cv2.resize(frame, None, fx=0.5, fy=0.5))
                cv2.waitKey(800)
                cv2.destroyWindow(window)
        cap.release()
        time.sleep(0.1)
    if not found:
        print("没有探测到可用摄像头。请关闭 OBS 等占用摄像头的程序后重试。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

