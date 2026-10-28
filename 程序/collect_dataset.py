"""快捷拍照数据集采集器：按数字选类别，空格保存并自动归档。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def open_camera(index: int):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in dict.fromkeys(backends):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


def load_json(path: Path, default: dict) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="按类别快捷拍照并自动归档")
    parser.add_argument("--classes", default="cube", help="逗号分隔类别，最多9个")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset" / "yolo_cube" / "images" / "raw")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--role", default="vertical")
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    classes = [x.strip() for x in args.classes.split(",") if x.strip()]
    if not 1 <= len(classes) <= 9:
        raise SystemExit("类别数量必须为1到9个")
    roles = load_json(ROOT / "配置" / "camera_roles.json", {})
    role = roles.get(args.role, {})
    index = args.camera if args.camera is not None else int(role.get("index", 0))
    settings = load_json(ROOT / "配置" / "camera_settings.json", {})
    cap = open_camera(index)
    if cap is None:
        raise SystemExit(f"无法打开摄像头编号 {index}（已尝试 DirectShow、Media Foundation 和默认后端）")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(settings.get("fourcc", "MJPG"))[:4]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(settings.get("width", 1920)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(settings.get("height", 1080)))
    cap.set(cv2.CAP_PROP_FPS, float(settings.get("fps", 30)))
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_root = args.output.parents[1]
    for folder in (
        dataset_root / "labels" / "raw",
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
        dataset_root / "labels" / "test",
    ):
        folder.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "采集记录.csv"
    new_log = not log_path.exists()
    log = log_path.open("a", newline="", encoding="utf-8-sig")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["filename", "class", "camera_index", "width", "height", "timestamp"])
    selected = 0
    window = "Dataset Capture"
    print("Keys: 1-9 select class; SPACE save; N add class; Q/ESC quit")
    print(" | ".join(f"{i + 1}:{x}" for i, x in enumerate(classes)))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取摄像头失败")
                return 2
            canvas = frame.copy()
            text = f"Class: {classes[selected]} | SPACE save | N add | Q quit"
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 44), (20, 20, 20), -1)
            cv2.putText(canvas, text, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            scale = min(1280 / canvas.shape[1], 1.0)
            cv2.imshow(window, cv2.resize(canvas, None, fx=scale, fy=scale))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("9") and key - ord("1") < len(classes):
                selected = key - ord("1")
                print(f"Selected class: {classes[selected]}")
            elif key == ord("n"):
                cv2.destroyWindow(window)
                value = input("Class name: ").strip()
                if value and value not in classes and len(classes) < 9:
                    classes.append(value)
                    selected = len(classes) - 1
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            elif key == 32:
                name = classes[selected]
                existing = list(args.output.glob(f"{name}_*.jpg"))
                number = len(existing) + 1
                while (args.output / f"{name}_{number:04d}.jpg").exists():
                    number += 1
                filename = f"{name}_{number:04d}.jpg"
                path = args.output / filename
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                writer.writerow([str(path.relative_to(args.output)), name, index, frame.shape[1], frame.shape[0], time.time()])
                log.flush()
                print(f"已保存: {path}")
    finally:
        log.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
