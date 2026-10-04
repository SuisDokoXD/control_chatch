"""Single-camera MVP for the intelligent sorting vision pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
CALIBRATION_PATH = ROOT / "config" / "tray_calibration.json"
REPORT_PATH = ROOT / "reports" / "latest_detections.json"
CAPTURE_DIR = ROOT / "captures"


def classify_color(roi: np.ndarray) -> str:
    """Return a coarse color label for a non-empty BGR region."""
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    pixels = pixels[(pixels[:, 1] > 35) & (pixels[:, 2] > 35)]
    if len(pixels) == 0:
        value = float(np.mean(hsv[:, :, 2]))
        return "white" if value > 170 else "black"
    hue = float(np.median(pixels[:, 0]))
    sat = float(np.median(pixels[:, 1]))
    value = float(np.median(pixels[:, 2]))
    if value < 55:
        return "black"
    if sat < 55:
        return "white" if value > 160 else "gray"
    if hue < 10 or hue >= 170:
        return "red"
    if hue < 25:
        return "orange"
    if hue < 38:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 130:
        return "blue"
    return "purple"


def load_calibration(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        matrix = np.asarray(payload["homography_px_to_mm"], dtype=np.float32)
        return matrix if matrix.shape == (3, 3) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def pixel_to_tray(point: tuple[float, float], matrix: np.ndarray | None) -> tuple[float, float] | None:
    if matrix is None:
        return None
    source = np.asarray([[point]], dtype=np.float32)
    result = cv2.perspectiveTransform(source, matrix)[0, 0]
    return float(result[0]), float(result[1])


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def calibrate(frame: np.ndarray, tray_width_mm: float, tray_height_mm: float) -> np.ndarray | None:
    points: list[tuple[int, int]] = []
    display_width = min(1280, frame.shape[1])
    scale = display_width / frame.shape[1]
    display = cv2.resize(frame, None, fx=scale, fy=scale)
    window = "托盘标定：按顺序点击四角，Enter保存，Esc取消"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((round(x / scale), round(y / scale)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = display.copy()
        for index, (x, y) in enumerate(points, start=1):
            px, py = round(x * scale), round(y * scale)
            cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(canvas, str(index), (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(window)
            return None
        if key in (10, 13) and len(points) == 4:
            source = np.asarray(points, dtype=np.float32)
            destination = np.asarray(
                [[0, 0], [tray_width_mm, 0], [tray_width_mm, tray_height_mm], [0, tray_height_mm]],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(source, destination)
            save_json(
                CALIBRATION_PATH,
                {
                    "tray_size_mm": [tray_width_mm, tray_height_mm],
                    "corners_px": points,
                    "homography_px_to_mm": matrix.tolist(),
                },
            )
            cv2.destroyWindow(window)
            return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单相机智能分拣视觉最小运行版本")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV摄像头编号")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLO权重路径；不存在时只运行采集和标定")
    parser.add_argument("--tray-width-mm", type=float, default=300.0)
    parser.add_argument("--tray-height-mm", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration(CALIBRATION_PATH)

    model = None
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    if weights.exists():
        try:
            from ultralytics import YOLO

            model = YOLO(str(weights))
            print(f"已加载 YOLO 权重: {weights}")
        except Exception as exc:  # optional dependency/model errors should not block preview
            print(f"YOLO 未加载，将继续运行采集模式: {exc}")
    else:
        print(f"未找到权重 {weights}，按 s/c 仍可拍照和标定。")

    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    camera = cv2.VideoCapture(args.camera, backend)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_FPS, args.fps)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if not camera.isOpened():
        print(f"无法打开摄像头编号 {args.camera}")
        return 2

    print("按 c 标定，s 保存图片，p 保存识别结果，q 或 Esc 退出。")
    latest: list[dict[str, Any]] = []
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("读取摄像头失败。请检查连接后重新运行。")
                break
            annotated = frame.copy()
            latest = []
            if model is not None:
                result = model(frame, verbose=False)[0]
                names = result.names
                for box in result.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    tray_point = pixel_to_tray(center, calibration)
                    color = classify_color(frame[max(0, y1):y2, max(0, x1):x2])
                    item = {
                        "shape": str(names[class_id]),
                        "color": color,
                        "x_mm": None if tray_point is None else round(tray_point[0], 2),
                        "y_mm": None if tray_point is None else round(tray_point[1], 2),
                        "z_mm": None,
                        "confidence": round(confidence, 4),
                    }
                    latest.append(item)
                    label = f"{item['shape']} {color} {confidence:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(annotated, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)

            display_width = min(1280, annotated.shape[1])
            display = cv2.resize(annotated, None, fx=display_width / annotated.shape[1], fy=display_width / annotated.shape[1])
            cv2.imshow("单相机视觉最小运行版本", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                matrix = calibrate(frame, args.tray_width_mm, args.tray_height_mm)
                if matrix is not None:
                    calibration = matrix
                    print(f"标定已保存: {CALIBRATION_PATH}")
            elif key == ord("s"):
                path = CAPTURE_DIR / f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"图片已保存: {path}")
            elif key == ord("p"):
                payload = {"timestamp": time.time(), "camera": args.camera, "detections": latest}
                save_json(REPORT_PATH, payload)
                print(f"识别结果已保存: {REPORT_PATH}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

