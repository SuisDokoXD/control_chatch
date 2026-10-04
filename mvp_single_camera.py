"""Single-camera minimum runtime for the intelligent sorting vision pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    print("缺少运行依赖。请先执行: python -m pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(2) from exc


ROOT = Path(__file__).resolve().parent
CALIBRATION_PATH = ROOT / "config" / "tray_calibration.json"
OBJECTS_PATH = ROOT / "config" / "objects.json"
REPORT_PATH = ROOT / "reports" / "latest_detections.json"
CAPTURE_DIR = ROOT / "captures"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def classify_color(roi: np.ndarray) -> str:
    """Return a coarse HSV color label; final thresholds need field calibration."""
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    pixels = pixels[(pixels[:, 1] > 35) & (pixels[:, 2] > 35)]
    if len(pixels) == 0:
        value = float(np.mean(hsv[:, :, 2]))
        return "white" if value > 170 else "black"
    hue = float(np.median(pixels[:, 0]))
    saturation = float(np.median(pixels[:, 1]))
    value = float(np.median(pixels[:, 2]))
    if value < 55:
        return "black"
    if saturation < 55:
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
    payload = load_json(path, {})
    try:
        matrix = np.asarray(payload["homography_px_to_mm"], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None
    return matrix if matrix.shape == (3, 3) else None


def pixel_to_tray(point: tuple[float, float], matrix: np.ndarray | None) -> tuple[float, float] | None:
    if matrix is None:
        return None
    source = np.asarray([[point]], dtype=np.float32)
    result = cv2.perspectiveTransform(source, matrix)[0, 0]
    return float(result[0]), float(result[1])


def object_heights(path: Path) -> dict[str, float]:
    payload = load_json(path, {})
    objects = payload.get("objects", {}) if isinstance(payload, dict) else {}
    result: dict[str, float] = {}
    for name, value in objects.items():
        if isinstance(value, dict) and value.get("height_mm") is not None:
            try:
                result[str(name)] = float(value["height_mm"])
            except (TypeError, ValueError):
                continue
    return result


def calibrate(frame: np.ndarray, tray_width_mm: float, tray_height_mm: float) -> np.ndarray | None:
    points: list[tuple[int, int]] = []
    display_width = min(1280, frame.shape[1])
    scale = display_width / frame.shape[1]
    display = cv2.resize(frame, None, fx=scale, fy=scale)
    window = "托盘标定：按顺序点击左上、右上、右下、左下，Enter保存，Esc取消"

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


def load_model(weights: Path, disabled: bool):
    if disabled:
        return None
    if not weights.exists():
        print(f"未找到权重 {weights}，当前仅运行采集、颜色和标定。")
        return None
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        print(f"已加载 YOLO 权重: {weights}")
        return model
    except Exception as exc:
        print(f"YOLO 未加载: {exc}")
        return None


def detect(frame: np.ndarray, model: Any, matrix: np.ndarray | None, heights: dict[str, float], confidence: float):
    annotated = frame.copy()
    detections: list[dict[str, Any]] = []
    if model is None:
        return annotated, detections
    result = model(frame, verbose=False)[0]
    names = result.names
    for box in result.boxes:
        score = float(box.conf[0])
        if score < confidence:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        class_id = int(box.cls[0])
        shape = str(names[class_id])
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        tray_point = pixel_to_tray(center, matrix)
        roi = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        item = {
            "shape": shape,
            "color": classify_color(roi),
            "x_mm": None if tray_point is None else round(tray_point[0], 2),
            "y_mm": None if tray_point is None else round(tray_point[1], 2),
            "z_mm": heights.get(shape),
            "confidence": round(score, 4),
            "bbox_px": [x1, y1, x2, y2],
        }
        detections.append(item)
        label = f"{shape} {item['color']} {score:.2f}"
        if item["x_mm"] is not None:
            label += f" ({item['x_mm']:.0f},{item['y_mm']:.0f})"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(annotated, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)
    return annotated, detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单相机智能分拣视觉最小运行版本")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV摄像头编号")
    parser.add_argument("--image", type=Path, help="离线处理单张图片，不打开摄像头")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLO权重路径")
    parser.add_argument("--objects-config", type=Path, default=OBJECTS_PATH)
    parser.add_argument("--tray-width-mm", type=float, default=300.0)
    parser.add_argument("--tray-height-mm", type=float, default=300.0)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--no-yolo", action="store_true", help="只运行相机、拍照和标定")
    parser.add_argument("--headless", action="store_true", help="不打开窗口，适合离线检查")
    parser.add_argument("--frames", type=int, default=0, help="无窗口读取指定帧数；用于相机连通性测试")
    parser.add_argument("--snapshot", type=Path, help="无窗口采集时保存最后一帧")
    parser.add_argument("--self-test", action="store_true", help="运行逻辑自检后退出")
    return parser.parse_args()


def run_self_test() -> int:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[:] = (0, 0, 255)
    if classify_color(image) != "red":
        print("颜色分类自检失败")
        return 1
    source = np.asarray([[0, 0], [300, 0], [300, 200], [0, 200]], dtype=np.float32)
    destination = np.asarray([[0, 0], [300, 0], [300, 200], [0, 200]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    point = pixel_to_tray((120, 80), matrix)
    if point is None or abs(point[0] - 120) > 0.01 or abs(point[1] - 80) > 0.01:
        print("坐标转换自检失败")
        return 1
    print("MVP逻辑自检通过")
    return 0


def process_image(args: argparse.Namespace, model: Any, calibration: np.ndarray | None, heights: dict[str, float]) -> int:
    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"无法读取图片: {args.image}")
        return 2
    annotated, detections = detect(frame, model, calibration, heights, args.confidence)
    payload = {"timestamp": time.time(), "source": str(args.image), "detections": detections}
    save_json(REPORT_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    output = CAPTURE_DIR / f"inference_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(output), annotated)
    print(f"标注结果图已保存: {output}")
    if not args.headless:
        scale = min(1280 / annotated.shape[1], 1)
        cv2.imshow("单张图片推理", cv2.resize(annotated, None, fx=scale, fy=scale))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration(CALIBRATION_PATH)
    heights = object_heights(args.objects_config)
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    model = load_model(weights, args.no_yolo)

    if args.image:
        return process_image(args, model, calibration, heights)
    if args.headless and args.frames <= 0:
        print("--headless 需要同时指定 --image、--self-test 或 --frames")
        return 2

    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    camera = cv2.VideoCapture(args.camera, backend)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_FPS, args.fps)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if not camera.isOpened():
        print(f"无法打开摄像头编号 {args.camera}。可先运行 camera_probe.py 检查设备。")
        return 2
    if args.headless:
        captured = 0
        started = time.monotonic()
        while captured < args.frames:
            ok, frame = camera.read()
            if not ok:
                print(f"相机在第 {captured + 1} 帧读取失败")
                camera.release()
                return 3
            captured += 1
        elapsed = max(time.monotonic() - started, 1e-6)
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.snapshot), frame)
            print(f"最后一帧已保存: {args.snapshot}")
        print(f"相机读取测试通过: {captured} 帧, {frame.shape[1]}x{frame.shape[0]}, 实测读取速度 {captured / elapsed:.1f} FPS")
        camera.release()
        return 0

    print("按 c 标定，s 保存图片，p 保存识别结果，q 或 Esc 退出。")
    latest: list[dict[str, Any]] = []
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("读取摄像头失败。请检查连接后重新运行。")
                return 3
            annotated, latest = detect(frame, model, calibration, heights, args.confidence)
            display_width = min(1280, annotated.shape[1])
            scale = display_width / annotated.shape[1]
            display = cv2.resize(annotated, None, fx=scale, fy=scale)
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

