"""Capture and inspect the vertical plus two side-camera views."""

from __future__ import annotations

import argparse
import json
import sys
import time
import math
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    print("缺少 OpenCV/NumPy，请先执行: python -m pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(2) from exc


ROOT = Path(__file__).resolve().parent.parent
CAMERA_ROLES = ("vertical", "side_a", "side_b")
CAMERA_ROLES_PATH = ROOT / "配置" / "camera_roles.json"
CAMERA_SETTINGS_PATH = ROOT / "配置" / "camera_settings.json"
DEFAULT_OUTPUT = ROOT / "captures" / "three_camera"


@dataclass(frozen=True)
class CameraSpec:
    role: str
    index: int
    device_name: str
    description: str = ""


@dataclass(frozen=True)
class FrameBundle:
    sequence: int
    frames: dict[str, np.ndarray]
    grab_times: dict[str, float]
    captured_at_utc: str

    @property
    def grab_spread_ms(self) -> float:
        if len(self.grab_times) < 2:
            return 0.0
        return (max(self.grab_times.values()) - min(self.grab_times.values())) * 1000.0


class FrameCaptureError(RuntimeError):
    """A camera failed while a three-view frame bundle was being captured."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON 配置 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 配置必须是对象: {path}")
    return payload


def resolve_camera_specs(
    roles_config: Mapping[str, Any],
    overrides: Mapping[str, int | None] | None = None,
    *,
    config_path: Path = CAMERA_ROLES_PATH,
) -> dict[str, CameraSpec]:
    """Resolve the three logical camera roles without guessing device indexes."""
    overrides = overrides or {}
    specs: dict[str, CameraSpec] = {}
    used_indices: dict[int, str] = {}

    for role in CAMERA_ROLES:
        entry = roles_config.get(role, {})
        if not isinstance(entry, Mapping):
            raise ValueError(f"相机配置 {role} 必须是对象")

        override = overrides.get(role)
        index = override if override is not None else entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(
                f"未配置 {role} 的有效摄像头编号；请在 {config_path} 设置 index，"
                f"或使用对应的 --{role.replace('_', '-')}-camera 参数。"
            )
        if index in used_indices:
            other_role = used_indices[index]
            raise ValueError(
                f"相机编号必须不同 (unique)：{other_role} 和 {role} 都配置为 {index}。"
            )

        device_name_value = entry.get("device_name")
        device_name = str(device_name_value).strip() if device_name_value else role
        description_value = entry.get("description")
        description = str(description_value).strip() if description_value else ""
        specs[role] = CameraSpec(role, index, device_name, description)
        used_indices[index] = role

    return specs


def capture_frame_bundle(
    cameras: Mapping[str, Any],
    sequence: int,
    clock: Callable[[], float] = time.monotonic,
) -> FrameBundle:
    """Grab all cameras first, then retrieve frames to reduce software skew.

    ``grab_times`` records when each OpenCV ``grab`` call returned. It is not an
    exposure timestamp and does not imply hardware synchronization.
    """
    missing = [role for role in CAMERA_ROLES if role not in cameras]
    if missing:
        raise FrameCaptureError(f"缺少相机句柄: {', '.join(missing)}")

    grab_times: dict[str, float] = {}
    for role in CAMERA_ROLES:
        if not cameras[role].grab():
            raise FrameCaptureError(f"camera {role} grab() failed")
        grab_times[role] = clock()

    frames: dict[str, np.ndarray] = {}
    for role in CAMERA_ROLES:
        ok, frame = cameras[role].retrieve()
        if not ok or frame is None:
            raise FrameCaptureError(f"camera {role} retrieve() failed")
        frames[role] = frame

    return FrameBundle(
        sequence=sequence,
        frames=frames,
        grab_times=grab_times,
        captured_at_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    )


def compose_preview(
    frames: Mapping[str, np.ndarray],
    *,
    labels: Mapping[str, str] | None = None,
    target_height: int = 480,
    max_width: int = 1440,
) -> np.ndarray:
    """Resize and join the three BGR views into a labeled preview image."""
    if target_height <= 0 or max_width <= 0:
        raise ValueError("预览宽度和高度必须为正数")
    missing = [role for role in CAMERA_ROLES if role not in frames]
    if missing:
        raise ValueError(f"预览缺少相机图像: {', '.join(missing)}")

    source_frames: list[np.ndarray] = []
    widths_at_target_height: list[int] = []
    for role in CAMERA_ROLES:
        frame = frames[role]
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"{role} 图像必须是三通道 BGR 图像")
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"{role} 图像尺寸无效")
        source_frames.append(frame)
        widths_at_target_height.append(max(1, round(width * target_height / height)))

    total_width = sum(widths_at_target_height)
    shrink = min(1.0, max_width / total_width)
    tile_height = max(1, round(target_height * shrink))
    tiles: list[np.ndarray] = []
    labels = labels or {}

    for role, frame, source_width in zip(CAMERA_ROLES, source_frames, widths_at_target_height):
        tile_width = max(1, round(source_width * shrink))
        tile = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        header_height = min(32, tile_height)
        cv2.rectangle(tile, (0, 0), (tile_width - 1, header_height - 1), (20, 20, 20), -1)
        cv2.putText(
            tile,
            labels.get(role, role),
            (8, max(14, header_height - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    preview = cv2.hconcat(tiles)
    if preview.shape[1] > max_width:
        preview = cv2.resize(
            preview,
            (max_width, max(1, round(preview.shape[0] * max_width / preview.shape[1]))),
            interpolation=cv2.INTER_AREA,
        )
    return preview


def save_frame_bundle(
    bundle: FrameBundle,
    output_dir: Path,
    specs: Mapping[str, CameraSpec],
    *,
    jpeg_quality: int = 95,
) -> Path:
    """Atomically save one three-view set; incomplete sets are never published."""
    if not 0 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 0 and 100")

    missing = [
        role for role in CAMERA_ROLES
        if bundle.frames.get(role) is None or specs.get(role) is None
    ]
    if missing:
        raise ValueError(f"帧组或相机配置缺少: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / f"frame_{bundle.sequence:06d}"
    if frame_dir.exists():
        raise FileExistsError(f"帧组目录已存在，拒绝覆盖: {frame_dir}")
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{frame_dir.name}.", dir=str(output_dir)))
    try:
        cameras_metadata: dict[str, dict[str, Any]] = {}
        for role in CAMERA_ROLES:
            frame = bundle.frames[role]
            spec = specs[role]
            filename = f"{role}.jpg"
            image_path = temp_dir / filename
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                raise OSError(f"无法保存图像: {image_path}")
            cameras_metadata[role] = {
                "index": spec.index,
                "device_name": spec.device_name,
                "image": filename,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            }

        metadata = {
            "sequence": bundle.sequence,
            "captured_at_utc": bundle.captured_at_utc,
            "sync_method": "sequential_grab_then_retrieve",
            "hardware_synchronized": False,
            "grab_spread_ms": round(bundle.grab_spread_ms, 3),
            "grab_return_times_monotonic_s": bundle.grab_times,
            "cameras": cameras_metadata,
        }
        (temp_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp_dir.rename(frame_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return frame_dir


def _camera_backends() -> tuple[int, ...]:
    candidates = (
        getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
        cv2.CAP_ANY,
    )
    return tuple(dict.fromkeys(candidates))


def _backend_name(backend: int) -> str:
    names = {
        getattr(cv2, "CAP_DSHOW", -1): "DirectShow",
        getattr(cv2, "CAP_MSMF", -2): "Media Foundation",
        cv2.CAP_ANY: "automatic",
    }
    return names.get(backend, str(backend))


def open_camera(spec: CameraSpec) -> tuple[Any, int]:
    """Open one camera, trying Windows capture backends and checking a frame."""
    failures: list[str] = []
    for backend in _camera_backends():
        capture = cv2.VideoCapture(spec.index, backend)
        if capture.isOpened():
            ok, frame = capture.read()
            if ok and frame is not None:
                return capture, backend
            failures.append(f"{_backend_name(backend)} opened but returned no frame")
        else:
            failures.append(f"{_backend_name(backend)} could not open")
        capture.release()
    details = "; ".join(failures)
    raise RuntimeError(
        f"无法打开 {spec.role} 相机（编号 {spec.index}，{spec.device_name}）：{details}。"
        "请运行 程序/camera_probe.py 检查设备，并关闭占用相机的程序。"
    )


def open_cameras(specs: Mapping[str, CameraSpec]) -> tuple[dict[str, Any], dict[str, int]]:
    cameras: dict[str, Any] = {}
    backends: dict[str, int] = {}
    try:
        for role in CAMERA_ROLES:
            camera, backend = open_camera(specs[role])
            cameras[role] = camera
            backends[role] = backend
    except Exception:
        for camera in cameras.values():
            camera.release()
        raise
    return cameras, backends


def resolve_capture_format(
    settings: Mapping[str, Any],
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
) -> tuple[int, int, float]:
    try:
        requested_width = int(width if width is not None else settings.get("width", 1920))
        requested_height = int(height if height is not None else settings.get("height", 1080))
        requested_fps = float(fps if fps is not None else settings.get("fps", 30))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("相机宽度、高度和帧率必须是有效数字") from exc
    if requested_width <= 0 or requested_height <= 0 or not math.isfinite(requested_fps) or requested_fps <= 0:
        raise ValueError("相机宽度、高度和帧率必须为正数")
    return requested_width, requested_height, requested_fps


def configure_camera(
    camera: Any,
    settings: Mapping[str, Any],
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    apply_exposure_settings: bool = False,
) -> dict[str, float]:
    """Set common format properties; exposure/WB changes require explicit opt-in."""
    requested_width, requested_height, requested_fps = resolve_capture_format(
        settings, width=width, height=height, fps=fps
    )
    fourcc = str(settings.get("fourcc", "MJPG"))[:4].ljust(4, " ")

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(requested_width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(requested_height))
    camera.set(cv2.CAP_PROP_FPS, float(requested_fps))

    if apply_exposure_settings:
        auto_exposure = settings.get("auto_exposure")
        if auto_exposure is not None:
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto_exposure else 0.25)
        auto_white_balance = settings.get("auto_white_balance")
        if auto_white_balance is not None:
            camera.set(cv2.CAP_PROP_AUTO_WB, 1 if auto_white_balance else 0)
        for key, property_id in (
            ("exposure", cv2.CAP_PROP_EXPOSURE),
            ("gain", cv2.CAP_PROP_GAIN),
            ("white_balance", cv2.CAP_PROP_WB_TEMPERATURE),
        ):
            value = settings.get(key)
            if value is not None:
                camera.set(property_id, float(value))

    return {
        "width": camera.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps": camera.get(cv2.CAP_PROP_FPS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="垂直相机 + 两台侧面相机的采集和预览（软件近似帧组，不提供硬件同步或融合）"
    )
    parser.add_argument("--vertical-camera", type=int, help="垂直相机编号；默认读取角色配置")
    parser.add_argument("--side-a-camera", type=int, help="第一台侧面相机编号；默认读取角色配置")
    parser.add_argument("--side-b-camera", type=int, help="第二台侧面相机编号；默认读取角色配置")
    parser.add_argument("--camera-roles", type=Path, default=CAMERA_ROLES_PATH)
    parser.add_argument("--camera-settings", type=Path, default=CAMERA_SETTINGS_PATH)
    parser.add_argument("--width", type=int, help="覆盖配置中的宽度")
    parser.add_argument("--height", type=int, help="覆盖配置中的高度")
    parser.add_argument("--fps", type=float, help="覆盖配置中的帧率")
    parser.add_argument("--apply-camera-settings", action="store_true", help="明确应用配置中的曝光/增益/白平衡")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--headless", action="store_true", help="不打开预览窗口，需配合 --frames")
    parser.add_argument("--frames", type=int, default=0, help="采集帧组数；0 表示预览模式持续采集")
    parser.add_argument("--preview-height", type=int, default=480)
    parser.add_argument("--max-preview-width", type=int, default=1440)
    parser.add_argument("--sync-warning-ms", type=float, default=25.0, help="软件 grab 返回时间跨度提示阈值")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--self-test", action="store_true", help="运行合成图像逻辑自检，不连接摄像头")
    return parser


def run_self_test() -> int:
    class SyntheticCamera:
        def __init__(self, value: int) -> None:
            self.frame = np.full((60, 80, 3), value, dtype=np.uint8)

        def grab(self) -> bool:
            return True

        def retrieve(self) -> tuple[bool, np.ndarray]:
            return True, self.frame

    cameras = {role: SyntheticCamera(number * 40) for number, role in enumerate(CAMERA_ROLES, 1)}
    bundle = capture_frame_bundle(cameras, sequence=1)
    preview = compose_preview(bundle.frames, target_height=120, max_width=640)
    if tuple(bundle.frames) != CAMERA_ROLES or preview.shape[2] != 3:
        print("三相机逻辑自检失败", file=sys.stderr)
        return 1
    print("三相机采集逻辑自检通过（合成图像；未连接实际相机，也不代表硬件同步）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.frames < 0 or args.sync_warning_ms < 0:
        parser.error("--frames 和 --sync-warning-ms 不能为负数")
    if args.headless and args.frames == 0:
        parser.error("--headless 需要指定正数 --frames")
    if any(value is not None and value <= 0 for value in (args.width, args.height, args.fps)):
        parser.error("--width、--height 和 --fps 必须为正数")
    if args.preview_height <= 0 or args.max_preview_width <= 0:
        parser.error("预览宽度和高度必须为正数")
    if args.jpeg_quality < 0 or args.jpeg_quality > 100:
        parser.error("--jpeg-quality 必须在 0 到 100 之间")

    try:
        roles_config = load_json(args.camera_roles)
        settings = load_json(args.camera_settings)
        resolve_capture_format(settings, width=args.width, height=args.height, fps=args.fps)
        specs = resolve_camera_specs(
            roles_config,
            {
                "vertical": args.vertical_camera,
                "side_a": args.side_a_camera,
                "side_b": args.side_b_camera,
            },
            config_path=args.camera_roles,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        cameras, backends = open_cameras(specs)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 2

    output_root = args.output if args.output.is_absolute() else ROOT / args.output
    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
    session_dir = output_root / session_name
    try:
        for role in CAMERA_ROLES:
            actual = configure_camera(
                cameras[role],
                settings,
                width=args.width,
                height=args.height,
                fps=args.fps,
                apply_exposure_settings=args.apply_camera_settings,
            )
            spec = specs[role]
            print(
                f"{role}: index={spec.index} ({spec.device_name}), backend={_backend_name(backends[role])}, "
                f"actual={actual['width']:.0f}x{actual['height']:.0f}@{actual['fps']:.1f} FPS"
            )

        if not args.headless:
            print("预览按 S 保存当前三路帧组，Q/Esc 退出。软件 grab 顺序不等同于硬件同步。")

        sequence = 0
        while args.frames == 0 or sequence < args.frames:
            sequence += 1
            try:
                bundle = capture_frame_bundle(cameras, sequence)
            except FrameCaptureError as exc:
                print(f"采集失败: {exc}", file=sys.stderr)
                return 3

            if bundle.grab_spread_ms > args.sync_warning_ms and (sequence == 1 or sequence % 60 == 0):
                print(
                    f"警告：第 {sequence} 组 grab() 返回跨度 {bundle.grab_spread_ms:.1f} ms；"
                    "这不是曝光时间差，也不能证明三台相机硬件同步。",
                    file=sys.stderr,
                )

            if args.headless:
                saved_dir = save_frame_bundle(bundle, session_dir, specs, jpeg_quality=args.jpeg_quality)
                print(f"已保存帧组 {sequence}: {saved_dir}")
                continue

            preview_labels = {
                role: f"{role} [{specs[role].index}]" for role in CAMERA_ROLES
            }
            preview = compose_preview(
                bundle.frames,
                labels=preview_labels,
                target_height=args.preview_height,
                max_width=args.max_preview_width,
            )
            cv2.imshow("三相机采集预览", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("s"), ord("S")):
                saved_dir = save_frame_bundle(bundle, session_dir, specs, jpeg_quality=args.jpeg_quality)
                print(f"已保存帧组 {sequence}: {saved_dir}")
            if args.frames > 0 and sequence >= args.frames:
                break
    finally:
        for camera in cameras.values():
            camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
