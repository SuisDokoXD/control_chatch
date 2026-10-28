"""双相机（垂直 + 侧视）联检：按配置最大分辨率抓帧，检查托盘定位，并报告标定状态。

用法：
    python scripts/camera_pair_check.py                 # 读 配置/camera_roles.json
    python scripts/camera_pair_check.py --vertical 2 --side 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from 程序 import camera_config  # noqa: E402
from 程序 import opencv_shape_vision as vision  # noqa: E402

OUT_DIR = ROOT / "captures" / "camera_pair"
ROLES = ROOT / "配置" / "camera_roles.json"
SETTINGS = ROOT / "配置" / "camera_settings.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def open_camera(index: int, settings: dict, warmup: int = 15):
    """按配置的 fourcc/分辨率/曝光打开相机，返回 (cap, 最近一帧, 实际参数)。"""
    fourcc = str(settings.get("fourcc") or "MJPG")[:4].ljust(4, " ")
    width = int(settings.get("width") or 2592)
    height = int(settings.get("height") or 1944)
    fps = float(settings.get("fps") or 15)
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if settings.get("fps"):
            cap.set(cv2.CAP_PROP_FPS, float(settings["fps"]))
        if settings.get("auto_exposure") is False and settings.get("exposure") is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, float(settings["exposure"]))
        if settings.get("auto_white_balance") is False and settings.get("white_balance") is not None:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(settings["white_balance"]))
        frame = None
        for _ in range(warmup):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
            time.sleep(0.02)
        if frame is None:
            cap.release()
            continue
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 1),
        }
        return cap, frame, info
    return None, None, None


def grab_pair(vertical_cap, side_cap):
    """尽量同时取两帧：先各自 grab 再 retrieve，缩小时间差。"""
    ok_v = vertical_cap.grab()
    ok_s = side_cap.grab()
    t0 = time.perf_counter()
    frame_v = vertical_cap.retrieve()[1] if ok_v else None
    frame_s = side_cap.retrieve()[1] if ok_s else None
    spread_ms = (time.perf_counter() - t0) * 1000.0
    return frame_v, frame_s, spread_ms


def analyse_vertical(frame: np.ndarray, cfg: dict) -> dict:
    """在垂直视角上找托盘四角黑点，并评估四边形离"矩形"有多远。"""
    small = cv2.resize(frame, None, fx=cfg["scale"], fy=cfg["scale"],
                       interpolation=cv2.INTER_AREA)
    quad = vision.find_tray_quad(small, cfg)
    cands = vision.find_dot_candidates(small, cfg)
    result: dict = {
        "dots_found": len(cands),
        "dot_ring": [round(c[2], 1) for c in cands[:4]],
        "quad": None,
        "quad_full": None,
        "aspect": None,
        "corner_angles": None,
    }
    if quad is not None:
        full = quad / cfg["scale"]
        result["quad_full"] = [[round(float(x), 1), round(float(y), 1)] for x, y in full]
        # 边长与对角线：矩形的话对边应等长、内角应接近 90°
        pts = full.astype(np.float64)
        sides = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
        result["aspect"] = round(max(sides[0], sides[1]) / max(min(sides[0], sides[1]), 1e-6), 3)
        angles = []
        for i in range(4):
            a, b, c = pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]
            v1, v2 = a - b, c - b
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
            angles.append(round(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))), 1))
        result["corner_angles"] = angles
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="双相机联检")
    parser.add_argument("--vertical", type=int, help="垂直相机编号；默认读角色配置")
    parser.add_argument("--side", type=int, help="侧视相机编号；默认读角色配置")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    roles = load_json(ROLES)
    settings = load_json(SETTINGS)
    v_index = args.vertical if args.vertical is not None else roles.get("vertical", {}).get("index")
    s_index = args.side if args.side is not None else roles.get("side_a", {}).get("index")
    if v_index is None or s_index is None:
        print("缺少相机编号：请在 配置/camera_roles.json 里设置，或用 --vertical/--side 指定。",
              file=sys.stderr)
        return 2
    # 每台相机各自解析参数：顶层默认值 + per_camera.<角色> 覆盖
    v_settings = camera_config.resolve(settings, "vertical", int(v_index))
    s_settings = camera_config.resolve(settings, "side_a", int(s_index))
    print(f"垂直 {camera_config.describe(v_settings)}")
    print(f"侧视 {camera_config.describe(s_settings)}")

    v_cap, v_frame, v_info = open_camera(int(v_index), v_settings)
    if v_cap is None:
        print(f"打不开垂直相机 {v_index}", file=sys.stderr)
        return 1
    s_cap, s_frame, s_info = open_camera(int(s_index), s_settings)
    if s_cap is None:
        v_cap.release()
        print(f"打不开侧视相机 {s_index}", file=sys.stderr)
        return 1

    try:
        v_frame, s_frame, spread = grab_pair(v_cap, s_cap)
        if v_frame is None or s_frame is None:
            print("抓帧失败", file=sys.stderr)
            return 1
        print(f"垂直：{v_info['width']}x{v_info['height']} @{v_info['fps']}fps　"
              f"侧视：{s_info['width']}x{s_info['height']} @{s_info['fps']}fps　"
              f"两帧 retrieve 时间差 {spread:.1f} ms")
        cv2.imwrite(str(out / "vertical.jpg"), v_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(out / "side.jpg"), s_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        cfg = dict(vision.DEFAULTS)
        report = analyse_vertical(v_frame, cfg)
        print(f"\n垂直视角托盘定位：找到 {report['dots_found']} 个角点候选，"
              f"环带亮度 {report['dot_ring']}")
        if report["quad_full"]:
            print(f"  托盘四边形（原图像素）：{report['quad_full']}")
            print(f"  对边比 {report['aspect']}（矩形应≈1.0）")
            print(f"  四个内角 {report['corner_angles']}°（矩形应≈90°）")
            for a in report["corner_angles"] or []:
                if abs(a - 90.0) > 3.0:
                    print(f"  ⚠ 有内角偏离 90° 超过 3°，说明垂直相机相对托盘有透视/倾斜")
                    break
        else:
            print("  ⚠ 没找到托盘四边形——请确认垂直相机对着托盘、四角黑点没被挡住")

        overlay = v_frame.copy()
        if report["quad_full"]:
            cv2.polylines(overlay, [np.int32(np.array(report["quad_full"]))], True, (0, 0, 255), 6)
        cv2.imwrite(str(out / "vertical_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

        (out / "report.json").write_text(json.dumps(
            {"vertical": v_info, "side": s_info, "spread_ms": round(spread, 2),
             "vertical_tray": report}, ensure_ascii=False, indent=1), encoding="utf-8")

        html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>双相机联检</title>
<style>
 body {{ margin:0; background:#14171a; color:#e8eaed; font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif; }}
 h1 {{ font-size:17px; margin:12px 16px 4px; }} h2 {{ font-size:15px; margin:18px 16px 4px; color:#8ab4f8; }}
 p {{ margin:0 16px 8px; color:#9aa0a6; }}
 img {{ width:min(88vw,1100px); display:block; margin:0 16px 16px; border-radius:8px; border:1px solid #2c3238; }}
 code {{ background:#22272c; padding:1px 5px; border-radius:4px; }}
</style></head><body>
<h1>双相机联检结果</h1>
<p>垂直 = 编号 {v_index}（{v_info['width']}×{v_info['height']}）　侧视 = 编号 {s_index}（{s_info['width']}×{s_info['height']}）</p>
<h2>垂直相机 + 托盘四边形（红）</h2>
<img src="vertical_overlay.jpg">
<h2>侧视相机</h2>
<img src="side.jpg">
</body></html>
"""
        (out / "index.html").write_text(html, encoding="utf-8")
        print(f"\n页面：{out / 'index.html'}")
    finally:
        v_cap.release()
        s_cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
