"""端到端交叉验证：把托盘平面从垂直相机投影到侧视相机，看是否落在托盘上。

这是整条标定链唯一的端到端检验，一次用上三样东西：
    内参（两台相机）+ 外参（R/t）+ 托盘平面（4 角黑点单应）

做法：
1. 两台各抓一帧（托盘可见、平板上已拿走）
2. 垂直帧里找 4 个黑点 → 和已存的角点比对 → solvePnP 求"托盘平面 → 垂直相机"的位姿
3. 用外参把托盘四角 + 网格转到侧视相机坐标系 → 投影成像素
4. 侧视帧里也找 4 个黑点 → 与投影位置对比，残差就是整条链的一致性误差
5. 出图：侧视画面叠投影轮廓 + 20mm 网格 + 投影点(绿) / 实测点(洋红)

用法：
    python scripts/verify_tray_stereo.py
    python scripts/verify_tray_stereo.py --exposure -4 --out captures/calibration/tray_stereo_check.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from 程序 import camera_config  # noqa: E402
from 程序 import overlay_text  # noqa: E402
from calibrate_tray_plane import find_dots, open_camera, order_points, pick_four  # noqa: E402

DEFAULT_JSON = ROOT / "配置" / "calibration_stereo.json"
DEFAULT_OUT = ROOT / "captures" / "calibration" / "tray_stereo_check.jpg"


def grab(role: str, exposure: float | None):
    settings = camera_config.load_settings()
    roles = camera_config.load_roles()
    resolved = camera_config.resolve(settings, role, roles.get(role, {}).get("index"))
    if exposure is not None:
        resolved["auto_exposure"] = False
        resolved["exposure"] = exposure
    cap = open_camera(resolved)
    if cap is None:
        return None, resolved
    frame = None
    for _ in range(30):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    cap.release()
    return frame, resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="托盘平面跨相机交叉验证")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--exposure", type=float, default=-4.0,
                        help="两台相机都用这个曝光抓托盘（托盘比平板暗，默认 -4）")
    args = parser.parse_args(argv)

    json_path = args.json if args.json.is_absolute() else ROOT / args.json
    data = json.loads(json_path.read_text(encoding="utf-8"))
    tray = data["tray"]
    W, H = tray["size_mm"]
    Kv = np.array(data["intrinsics"]["vertical"]["mtx"], np.float64)
    Dv = np.array(data["intrinsics"]["vertical"]["dist"], np.float64)
    Ks = np.array(data["intrinsics"]["side_a"]["mtx"], np.float64)
    Ds = np.array(data["intrinsics"]["side_a"]["dist"], np.float64)
    R = np.array(data["extrinsics"]["R"], np.float64)
    T = np.array(data["extrinsics"]["T"], np.float64).reshape(3, 1)
    print(f"标定：基线 {data['extrinsics']['baseline_mm']:.1f}mm  "
          f"托盘 {W}×{H}mm  图像 {data['image_size']}")

    frame_v, res_v = grab("vertical", args.exposure)
    frame_s, res_s = grab("side_a", args.exposure)
    if frame_v is None or frame_s is None:
        print("抓帧失败", file=sys.stderr)
        return 1
    print(f"垂直 {camera_config.describe(res_v)}")
    print(f"侧视 {camera_config.describe(res_s)}")

    # ---- 垂直帧：黑点 → 和已存角点比对 → 求托盘平面位姿
    dots_v = find_dots(cv2.cvtColor(frame_v, cv2.COLOR_BGR2GRAY))
    if len(dots_v) < 4:
        print(f"垂直帧只找到 {len(dots_v)} 个黑点，无法验证", file=sys.stderr)
        return 1
    picked = pick_four(dots_v, frame_v.shape[:2])
    if picked is None:
        print(f"垂直帧只找到 {len(dots_v)} 个黑点、凑不齐四个象限，无法验证", file=sys.stderr)
        return 1
    pts_v = order_points(np.array([picked[k]["center"] for k in ("tl", "tr", "br", "bl")],
                                  np.float32))
    saved = np.array([tray["corners_px"][k] for k in ("TL", "TR", "BR", "BL")], np.float32)
    print(f"\n垂直帧黑点 vs 标定时存的角点（应 < 5px）：")
    for name, a, b in zip(("TL", "TR", "BR", "BL"), pts_v, saved):
        print(f"  {name} 现在 ({a[0]:7.1f},{a[1]:7.1f})  标定时 ({b[0]:7.1f},{b[1]:7.1f})  "
              f"差 {np.linalg.norm(a - b):5.2f}px")

    obj = np.float32([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]])
    ok, rvec, tvec = cv2.solvePnP(obj, pts_v.reshape(-1, 1, 2), Kv, Dv, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("solvePnP 失败", file=sys.stderr)
        return 1
    proj_v, _ = cv2.projectPoints(obj, rvec, tvec, Kv, Dv)
    err_v = np.linalg.norm(proj_v.reshape(-1, 2) - pts_v, axis=1)
    Rt, _ = cv2.Rodrigues(rvec)
    print(f"托盘平面位姿：solvePnP 重投影误差 中位 {np.median(err_v):.2f}px  "
          f"相机到托盘左上角距离 {np.linalg.norm(tvec):.0f}mm")

    # ---- 转到侧视相机并投影
    def to_side(p_mm: np.ndarray) -> np.ndarray:
        """托盘毫米坐标 (N,3) → 侧视相机像素 (N,2)。"""
        x_v = (Rt @ p_mm.T + tvec).T                     # 垂直相机系
        x_s = (R @ x_v.T + T).T                          # 侧视相机系
        px, _ = cv2.projectPoints(x_s.astype(np.float64), np.zeros(3), np.zeros(3), Ks, Ds)
        return px.reshape(-1, 2)

    corners3 = np.float32([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]])
    proj_s = to_side(corners3)
    print(f"\n投影到侧视相机的托盘四角：")
    for name, p in zip(("TL", "TR", "BR", "BL"), proj_s):
        print(f"  {name} ({p[0]:7.1f},{p[1]:7.1f})")

    # ---- 侧视帧：找黑点，和投影对比
    dots_s = find_dots(cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY), ksize=81,
                       circ_min=0.30, aspect=(0.3, 3.3))
    print(f"\n侧视帧黑点候选 {len(dots_s)} 个")
    print("投影点 vs 侧视实测点（最近邻匹配）：")
    residuals = []
    for name, p in zip(("TL", "TR", "BR", "BL"), proj_s):
        if not dots_s:
            break
        best = min(dots_s, key=lambda c: np.linalg.norm(c["center"] - p))
        d = float(np.linalg.norm(best["center"] - p))
        residuals.append(d)
        flag = "✓" if d < 15 else "✗"
        print(f"  {flag} {name} 投影 ({p[0]:7.1f},{p[1]:7.1f})  实测 "
              f"({best['center'][0]:7.1f},{best['center'][1]:7.1f})  差 {d:5.1f}px  "
              f"(候选 面积 {best['area']} 圆度 {best['circ']:.2f})")
    if residuals:
        print(f"\n一致性：残差 中位 {np.median(residuals):.1f}px  最大 {max(residuals):.1f}px  "
              f"（<15px 说明内参/外参/托盘平面三者一致）")

    # ---- 出图：侧视画面叠投影网格 + 轮廓 + 点
    vis = frame_s.copy()
    for mm in np.arange(0, W + 1e-6, 20.0):
        seg = to_side(np.float32([[mm, 0, 0], [mm, H, 0]]))
        cv2.line(vis, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (0, 200, 255), 3)
    for mm in np.arange(0, H + 1e-6, 20.0):
        seg = to_side(np.float32([[0, mm, 0], [W, mm, 0]]))
        cv2.line(vis, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (0, 200, 255), 3)
    cv2.polylines(vis, [proj_s.astype(np.int32)], True, (0, 0, 255), 5)
    for name, p in zip(("TL", "TR", "BR", "BL"), proj_s):
        cv2.circle(vis, tuple(p.astype(int)), 14, (0, 255, 0), 4)
        vis = overlay_text.draw_text(vis, name, (int(p[0]) + 18, int(p[1]) - 46), 44, (0, 255, 0))
    for d in dots_s:
        c = d["center"].astype(int)
        cv2.drawMarker(vis, tuple(c), (255, 0, 255), cv2.MARKER_CROSS, 40, 4)
    vis = overlay_text.draw_text(
        vis, f"侧视相机：红框/黄网格 = 由内参+外参+托盘单应投影而来  "
             f"洋红十字 = 实测黑点  残差中位 {np.median(residuals) if residuals else float('nan'):.1f}px",
        (20, 20), 40, (0, 0, 255))
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n验收图 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
